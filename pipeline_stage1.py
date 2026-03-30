import os
import uuid
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# LangChain 切分工具
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 向量化模型
from FlagEmbedding import BGEM3FlagModel

# 数据库
import chromadb
from neo4j import GraphDatabase

class OfflineDataPipeline:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_pwd, chroma_path=None, embedding_model=None):
        if chroma_path is None:
            chroma_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
        print("初始化流水线...")
        # 1. 初始化 BGE-M3 向量模型 (支持多语言和长文本)
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        
        # 2. 初始化 ChromaDB 向量数据库
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.vector_collection = self.chroma_client.get_or_create_collection(name="llm_knowledge_chunks")
        
        # 3. 初始化 Neo4j 图数据库连接 (可选，不可用时跳过图谱写入)
        try:
            self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pwd))
            self.neo4j_driver.verify_connectivity()
            self.neo4j_available = True
        except Exception as e:
            print(f"⚠️  Neo4j 不可用，跳过图谱写入: {e}")
            self.neo4j_driver = None
            self.neo4j_available = False
        
        print("所有组件初始化完成！")

    def parse_pdf_with_mineru(self, pdf_path: str, output_dir: str) -> str:
        """
        步骤 1: 使用 pymupdf (fitz) 将 PDF 解析为 Markdown 文本。
        """
        import fitz
        print(f"正在解析 PDF: {pdf_path}")
        base_name = os.path.basename(pdf_path).replace(".pdf", "")
        md_file_path = os.path.join(output_dir, "magic-pdf", base_name, "auto", f"{base_name}.md")
        
        if not os.path.exists(md_file_path):
            os.makedirs(os.path.dirname(md_file_path), exist_ok=True)
            doc = fitz.open(pdf_path)
            md_lines = []
            for page in doc:
                blocks = page.get_text("blocks")
                for b in sorted(blocks, key=lambda x: (x[1], x[0])):
                    text = b[4].strip()
                    if text:
                        md_lines.append(text)
                md_lines.append("")
            doc.close()
            md_text = "\n".join(md_lines)
            with open(md_file_path, "w", encoding="utf-8") as f:
                f.write(md_text)
        
        with open(md_file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def chunk_markdown(self, markdown_text: str) -> List[Dict]:
        """
        步骤 2: 将 Markdown 文本按标题和长度进行切块 (Chunking)
        """
        print("正在进行文本切块...")
        # 首先按 Markdown 标题切分，保留逻辑结构
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(markdown_text)
        
        # 如果某个标题下的内容依然过长，再用递归字符切分器进行细切
        chunk_size = 500
        chunk_overlap = 50
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        final_splits = text_splitter.split_documents(md_header_splits)
        
        chunks_data = []
        for i, split in enumerate(final_splits):
            chunk_id = str(uuid.uuid4())
            chunks_data.append({
                "chunk_id": chunk_id,
                "text": split.page_content,
                "metadata": split.metadata # 包含 Header 1/2/3 信息
            })
        return chunks_data

    def ingest_to_databases(self, doc_name: str, chunks_data: List[Dict]):
        """
        步骤 3 & 4: 向量化并进行双库写入 (Chroma + Neo4j)
        """
        print(f"正在向量化并入库，共 {len(chunks_data)} 个 Chunks...")
        texts = [c["text"] for c in chunks_data]
        ids = [c["chunk_id"] for c in chunks_data]
        metadatas = [{"doc_name": doc_name, **c["metadata"]} for c in chunks_data]
        
        # --- A. 向量化 (BGE-M3) ---
        # BGE-M3 返回一个字典，dense_vecs 是稠密向量
        embeddings = self.embedding_model.encode(texts, batch_size=12, max_length=1024)['dense_vecs'].astype('float32')
        
        # --- B. 写入 ChromaDB ---
        self.vector_collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        # --- C. 写入 Neo4j (创建文档与块的层级关系) ---
        if not self.neo4j_available:
            print("⚠️  跳过 Neo4j 写入（服务不可用）")
            print(f"文档 {doc_name} 入库完成！")
            return
        doc_id = str(uuid.uuid4())
        with self.neo4j_driver.session() as session:
            # 1. 创建源文档节点
            session.run(
                "MERGE (d:Document {name: $doc_name}) SET d.id = $doc_id",
                doc_name=doc_name, doc_id=doc_id
            )
            # 2. 批量创建 Chunk 节点并连接到源文档
            session.run(
                """
                MATCH (d:Document {name: $doc_name})
                UNWIND $chunks AS chunk
                CREATE (c:Chunk {id: chunk.chunk_id, text: chunk.text})
                CREATE (c)-[:PART_OF]->(d)
                """,
                doc_name=doc_name,
                chunks=[{"chunk_id": c["chunk_id"], "text": c["text"]} for c in chunks_data]
            )
        print(f"文档 {doc_name} 入库完成！")

    def run(self, pdf_path: str):
        """执行完整流水线"""
        doc_name = os.path.basename(pdf_path)
        output_dir = "./parsed_md"
        
        # 1. 解析
        md_text = self.parse_pdf_with_mineru(pdf_path, output_dir)
        # 2. 切块
        chunks = self.chunk_markdown(md_text)
        # 3. 入库
        self.ingest_to_databases(doc_name, chunks)
        print("✅ 链路一处理完毕。")

    def close(self):
        if self.neo4j_driver is not None:
            self.neo4j_driver.close()

# --- 运行示例 ---
if __name__ == "__main__":
    # 配置你的 Neo4j 账号密码
    pipeline = OfflineDataPipeline(
        neo4j_uri=os.getenv("NEO4J_URI",  "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_pwd=os.getenv("NEO4J_PWD",  "password"),
    )
    
    # 假设你有一篇大模型论文的 PDF
    # pipeline.run("./data/attention_is_all_you_need.pdf")
    
    pipeline.close()