import os
import operator
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from FlagEmbedding import BGEM3FlagModel, FlagReranker
import chromadb
from neo4j import GraphDatabase

# ==========================================
# 1. 定义整个工作流的“状态” (State)
# ==========================================
class GraphRAGState(TypedDict):
    original_query: str                                        # 用户原始问题
    rewritten_query: str                                       # LLM重写后的问题
    route_type: str                                            # 路由决策
    retrieved_contexts: Annotated[List[str], operator.add]     # 多节点结果自动合并
    retrieved_metadatas: Annotated[List[dict], operator.add]   # 对应的元数据（文档名等）
    final_answer: str                                          # 最终生成的答案

# ==========================================
# 2. 初始化核心组件 (大模型、数据库、Reranker)
# ==========================================
# 建议在实际项目中把这些封装成类，这里为了展示逻辑直观呈现
llm = ChatOpenAI(
    temperature=0,
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL_NAME"),
)
embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, devices="cuda:0")  # 与链路一保持一致
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, devices="cuda:0")

# 连接链路一建好的 Chroma 向量库
_RAG_DIR = os.path.dirname(os.path.abspath(__file__))
chroma_client = chromadb.PersistentClient(path=os.path.join(_RAG_DIR, "chroma_db"))
_vector_collection = None

def _get_vector_collection():
    global _vector_collection
    if _vector_collection is None:
        _vector_collection = chroma_client.get_collection(name="llm_knowledge_chunks")
    return _vector_collection

# 连接 Neo4j 图数据库 (可选)
try:
    neo4j_driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI",  "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PWD", "password")),
    )
    neo4j_driver.verify_connectivity()
    _neo4j_available = True
except Exception as _neo4j_err:
    print(f"⚠️  Neo4j 不可用，图谱检索将返回空结果: {_neo4j_err}")
    neo4j_driver = None
    _neo4j_available = False

# ==========================================
# 3. 定义 LangGraph 的各个节点 (Nodes)
# ==========================================

def query_analysis_node(state: GraphRAGState) -> GraphRAGState:
    """节点 1：查询分析与路由决策"""
    print("--- [节点] 查询分析与路由 ---")
    query = state["original_query"]
    
    # 简单的 Prompt，让 LLM 决定怎么查
    prompt = ChatPromptTemplate.from_template(
        "你是一个大模型领域的知识路由专家。请分析以下用户问题：\n"
        "问题：{query}\n"
        "如果问题是关于概念原理、技术细节、长文本描述，请回复 'vector_only'。\n"
        "如果问题是关于实体关系（如谁提出了什么、某模型基于什么架构、属于哪个机构），请回复 'graph_only'。\n"
        "如果是复杂的综合问题，回复 'hybrid'。\n"
        "仅输出这三个词中的一个。"
    )
    chain = prompt | llm
    route_result = chain.invoke({"query": query}).content.strip().lower()
    if route_result not in ("vector_only", "graph_only", "hybrid"):
        route_result = "vector_only"
    
    print(f"  路由决策: {route_result}")
    # 这里也可以顺便做 Query Rewrite，为了代码简洁暂略
    return {"route_type": route_result, "rewritten_query": query}

def vector_retrieval_node(state: GraphRAGState) -> GraphRAGState:
    """节点 2A：向量检索 (从 ChromaDB 捞取文本块)"""
    print("--- [节点] 向量数据库检索 ---")
    query = state["rewritten_query"]
    
    # 用 BGE-M3 编码 query，与链路一写入时保持一致
    query_vec = embedding_model.encode([query])['dense_vecs'].astype('float32').tolist()
    results = _get_vector_collection().query(query_embeddings=query_vec, n_results=5)
    contexts = results['documents'][0] if results['documents'] else []
    metadatas = results['metadatas'][0] if results['metadatas'] else [{} for _ in contexts]
    
    print(f"  向量检索命中 {len(contexts)} 条，来源文档:")
    for i, (ctx, meta) in enumerate(zip(contexts, metadatas)):
        doc = meta.get('doc_name', '未知')
        print(f"    [{i+1}] 📄 {doc} | {ctx[:80].replace(chr(10), ' ')}...")
    
    return {"retrieved_contexts": contexts, "retrieved_metadatas": metadatas}

def graph_retrieval_node(state: GraphRAGState) -> GraphRAGState:
    """节点 2B：图谱检索 (用 LLM 提取关键词，在 Neo4j 中做全文匹配)"""
    print("--- [节点] 图数据库检索 ---")
    query = state["rewritten_query"]
    contexts = []
    metadatas = []

    if not _neo4j_available:
        print("⚠️  Neo4j 不可用，图谱检索返回空结果")
        return {"retrieved_contexts": contexts, "retrieved_metadatas": metadatas}

    # Step 1: 用 LLM 从 query 中提取 2~4 个核心关键词（英文/中文均可）
    kw_prompt = ChatPromptTemplate.from_template(
        "请从以下问题中提取 2~4 个最核心的英文技术关键词，用于在论文数据库中做全文检索。"
        "只输出关键词列表，用英文逗号分隔，不要输出其他内容。\n问题：{query}"
    )
    kw_result = (kw_prompt | llm).invoke({"query": query}).content.strip()
    keywords = [k.strip() for k in kw_result.replace("，", ",").split(",") if k.strip()][:4]
    print(f"  提取关键词: {keywords}")

    # Step 2: 对每个关键词在 Neo4j 的 Chunk.text 中做 CONTAINS 匹配，取相关片段
    seen = set()
    with neo4j_driver.session() as session:
        for kw in keywords:
            cypher = (
                "MATCH (c:Chunk)-[:PART_OF]->(d:Document) "
                "WHERE toLower(c.text) CONTAINS toLower($kw) "
                "RETURN c.text, d.name LIMIT 3"
            )
            result = session.run(cypher, kw=kw)
            for record in result:
                text = record['c.text']
                doc_name = record['d.name']
                key = text[:80]
                if key in seen:
                    continue
                seen.add(key)
                contexts.append(text)
                metadatas.append({"doc_name": doc_name, "source": "graph", "keyword": kw})

    print(f"  图谱检索命中 {len(contexts)} 条，来源文档:")
    for i, (ctx, meta) in enumerate(zip(contexts, metadatas)):
        doc = meta.get('doc_name', '未知')
        kw = meta.get('keyword', '')
        print(f"    [{i+1}] 🔑 {kw} | 📄 {doc} | {ctx[:80].replace(chr(10), ' ')}...")

    return {"retrieved_contexts": contexts, "retrieved_metadatas": metadatas}

def rerank_node(state: GraphRAGState) -> GraphRAGState:
    """节点 3：重排序 (去除噪声，保留最相关的信息)"""
    print("--- [节点] BGE-Reranker 重排序 ---")
    query = state["rewritten_query"]
    contexts = state.get("retrieved_contexts", [])
    metadatas = state.get("retrieved_metadatas", [{} for _ in contexts])
    
    if not contexts:
        return state
        
    # 构建 pairs 给 BGE-Reranker 打分
    pairs = [[query, ctx] for ctx in contexts]
    scores = reranker.compute_score(pairs)
    if not isinstance(scores, list):
        scores = [scores]
    
    # 根据得分从高到低排序，过滤掉得分过低的噪声，这里保留 Top 3
    scored = sorted(zip(contexts, metadatas, scores), key=lambda x: x[2], reverse=True)
    
    print(f"  重排打分结果 (共 {len(scored)} 条):")
    for i, (ctx, meta, score) in enumerate(scored):
        doc = meta.get('doc_name', '未知')
        marker = "✅" if i < 3 else "❌"
        print(f"    {marker} [{i+1}] score={score:.4f} | 📄 {doc} | {ctx[:60].replace(chr(10), ' ')}...")
    
    top_contexts = [ctx for ctx, meta, score in scored[:3]]
    top_metadatas = [meta for ctx, meta, score in scored[:3]]
    
    print(f"  重排后保留了 {len(top_contexts)} 条高质量参考资料。")
    return {"retrieved_contexts": top_contexts, "retrieved_metadatas": top_metadatas}

def generation_node(state: GraphRAGState) -> GraphRAGState:
    """节点 4：最终大模型生成"""
    print("--- [节点] 组装 Prompt 与生成答案 ---")
    query = state["original_query"]
    contexts_list = state["retrieved_contexts"]
    metadatas_list = state.get("retrieved_metadatas", [{} for _ in contexts_list])
    
    # 构建带编号和来源标注的参考资料块
    numbered_contexts = []
    for i, (ctx, meta) in enumerate(zip(contexts_list, metadatas_list), 1):
        doc = meta.get('doc_name', '未知来源')
        numbered_contexts.append(f"[{i}] 来源：{doc}\n{ctx}")
    contexts_str = "\n\n".join(numbered_contexts)
    
    print(f"  输入参考资料 {len(numbered_contexts)} 条:")
    for i, (ctx, meta) in enumerate(zip(contexts_list, metadatas_list), 1):
        doc = meta.get('doc_name', '未知来源')
        print(f"    [{i}] 📄 {doc}")
        print(f"        {ctx[:120].replace(chr(10), ' ')}...")
    
    prompt = ChatPromptTemplate.from_template(
        "你是一个顶尖的计算机科学与大模型技术研究助手。请严格根据以下带编号的参考资料回答用户问题。\n"
        "要求：\n"
        "1. 回答中每个关键论断必须以 [编号] 的形式标注来源，例如 [1][2]。\n"
        "2. 尽量引用原文中的关键表述，用引号括起来。\n"
        "3. 如果参考资料中无法得出答案，请诚实说明，不要编造。\n\n"
        "【参考资料】:\n{contexts}\n\n"
        "【用户问题】: {query}\n"
        "【你的回答】:"
    )
    chain = prompt | llm

    # 流式输出
    print("  ⏳ 生成中（流式）：")
    full_answer = ""
    for chunk in chain.stream({"contexts": contexts_str, "query": query}):
        token = chunk.content
        print(token, end="", flush=True)
        full_answer += token
    print()  # 换行

    return {"final_answer": full_answer}

# ==========================================
# 4. 定义条件路由逻辑 (Conditional Edges)
# ==========================================
def route_query(state: GraphRAGState) -> str:
    """根据路由分析结果，决定走向哪个检索节点"""
    route = state["route_type"]
    if route == "graph_only":
        return "graph_retrieval_node"
    else:
        # vector_only 和 hybrid 都先走向量检索
        return "vector_retrieval_node"

# ==========================================
# 5. 构建与编译 LangGraph 工作流
# ==========================================
workflow = StateGraph(GraphRAGState)

# 添加节点
workflow.add_node("query_analysis", query_analysis_node)
workflow.add_node("vector_retrieval_node", vector_retrieval_node)
workflow.add_node("graph_retrieval_node", graph_retrieval_node)
workflow.add_node("rerank_node", rerank_node)
workflow.add_node("generation_node", generation_node)

# 定义边 (流程流转)
workflow.set_entry_point("query_analysis")

# 动态路由：分析完问题后，去查哪个库？
workflow.add_conditional_edges(
    "query_analysis",
    route_query,
    {
        "vector_retrieval_node": "vector_retrieval_node",
        "graph_retrieval_node": "graph_retrieval_node",
    }
)

# hybrid：向量检索后继续走图谱检索；vector_only/graph_only 直接汇聚到重排
workflow.add_conditional_edges(
    "vector_retrieval_node",
    lambda s: "graph_retrieval_node" if s["route_type"] == "hybrid" else "rerank_node",
    {"graph_retrieval_node": "graph_retrieval_node", "rerank_node": "rerank_node"}
)
workflow.add_edge("graph_retrieval_node", "rerank_node")

# 重排后生成最终答案，然后结束
workflow.add_edge("rerank_node", "generation_node")
workflow.add_edge("generation_node", END)

# 编译图
app = workflow.compile()

# ==========================================
# 运行示例
# ==========================================
if __name__ == "__main__":
    test_query = "目前主流的多模态对齐算法有哪些？它们分别优化了哪些缺陷？"
    
    print(f"用户提问: {test_query}\n")
    
    # 运行 LangGraph
    final_state = app.invoke({"original_query": test_query})
    
    print("\n================ 最终回答 ================\n")
    print(final_state["final_answer"])