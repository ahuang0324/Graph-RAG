"""
run_chat.py — GraphRAG 核心运行入口

逻辑：
  1. 扫描 ./data/ 下有无未入库的 PDF
  2. 有 → 执行链路一（解析+切块+向量/图谱写库）
  3. 执行链路三（在线检索+问答）
"""
import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import glob
from dotenv import load_dotenv

load_dotenv()

from pipeline_stage1 import OfflineDataPipeline
from pipeline_stage2 import app, embedding_model   # 已编译好的 LangGraph app，复用 embedding_model

DATA_DIR   = "./data"
PARSED_DIR = "./parsed_md"

NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PWD  = os.getenv("NEO4J_PWD")


def _get_unindexed_pdfs() -> list[str]:
    """返回 data/ 下尚未生成对应 parsed_md 的 PDF 路径列表。"""
    all_pdfs = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    unindexed = []
    for pdf in all_pdfs:
        base = os.path.basename(pdf).replace(".pdf", "")
        md_path = os.path.join(PARSED_DIR, "magic-pdf", base, "auto", f"{base}.md")
        if not os.path.exists(md_path):
            unindexed.append(pdf)
    return unindexed


def ingest_if_needed():
    """检查并执行链路一：只处理尚未入库的 PDF。"""
    pdfs = _get_unindexed_pdfs()
    if not pdfs:
        print("✅ 所有 PDF 已入库，跳过链路一。")
        return

    print(f"🔍 发现 {len(pdfs)} 个未入库的 PDF，开始链路一处理...")
    pipeline = OfflineDataPipeline(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_pwd=NEO4J_PWD,
        embedding_model=embedding_model,
    )
    for pdf in pdfs:
        pipeline.run(pdf)
    pipeline.close()
    print("✅ 链路一完成。\n")


def ask(query: str) -> str:
    """对外暴露的问答接口：输入问题，返回答案字符串。"""
    final_state = app.invoke({"original_query": query})
    return final_state["final_answer"]


def chat_loop():
    """交互式问答循环。"""
    ingest_if_needed()
    print("=" * 50)
    print("GraphRAG 问答系统已就绪，输入 'q' 退出")
    print("=" * 50)
    while True:
        query = input("\n❓ 你的问题: ").strip()
        if query.lower() in ("q", "quit", "exit"):
            print("再见！")
            break
        if not query:
            continue
        print("\n⏳ 检索中...")
        answer = ask(query)
        print(f"\n💡 回答:\n{answer}")


if __name__ == "__main__":
    chat_loop()
