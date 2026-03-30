"""
run_test.py — 自动化测试脚本

5 条覆盖不同检索路径的测试问题，调用 run_chat.ask() 批量运行并打印结果。
"""
import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from run_chat import ingest_if_needed, ask

TEST_QUERIES = [
    # vector_only：概念原理类
    "Transformer 的自注意力机制是如何计算的？它相比 RNN 有哪些优势？",
    # vector_only：技术细节类
    "FlashAttention 是如何解决标准注意力机制显存瓶颈的？",
    # graph_only：实体关系类
    "LoRA 是由哪个机构提出的？它基于什么原理对大模型进行微调？",
    # hybrid：演进路径类
    "从 GPT-3 到 InstructGPT，RLHF 解决了哪些问题？对齐技术是如何演进的？",
    # hybrid：多跳推理类
    "GraphRAG 相比传统 RAG 有哪些核心改进？它在多跳推理场景下的优势是什么？",
]


def main():
    ingest_if_needed()

    print("=" * 60)
    print(f"GraphRAG 自动测试 — 共 {len(TEST_QUERIES)} 条问题")
    print("=" * 60)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n【问题 {i}/{len(TEST_QUERIES)}】{query}")
        print("-" * 60)
        ask(query)
        # 回答已在 generation_node 中流式打印，此处无需重复输出
        print("=" * 60)


if __name__ == "__main__":
    main()
