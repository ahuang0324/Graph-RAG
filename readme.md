# GraphRAG

基于 LangGraph 构建的两阶段检索增强生成（RAG）流水线。离线阶段将 PDF 解析、切块后分别写入 ChromaDB（稠密向量）和 Neo4j（知识图谱）；在线阶段由 LangGraph 工作流对每条查询进行路由，走向量检索、图谱检索或混合检索，经 BGE-Reranker 重排后再送入大模型生成答案。

## 架构

```
离线阶段（链路一）
  PDF → PyMuPDF → Markdown → Chunking → BGE-M3 → ChromaDB
                                                 → Neo4j

在线阶段（链路二）
  Query → LLM 路由 → 向量检索（ChromaDB）
                   → 图谱检索（Neo4j）
                   → BGE-Reranker 重排
                   → LLM 流式生成
```

路由器将每条查询分类为 `vector_only`、`graph_only` 或 `hybrid`。混合模式会同时命中两个数据库，合并结果后再重排。

## 环境要求

- Python 3.10+
- 推荐 CUDA GPU（BGE-M3 和 BGE-Reranker 默认在 `cuda:0` 上运行）
- Neo4j 实例（可选，不可用时流水线会自动降级，跳过图谱写入/检索）
- 兼容 OpenAI 接口的 LLM 服务

## 安装依赖

```bash
pip install -r requirement.txt
```

BGE-M3 和 BGE-Reranker 的模型权重会在首次运行时从 Hugging Face 自动下载：

- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)

网络受限时可手动下载后放到本地路径，再将代码中的模型名替换为本地路径即可。

## 配置

在项目根目录创建 `.env` 文件，填入以下内容：

```
# 大模型（兼容 OpenAI 接口的任意服务均可）
API_KEY=your_api_key
BASE_URL=https://your-llm-endpoint/v1
MODEL_NAME=your-model-name

# Neo4j 图数据库（可选）
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PWD=your_password

# 内部依赖，保持不变
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

字段说明：

- `API_KEY` / `BASE_URL` / `MODEL_NAME`：LLM 服务的鉴权信息和模型标识，支持 OpenAI、火山引擎、DeepSeek 等任意兼容接口。
- `NEO4J_URI`：Neo4j 连接地址，支持本地 `bolt://` 或云端 `neo4j+s://`。不填或服务不可用时系统自动跳过图谱相关步骤。
- `NEO4J_USER` / `NEO4J_PWD`：Neo4j 用户名和密码。

## 运行流程

### 第一步：准备数据

将需要入库的 PDF 论文放到 `./data/` 目录下：

```bash
mkdir -p data
cp your_paper.pdf data/
```

### 第二步：启动问答

```bash
python run_chat.py
```

首次运行时，程序会自动检测 `./data/` 下未入库的 PDF，依次完成解析、切块、向量化、写库（链路一），然后进入交互式问答循环。后续运行会跳过已处理的文件，直接进入问答。

输入问题后回车，输入 `q` 退出。

### 第三步（可选）：批量测试

```bash
python run_test.py
```

内置 5 条覆盖不同路由路径的测试问题，批量运行并打印每条问题的检索过程与答案。

### 单独运行链路一（批量入库）

```python
from pipeline_stage1 import OfflineDataPipeline

pipeline = OfflineDataPipeline(neo4j_uri=..., neo4j_user=..., neo4j_pwd=...)
pipeline.run("./data/paper.pdf")
pipeline.close()
```

## 项目结构

```
.
├── pipeline_stage1.py   # 离线入库：PDF → ChromaDB + Neo4j
├── pipeline_stage2.py   # 在线检索：LangGraph 工作流
├── run_chat.py          # 交互式问答入口
├── run_test.py          # 批量测试脚本
├── data/                # 源 PDF（不纳入版本管理）
├── parsed_md/           # PDF 解析生成的中间 Markdown（不纳入版本管理）
├── chroma_db/           # ChromaDB 持久化文件（不纳入版本管理）
└── requirement.txt
```

## 模型

| 用途 | 模型 |
|------|------|
| 向量化 | BAAI/bge-m3 |
| 重排序 | BAAI/bge-reranker-v2-m3 |
| 生成 | 通过 `.env` 配置，支持任意兼容 OpenAI 接口的服务 |
