# GraphAssistant — 学术文献知识库助手

基于 LangChain 1.0、FAISS、Neo4j 构建的本地学术文献知识库助手，支持文献入库、多轮问答、深度研究报告生成与 Idea 挖掘，提供 FastAPI 接口层和 Streamlit 可视化前端。

---

## 功能概览

### 文献管理
- 支持 PDF / DOCX / TXT 格式解析
- LLM 自动提取标题、作者、摘要、关键词、年份等元数据
- 文本分块后写入 FAISS 向量索引 + SQLite 元数据库 + Neo4j 图数据库
- 可选实体与关系抽取，构建学术知识图谱

### 检索模式（三选一）
| 模式 | 说明 |
|------|------|
| `semantic` | FAISS 向量语义检索 |
| `hybrid` | FAISS + BM25 融合，RRF 重排序 |
| `graph` | Neo4j 实体匹配 + Chunk 回溯，可附加 1-hop 邻居上下文 |

### 多轮问答
- 原生 Agent 架构，按 `thread_id` 管理多轮上下文
- 每轮回答附带来源引用与 token 用量统计
- 问答接口额外返回 `ll_keywords` / `hl_keywords`，便于观察当前查询的实体词与主题词

### 查询关键词提取（Phase 4.1）
- 新增 `src/retrieval/keyword_extractor.py`
- 规则优先提取低层关键词（实体、模型、数据集、方法名）和高层关键词（趋势、主题、方向）
- 规则失效时自动回退到 LLM 结构化抽取
- `QAAgent`、`DeepResearchAgent`、`GraphRetriever` 已接入该提取器

### 关系向量索引（Phase 4.2）
- 新增 `src/storage/relation_vector_store.py`
- 为每条关系构造 `source + relation_type + target + description` 检索文本
- 与 chunk 向量索引独立保存，默认目录为 `data/relation_faiss`
- 文档删除或增量重建时，关系索引会按 `doc_id` 同步清理

### Local / Global / Mix 检索（Phase 4.3）
- `local`：融合图检索和语义检索，适合实体、方法、数据集等具体问题
- `global`：优先走关系向量索引和图关系，适合趋势、主题、方向等宏观问题
- `mix`：综合 `semantic + local + global` 三路结果做统一融合
- 前端、CLI、API 已全部支持这三种新模式

### 深度研究（Plan-Execute-Report）
1. LLM 将研究问题拆解为 3~5 个子问题
2. 对每个子问题独立检索 + 局部分析
3. 可选附加社区视角（Louvain 社区摘要）
4. 汇总生成完整 Markdown 研究报告
5. 可一键生成结构化 Idea 报告（Research Gaps / 方法对比 / 建议方向）

### 社区检测
- 基于 NetworkX + python-louvain 对实体关系图做 Louvain 社区划分
- LLM 为每个社区生成 100 字主题摘要，写回 Neo4j

### 接口与前端
- **FastAPI**：RESTful 接口，覆盖文献管理、问答、研究、Idea、社区检测
- **Streamlit**：四页可视化前端，侧边栏导航，支持报告下载

---

## 环境依赖

- Python 3.10+
- Conda 环境：`llm_universe`
- 阿里云 DashScope API Key（Qwen 系列模型 + text-embedding-v3）
- Neo4j 4.x / 5.x（本地或远程）

安装依赖：
```bash
F:/Anaconda/envs/llm_universe/python.exe -m pip install -r requirements.txt
```

配置 `.env`（参考 `.env.example`）：
```env
API_KEY=your_dashscope_api_key
MODEL_NAME=qwen-plus
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL_NAME=text-embedding-v3
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
APP_HOST=127.0.0.1
APP_PORT=8000
```

---

## 启动方式

### 方式一：Streamlit 前端（推荐）

需要开两个终端：

**终端 1 — 启动 FastAPI 后端：**
```bash
F:/Anaconda/envs/llm_universe/python.exe c:/Users/Administrator/Desktop/GraphAssitant/main.py
```

**终端 2 — 启动 Streamlit 前端：**
```bash
F:/Anaconda/envs/llm_universe/python.exe -m streamlit run c:/Users/Administrator/Desktop/GraphAssitant/app.py
```

浏览器访问 `http://localhost:8501`

---

### 方式二：命令行

**文献入库：**
```bash
# 入库单文件
F:/Anaconda/envs/llm_universe/python.exe ingest.py --file "C:/path/to/paper.pdf"

# 入库整个目录
F:/Anaconda/envs/llm_universe/python.exe ingest.py --dir "C:/path/to/papers/"

# 同时启用实体抽取
F:/Anaconda/envs/llm_universe/python.exe ingest.py --dir "C:/path/to/papers/" --enable-entity-extraction
```

**多轮问答：**
```bash
F:/Anaconda/envs/llm_universe/python.exe chat.py --thread-id t1 --top-k 3 --retriever-mode hybrid
```

**深度研究：**
```bash
# 基础研究报告
F:/Anaconda/envs/llm_universe/python.exe research.py --question "AmpAgent解决了什么问题，其方法有哪些局限？" --retriever-mode hybrid

# 附加 Idea 报告
F:/Anaconda/envs/llm_universe/python.exe research.py --question "..." --with-idea

# 保存报告到文件
F:/Anaconda/envs/llm_universe/python.exe research.py --question "..." --save-report --output reports/report.md
```

---

### 方式三：FastAPI 接口

启动后访问 `http://127.0.0.1:8000/docs` 查看 Swagger 交互文档。

主要接口：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/documents` | 文献列表 |
| POST | `/documents/ingest-file` | 入库单文件 |
| POST | `/documents/ingest-directory` | 入库目录 |
| POST | `/chat` | 多轮问答 |
| POST | `/research` | 深度研究 |
| POST | `/idea` | Idea 报告生成 |
| POST | `/community/detect` | 触发社区检测 |
| GET | `/community/list` | 社区摘要列表 |

`/chat` 响应新增字段：
- `ll_keywords`：当前问题抽取出的低层关键词
- `hl_keywords`：当前问题抽取出的高层关键词

---

## 目录结构

```
GraphAssistant/
├── .env                        # 环境变量配置（不提交）
├── app.py                      # Streamlit 前端入口
├── main.py                     # FastAPI 后端入口
├── ingest.py                   # 文献入库 CLI
├── chat.py                     # 多轮问答 CLI
├── research.py                 # 深度研究 CLI
│
├── src/                        # 核心业务逻辑
│   ├── config.py               # 配置管理（读取 .env）
│   ├── llm_client.py           # LLM 客户端封装（DashScope/Qwen）
│   ├── document_parser.py      # 文档解析（PDF/DOCX/TXT）
│   ├── chunker.py              # 文本分块
│   ├── metadata_extractor.py   # LLM 元数据提取
│   ├── database.py             # SQLite 元数据存储（SQLAlchemy ORM）
│   ├── embedder.py             # 向量化（text-embedding-v3）
│   ├── vector_store.py         # FAISS 向量索引
│   ├── ingestion_pipeline.py   # 文献入库流水线
│   ├── graph_store.py          # Neo4j 图存储
│   ├── entity_extractor.py     # 实体与关系抽取
│   ├── retriever.py            # 语义检索器
│   ├── bm25_retriever.py       # BM25 稀疏检索器
│   ├── hybrid_retriever.py     # 混合检索（RRF 融合）
│   ├── graph_retriever.py      # 图检索（Neo4j 实体匹配）
│   ├── token_tracker.py        # Token 用量统计与费用估算
│   ├── community_detector.py   # Louvain 社区检测
│   └── agents/
│       ├── base_agent.py       # Agent 基类（多轮会话管理）
│       ├── qa_agent.py         # 多轮问答 Agent
│       ├── deep_research_agent.py  # 深度研究 Agent
│       └── idea_agent.py       # Idea 生成 Agent
│
├── api/                        # FastAPI 接口层
│   ├── schemas.py              # 请求/响应 Pydantic 模型
│   └── routers/
│       ├── documents.py        # 文献管理接口
│       ├── chat.py             # 问答接口
│       ├── research.py         # 研究与 Idea 接口
│       └── community.py        # 社区检测接口
│
├── ui/                         # Streamlit 页面组件
│   ├── page_documents.py       # 文献管理页
│   ├── page_chat.py            # 多轮问答页
│   ├── page_research.py        # 深度研究页
│   └── page_community.py       # 社区检测页
│
├── tests/                      # 测试脚本
│   ├── test_connection.py      # LLM 连通性测试
│   ├── test_parser.py          # 文档解析测试
│   ├── test_metadata.py        # 元数据提取测试
│   ├── test_database.py        # SQLite CRUD 测试
│   ├── test_vector_search.py   # 向量检索测试
│   ├── test_graph.py           # Neo4j 写入测试
│   ├── test_entity_extraction.py  # 实体抽取测试
│   ├── test_hybrid_search.py   # 四种检索模式对比
│   ├── test_community.py       # 社区检测测试
│   ├── test_research.py        # 深度研究端到端测试
│   ├── test_api.py             # FastAPI 接口验证
│   └── test_keyword_extractor.py  # Phase 4.1 关键词提取测试
│
├── data/                       # 本地数据（gitignore）
│   ├── faiss/                  # FAISS 索引文件
│   └── metadata.db             # SQLite 数据库
│
├── reports/                    # 研究报告落盘目录
└── doc/
    └── 开发计划.md
```

---

## 技术栈

| 层次 | 技术 |
|------|------|
| LLM | 阿里云 DashScope / Qwen（via LangChain 1.0 ChatOpenAI） |
| 向量检索 | FAISS + text-embedding-v3（1024 维） |
| 稀疏检索 | BM25Okapi（rank-bm25） |
| 图数据库 | Neo4j + Python Driver |
| 社区检测 | NetworkX + python-louvain |
| 元数据存储 | SQLite + SQLAlchemy 2.0 ORM |
| 后端接口 | FastAPI + uvicorn |
| 前端 | Streamlit |
| 文档解析 | PyMuPDF（PDF）、python-docx（DOCX） |

## 效果图
![alt text](/assets/image.png)
![alt text](/assets/image1.png)
![alt text](/assets/image2.png)
