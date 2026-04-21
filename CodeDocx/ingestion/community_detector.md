# community_detector.py

## 所在层次
摄入层 `src/ingestion/`

## 主体功能
基于 Neo4j 中的 Entity-RELATES_TO 图构建 NetworkX 无向图，使用 Louvain 算法划分社区，
并为每个足够大的社区调用 LLM 生成摘要后写回 Neo4j。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `CommunityStats` (dataclass) | 统计信息：总实体数 / 检测社区数 / 写入数 / 跳过数 |
| `CommunityDetector.__init__(graph_store)` | 初始化 GraphStore 引用和 LLM 摘要链 |
| `CommunityDetector._build_graph(relations)` | 构建 NetworkX 无向图，同时维护 entity_info 映射 |
| `CommunityDetector.detect_and_write(min_size)` | 主入口：构图 → Louvain 划分 → 按社区生成摘要 → 写入 Neo4j |

## 调用关系
- **被调用方**：`src/workflows/ingestion_pipeline.py`（入库流水线社区检测阶段，可选）
- **依赖方**：`python-louvain (community)`、`networkx`、`src/storage/graph_store.GraphStore`、`src/agents/prompt_loader.load_prompt_pair("community_summary")`

## 注意事项
- `min_size` 参数控制社区最小节点数，过小的社区跳过不生成摘要
- Louvain 算法具有随机性，每次运行结果可能略有差异
- LLM 摘要调用 `temperature=0.1`，保证摘要一致性
