# schemas.py

## 所在层次
API 层 `api/`

## 主体功能
所有 FastAPI 路由的请求/响应 Pydantic 模型定义，是前后端数据契约的唯一来源。

## 关键类与方法

**文献管理**
| 类 | 作用 |
|---|---|
| `DocumentItem` | 文献完整信息（含状态字段），用于列表响应 |
| `DocumentStatusResponse` | 文档状态轮询响应（部分字段，不含元数据） |
| `IngestFileRequest` / `IngestDirectoryRequest` | 入库请求 |
| `IngestStartResponse` | 异步入库启动响应（含 `doc_id` 和初始状态） |
| `DeleteDocumentResult` | 删除结果（含各类型删除计数） |

**图谱可视化**
| 类 | 作用 |
|---|---|
| `GraphStatsResponse` | 图谱统计（节点数/关系数/标签分布） |
| `GraphNode` / `GraphEdge` | 图谱节点/边数据结构 |
| `GraphSubgraphResponse` | 子图查询响应 |

**问答与研究**
| 类 | 作用 |
|---|---|
| `ChatRequest` / `ChatResponse` | 传统 QA 接口（非流式） |
| `ResearchRequest` / `ResearchResponse` | 深度研究接口 |
| `IdeaRequest` / `IdeaResponse` | Idea 生成接口 |
| `AgentChatRequest` | MasterAgent 流式对话请求 |
| `SessionMeta` | 会话列表元数据（含 `total_cost_cny`） |

## 调用关系
- **被调用方**：所有 `api/routers/` 路由文件
- **依赖方**：`pydantic.BaseModel`

## 注意事项
- `DocumentStatusResponse` 是 `DocumentItem` 的子集（仅状态字段），前端轮询时用 `upsert` 合并而非替换
- `DocumentItem.status` 来自 `DocumentStatusStore`，与元数据字段分离
- `SessionMeta.total_cost_cny` 为 Phase 9-3 新增字段，旧会话默认 `0.0`
