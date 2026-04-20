# token_tracker.py

## 所在层次
基础设施层 `src/infrastructure/`

## 主体功能
Token 用量追踪，从 LangChain `AIMessage.response_metadata` 中提取 token 用量，估算人民币费用。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `TokenUsage` (dataclass) | 用量记录：`prompt_tokens / completion_tokens / total_tokens / model_name / estimated_cost_cny` |
| `TokenUsage.from_langchain_message(message, model_name)` | 从 AIMessage 提取 token 用量，按模型定价估算费用 |

## 调用关系
- **被调用方**：`QAAgent`、`DeepResearchAgent`（每次 LLM 调用后提取用量）
- **依赖方**：无外部依赖

## 注意事项
- 定价表 `_PRICE_PER_1K_TOKENS` 按模型名查找，未知模型使用默认值 `0.0005 CNY/1K`
- `MasterAgent` 有独立的 token 统计逻辑（`_accumulate_tokens`），不使用此模块
- `response_metadata` 字段名因 provider 不同可能有差异（`token_usage` vs 直接字段）
