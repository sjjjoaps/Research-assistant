# 基础设施层（infrastructure）

目录：`src/infrastructure/`

## 模块清单

| 文件 | 主要类/函数 | 职责 |
|---|---|---|
| `config.py` | `Settings`, `settings` | 全局配置（环境变量、路径、模型参数、多模态开关） |
| `llm_client.py` | `get_native_llm()`, `get_vision_llm()` | LLM 客户端双工厂（文本 + 视觉），lru_cache 复用 |
| `embedder.py` | `get_embedder()`, `Embedder` | 文本嵌入，支持 OpenAI 兼容接口 |
| `token_tracker.py` | `TokenTracker` | Token 用量统计与费用估算 |
| `resilience.py` | 重试装饰器、熔断机制 | LLM 调用重试与熔断 |
| `long_term_memory.py` | `LongTermMemory`, `_slugify()` | 长期记忆读写（Markdown 文件 + MEMORY.md 索引） |
| `json_utils.py` | `extract_json()` | JSON 提取（兼容 markdown 包裹、嵌套对象、列表） |
