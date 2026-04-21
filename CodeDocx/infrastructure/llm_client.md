# llm_client.py

## 所在层次
基础设施层 `src/infrastructure/`

## 主体功能
LLM 客户端双工厂，分别返回文本 LLM 和视觉 LLM 实例（兼容 OpenAI 接口的任意模型）。
使用 `lru_cache` 缓存实例，避免重复创建。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `get_native_llm(temperature)` | 文本 LLM 工厂，从 `settings.model_name` 创建实例，返回 dict 格式响应 |
| `get_vision_llm(temperature)` | 视觉 LLM 工厂，从 `settings.vision_model_name` 创建实例，用于图片/表格描述 |

## 调用关系
- **被调用方**：几乎所有需要 LLM 的模块（`MetadataExtractor`、`EntityExtractor`、`MasterAgent`、`modal_processors` 等）
- **依赖方**：`src/infrastructure/config.settings`

## 注意事项
- `api_key` 未配置时抛出 `ValueError`，不静默失败
- `get_vision_llm()` 使用 `settings.vision_model_name`（默认 `qwen-vl-plus`），与文本 LLM 独立配置
- `temperature=0.0` 用于结构化提取，`0.1~0.7` 用于创意生成
