# reranker.py

## 所在层次
检索层 `src/retrieval/`

## 主体功能
封装外部 Reranker API 调用，对 RRF 融合后的候选 chunk 进行精排。
API 失败时自动降级，保留原始 RRF 顺序的 top_n 个结果。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `APIReranker.__init__(api_url, api_key, model, top_n, timeout, max_candidates)` | 初始化 Reranker 配置参数 |
| `APIReranker.rerank(query, chunks)` | 精排入口：截断超量候选 → 调用 API → 失败时降级 |
| `APIReranker._call_api(query, chunks)` | 发送 POST 请求，解析 `results[].relevance_score` 重排序 |
| `get_reranker()` | `@lru_cache` 单例工厂，`RERANKER_ENABLED=false` 时返回 `None` |

## 调用关系
- **被调用方**：`HybridRetriever`、`MixRetriever`（通过 `get_reranker()` 获取单例）
- **依赖方**：外部 Reranker API（兼容 HuggingFace TEI / SiliconFlow / Jina 标准接口）、`src/infrastructure/config.py`

## 注意事项
- 请求格式：`{"model": "...", "query": "...", "documents": ["text1", ...]}`
- 响应格式：`{"results": [{"index": 0, "relevance_score": 0.95}, ...]}`
- 降级策略：API 失败时返回 `chunks[:top_n]`（RRF 原始顺序），不再走第二次 RRF
- `max_candidates=50` 防止超大候选集导致 API 超时
- `get_reranker()` 使用 `@lru_cache(maxsize=1)` 保证全局单例，配置变更需重启服务
