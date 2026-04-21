# keyword_extractor.py

## 所在层次
检索层 `src/retrieval/`

## 主体功能
两阶段查询关键词提取器，从用户查询中分离两类关键词：
- **ll_keywords**（低层）：实体、模型、数据集、方法名等精确术语
- **hl_keywords**（高层）：研究趋势、主题、方向等宏观语义词

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `KeywordResult` (dataclass) | 提取结果结构：`ll_keywords / hl_keywords / raw_query` |
| `KeywordExtractor.extract(query)` | 主入口：规则提取 → 若两类均为空则 LLM 兜底 |
| `KeywordExtractor._rule_extract(query)` | 正则短语提取 + 分类规则（无 LLM 调用） |
| `KeywordExtractor._extract_phrases(query)` | 多模式正则匹配（混合词、HL 短语、普通短语），按位置排序 |
| `KeywordExtractor._classify_phrase(phrase)` | 返回 "ll" / "hl" / None，基于后缀/词根/大写缩写等规则 |
| `KeywordExtractor._llm_extract(query)` | LLM 结构化抽取兜底，解析 JSON `{ll_keywords, hl_keywords}` |

## 调用关系
- **被调用方**：`GraphRetriever`、`GlobalRetriever`、`LightRAGDualRetriever`
- **依赖方**：`src/agents/prompt_loader.load_system_prompt()`（加载 `keyword_extractor_fallback` 提示词）、`src/infrastructure/llm_client.get_llm()`（LLM 兜底时懒加载）

## 注意事项
- LLM 懒加载：仅在规则提取失败时才创建 LLM 实例，避免无效开销
- 每类关键词上限 `_MAX_KEYWORDS_PER_GROUP = 8`，防止过多关键词稀释召回精度
- `_HL_ONLY_WORDS`（如 "trend"、"survey"）强制归为高层，不出现在 ll_keywords 中
- 中文数字/英文模型名、大写缩写（GPT、BERT 等）识别为低层关键词
