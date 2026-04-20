# time_filter.py

## 所在层次
检索层 `src/retrieval/`

## 主体功能
时间感知过滤器（Phase 9-2），为检索结果增加时间维度的过滤与自适应放宽。
借鉴 AgenticRAG Progressive Adaptive Retrieval 机制。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `extract_time_constraint(query)` | 从查询文本解析时间约束（支持近N年/年份范围/开放上限/模糊时间词） |
| `filter_by_time(chunks, constraint)` | 按时间约束过滤 chunk；`year=None` 的 chunk 默认保留 |
| `progressive_retrieve(chunks, constraint, min_count=3)` | 渐进式召回：严格过滤 → 放宽 year_from -2 → 放弃约束 |

## 调用关系
- **被调用方**：`tool_registry._do_retrieve()`（在检索器返回后应用时间过滤）
- **依赖方**：无外部依赖（标准库 `re`、`datetime`）

## 注意事项
- `year=None` 的 chunk 始终保留（宁可召回让 LLM 判断，不因缺失 year 丢弃）
- 三档放宽策略：第一档严格过滤 → 第二档 year_from -2 → 第三档放弃约束
- 支持中文数字（近三年/近十年）和英文模糊词（latest/recent）
- 年份范围限制：`2000 <= year <= current + 5`，防止匹配噪声数字串
