# deep_research_agent.py

## 所在层次
Agent 层 `src/agents/`

## 主体功能
深度研究 Agent，实现 Plan-Execute-Report 多步推理。
固定五步编排：规划子问题 → 逐步检索 → 分析证据 → 可选社区摘要 → 生成完整报告。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `SubQuestionResult` (dataclass) | 子问题结果：`question / chunks / sources / analysis / token_usage` |
| `ResearchReport` (dataclass) | 最终报告：`sub_results / report_text / total_token_usage` |
| `DeepResearchAgent.run(thread_id, question, retriever_mode, use_community)` | 五步研究主流程 |
| `DeepResearchAgent._step_plan(question)` | Step1：LLM 将问题拆解为子问题列表 |
| `DeepResearchAgent._step_retrieve(sub_q, mode)` | Step2：对单个子问题执行检索 |
| `DeepResearchAgent._step_analyze(sub_q, chunks)` | Step3：生成证据约束结论 |
| `DeepResearchAgent._step_report(question, sub_results, community_summaries)` | Step5：汇总生成 Markdown 报告 |

## 调用关系
- **被调用方**：`tool_registry.deep_research`（MasterAgent 工具）
- **依赖方**：`BaseAgent`、各检索器（按 mode 切换）、`KeywordExtractor`、`src/agents/prompt_loader`

## 注意事项
- `use_community=True` 时 Step4 查询 Neo4j 社区摘要作为额外视角
- Phase 4.1 对每个子问题提取关键词并以 debug 日志记录（不影响主流程）
- 每步均有独立 `TokenUsage` 统计，汇总到 `ResearchReport.total_token_usage`
