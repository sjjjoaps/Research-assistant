# idea_agent.py

## 所在层次
Agent 层 `src/agents/`

## 主体功能
研究 Idea 生成 Agent，消费 `DeepResearchAgent` 产出的 `ResearchReport`，
生成固定模板的 Idea 报告。不直接检索，避免超出已有证据范围。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `IdeaReport` (Pydantic) | Idea 报告结构：`research_gaps / method_comparisons / suggested_directions / evidence_basis / confidence_note` |
| `IdeaAgent.generate(research_report)` | 主入口：将 `ResearchReport` 格式化后调用 LLM 生成 `IdeaReport` |

## 调用关系
- **被调用方**：`tool_registry.generate_research_ideas`（MasterAgent 工具）
- **依赖方**：`DeepResearchAgent.ResearchReport`、`src/agents/prompt_loader.load_prompt_pair("idea_agent")`

## 注意事项
- 不执行任何检索，所有分析严格基于传入的 `ResearchReport` 内容
- `confidence_note` 字段要求 LLM 明确说明证据充分度，防止幻觉
- 使用 Pydantic 结构化输出，确保字段完整性
