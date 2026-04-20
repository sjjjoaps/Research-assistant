# Prompt 规范文档（prompt_spec）

本文档规定项目 `prompt/` 目录下所有提示词文件的命名约定、结构规范、模板变量命名规则，以及新增提示词的检查清单。

---

## 一、目录结构

```
prompt/
├── {name}.md                  ← 主提示词（合并格式，优先）
└── tools/
    └── {tool_name}.md         ← 工具描述提示词（五层结构）
```

- 主提示词放在 `prompt/` 根目录，文件名即 `name`
- 工具描述提示词放在 `prompt/tools/` 子目录，文件名即工具函数名

---

## 二、命名约定

### 主提示词命名规则

| 模式 | 示例 | 说明 |
|---|---|---|
| `{agent_name}.md` | `master_agent.md` | Agent 的 system+human 提示词对 |
| `{module_name}.md` | `metadata_extractor.md` | 模块的 system+human 提示词对 |
| `{module_name}_fallback.md` | `keyword_extractor_fallback.md` | 仅含 system prompt 的 LLM 兜底提示词 |
| `{workflow_name}.md` | `session_compact.md` | 工作流专用提示词 |

命名规则：
- 全小写，单词间用下划线 `_` 分隔
- 名称与调用它的模块/函数保持一致（便于追踪）
- `_fallback` 后缀表示该文件仅含 system prompt（无 human template）

### 工具描述命名规则

文件名 = 工具函数名（与 `@tool("tool_name")` 装饰器参数完全一致）。

---

## 三、主提示词三层结构规范

每个主提示词文件（`{name}.md`）采用合并格式，包含三层：

```markdown
# {name} Prompt

## System Prompt

### 边界层
- 禁止 ...（明确列出不允许的行为）

### 决策层
{角色定义、任务目标、行为准则}

### 任务示例
**示例 1**
输入：...
输出：...

---

## Human Prompt Template

{含 {变量名} 占位符的用户消息模板}
```

### 三层说明

| 层次 | 作用 | 典型内容 |
|---|---|---|
| **边界层** | 明确禁止行为，防止幻觉和越界 | `禁止凭空生成...`、`禁止伪造来源...` |
| **决策层** | 角色定义、任务目标、行为准则 | 角色描述、分步推进规则、工具使用策略 |
| **任务示例** | 具体输入/输出示例，帮助 LLM 理解期望格式 | 完整的输入→输出示例对 |

### 分隔符规范

- System Prompt 与 Human Prompt Template 之间用 `---`（单独一行）分隔
- `prompt_loader.py` 按此分隔符解析，缺少时 human template 返回空字符串

---

## 四、模板变量命名规范

Human Prompt Template 中的 `{变量名}` 占位符命名规则：

| 变量类型 | 命名规范 | 示例 |
|---|---|---|
| 用户输入 | `{query}` / `{question}` / `{user_input}` | `{query}` |
| 文档内容 | `{context}` / `{document_text}` | `{context}` |
| 历史对话 | `{history}` | `{history}` |
| 实体/关系 | `{entity_name}` / `{relation_type}` | `{entity_name}` |
| 子问题列表 | `{sub_questions}` | `{sub_questions}` |
| 研究报告 | `{report}` / `{research_report}` | `{research_report}` |
| 描述文本 | `{existing_description}` / `{new_description}` | `{existing_description}` |

规则：
- 全小写，单词间用下划线 `_` 分隔
- 名称应自解释，不使用 `{text1}`、`{input}` 等模糊名称
- 同一变量在不同文件中保持一致（如 `{query}` 不在某文件中改为 `{q}`）

---

## 五、工具描述五层结构规范

`prompt/tools/{tool_name}.md` 文件采用五层结构：

```markdown
# {tool_name}

## 工具作用
{一句话描述工具的核心功能}

## 何时使用
- {触发条件 1}
- {触发条件 2}

## 何时不使用
- {反例 1}
- {反例 2}

## 使用约束
- {参数约束、默认值说明、注意事项}

## 使用示例
### 示例 1：{场景名}
输入：{参数示例}
输出：{预期输出描述}
```

### 五层说明

| 层次 | 作用 |
|---|---|
| **工具作用** | 一句话说明工具做什么，LLM 据此判断是否调用 |
| **何时使用** | 正向触发条件，帮助 LLM 识别适用场景 |
| **何时不使用** | 反例，防止 LLM 在不适合的场景滥用工具 |
| **使用约束** | 参数范围、默认值、副作用等约束说明 |
| **使用示例** | 具体输入/输出示例，提升 LLM 调用准确性 |

---

## 六、新增提示词检查清单

新增或修改提示词文件时，按以下清单逐项确认：

### 文件结构
- [ ] 文件名符合命名约定（小写下划线，与调用模块一致）
- [ ] 合并格式文件包含 `## System Prompt` 和 `## Human Prompt Template` 两个章节
- [ ] System 与 Human 之间有 `---` 分隔符（单独一行）
- [ ] 工具描述文件包含五层结构（工具作用/何时使用/何时不使用/使用约束/使用示例）

### 内容质量
- [ ] 边界层明确列出至少 3 条禁止行为
- [ ] 决策层包含角色定义和行为准则
- [ ] 任务示例包含至少 1 个完整的输入→输出示例对
- [ ] 模板变量命名符合规范，无模糊变量名
- [ ] 工具描述的"何时不使用"至少包含 2 条反例

### 集成验证
- [ ] 调用 `load_prompt_pair("{name}")` 或 `load_system_prompt("{name}")` 能正常返回内容
- [ ] Human template 中的 `{变量名}` 与调用方传入的 `format()` 参数完全匹配
- [ ] 修改后运行相关模块的单元测试（`tests/test_{module}.py`）

---

## 七、现有提示词清单

| 文件 | 类型 | 调用方 |
|---|---|---|
| `master_agent.md` | system+human | `prompt_loader.load_master_agent_system_prompt()` |
| `qa_agent.md` | system+human | `QAAgent` |
| `qa_chain.md` | system+human | `src/qa_chain.py` |
| `metadata_extractor.md` | system+human | `MetadataExtractor` |
| `relation_extractor.md` | system+human | `EntityExtractor` |
| `deep_research_plan.md` | system+human | `DeepResearchAgent._step_plan()` |
| `deep_research_analyze.md` | system+human | `DeepResearchAgent._step_analyze()` |
| `deep_research_report.md` | system+human | `DeepResearchAgent._step_report()` |
| `idea_agent.md` | system+human | `IdeaAgent` |
| `community_summary.md` | system+human | `CommunityDetector` |
| `description_merger.md` | system+human | `DescriptionMerger` |
| `session_compact.md` | system+human | `SessionManager.compact()` |
| `modal_image.md` | system+human | `ImageProcessor` |
| `modal_table.md` | system+human | `TableProcessor` |
| `memory_extractor.md` | system only | `memory_extractor.extract_and_save_memory()` |
| `keyword_extractor_fallback.md` | system only | `KeywordExtractor._llm_extract()` |
| `citation_extractor_fallback.md` | system only | `CitationExtractor`（LLM 兜底） |
| `tools/retrieve_knowledge.md` | 工具描述 | `tool_prompt_loader` → `MasterAgent` |
| `tools/deep_research.md` | 工具描述 | `tool_prompt_loader` → `MasterAgent` |
| `tools/generate_research_ideas.md` | 工具描述 | `tool_prompt_loader` → `MasterAgent` |
| `tools/list_documents.md` | 工具描述 | `tool_prompt_loader` → `MasterAgent` |
| `tools/get_document_metadata.md` | 工具描述 | `tool_prompt_loader` → `MasterAgent` |
| `tools/search_by_entity.md` | 工具描述 | `tool_prompt_loader` → `MasterAgent` |
| `tools/get_knowledge_graph_stats.md` | 工具描述 | `tool_prompt_loader` → `MasterAgent` |
| `tools/save_user_memory.md` | 工具描述 | `tool_prompt_loader` → `MasterAgent` |
