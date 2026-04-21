# SSE 事件协议文档

> 适用接口：`POST /agent/chat`
> 传输格式：`text/event-stream`（Server-Sent Events）
> 每条事件格式：`event: <type>\ndata: <json>\n\n`

---

## 事件类型一览

| 事件类型 | 触发时机 | 必含字段 |
|---|---|---|
| `session_start` | 会话建立时（首条事件） | `session_id` |
| `thinking` | Agent 开始新一轮推理 | `iteration` |
| `tool_start` | 工具调用开始 | `tool_name`, `display_message` |
| `tool_end` | 工具调用结束 | `tool_name`, `result_summary`, `elapsed_ms` |
| `text_delta` | 流式文本片段 | `delta` |
| `sources` | 检索来源列表 | `sources` |
| `usage` | Token 用量统计 | `total_tokens`, `prompt_tokens`, `completion_tokens`, `estimated_cost_cny` |
| `done` | 本轮对话完成 | `session_id` |
| `error` | 发生错误 | `message`, `code` |

---

## 事件详细格式

### session_start
```json
{"type": "session_start", "session_id": "session_abc123"}
```

### thinking
```json
{"type": "thinking", "iteration": 1}
```

### tool_start
```json
{
  "type": "tool_start",
  "tool_name": "retrieve_knowledge",
  "display_message": "正在检索知识库..."
}
```

### tool_end
```json
{
  "type": "tool_end",
  "tool_name": "retrieve_knowledge",
  "result_summary": "找到 5 个相关片段",
  "elapsed_ms": 320
}
```

### text_delta
```json
{"type": "text_delta", "delta": "根据检索结果，"}
```

### sources
```json
{
  "type": "sources",
  "sources": [
    {"file_path": "paper.pdf", "chunk_index": 3, "section_type": "method", "content": "..."}
  ]
}
```

### usage
```json
{
  "type": "usage",
  "total_tokens": 1200,
  "prompt_tokens": 900,
  "completion_tokens": 300,
  "estimated_cost_cny": 0.012
}
```

### done
```json
{"type": "done", "session_id": "session_abc123"}
```

### error
```json
{"type": "error", "message": "工具调用失败：连接超时", "code": "tool_failed"}
```

---

## 典型事件序列（正常流）

```
session_start
thinking (iteration=1)
tool_start (retrieve_knowledge)
tool_end   (retrieve_knowledge, elapsed_ms=320)
text_delta × N
sources
usage
done
```

---

## 前端接入示例（TypeScript）

```typescript
const res = await fetch('/agent/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
  body: JSON.stringify({ session_id: 'session_abc', user_input: '什么是 RAG？' }),
})

const reader = res.body!.getReader()
const decoder = new TextDecoder()
let buf = ''

while (true) {
  const { done, value } = await reader.read()
  if (done) break
  buf += decoder.decode(value, { stream: true })
  for (const part of buf.split('\n\n')) {
    const dataLine = part.split('\n').find(l => l.startsWith('data:'))
    if (dataLine) {
      const event = JSON.parse(dataLine.slice(5).trim())
      // handle event.type ...
    }
  }
}
```

---

## 请求体字段

`POST /agent/chat` 请求体（`application/json`）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `user_input` | string | ✅ | 用户输入文本 |
| `session_id` | string | ❌ | 会话 ID；为空时由服务端自动生成 UUID4 |

---

## 错误码（error.code）

| code | 含义 | 来源 |
|---|---|---|
| `AGENT_INIT_ERROR` | MasterAgent 初始化失败 | `agent.py` 生成器内捕获 |
| `ROUTER_ERROR` | 流式生成过程中未预期异常 | `agent.py` 最外层兜底 |
| `session_invalid` | session_id 格式非法（HTTP 422） | FastAPI 请求校验 |
