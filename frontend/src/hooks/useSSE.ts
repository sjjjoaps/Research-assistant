import { useCallback } from 'react'
import { streamChat, listSessions } from '../api/client'
import { useChatStore } from '../stores/chatStore'
import type { ToolCall } from '../types'

export function useSSE() {
  const {
    setStreaming, appendDelta, setSources,
    upsertToolCall, clearToolCalls, addMessage,
    upsertSession, setCurrentSession,
  } = useChatStore()

  const send = useCallback(async (sessionId: string, message: string) => {
    addMessage({ role: 'user', content: message })
    setStreaming(true)
    clearToolCalls()

    try {
      for await (const event of streamChat(sessionId, message)) {
        switch (event.type) {
          case 'session_start':
            setCurrentSession(event.session_id)
            break
          case 'text_delta':
            appendDelta(event.delta)
            break
          case 'sources':
            setSources(event.sources)
            break
          case 'tool_start': {
            const tc: ToolCall = {
              tool_name: event.tool_name,
              status: 'loading',
              display_message: event.display_message,
            }
            upsertToolCall(tc)
            break
          }
          case 'tool_end': {
            const tc: ToolCall = {
              tool_name: event.tool_name,
              status: 'done',
              result_summary: event.result_summary,
              elapsed_ms: event.elapsed_ms,
            }
            upsertToolCall(tc)
            break
          }
          case 'error':
            appendDelta(`\n\n[错误] ${event.message}`)
            break
          case 'done':
            listSessions()
              .then((data: unknown) => {
                const list: import('../types').Session[] = Array.isArray(data)
                  ? data
                  : (data as { sessions?: import('../types').Session[] }).sessions ?? []
                list.forEach(s => upsertSession(s))
              })
              .catch(() => undefined)
            break
        }
      }
    } finally {
      setStreaming(false)
    }
  }, [addMessage, setStreaming, clearToolCalls, appendDelta, setSources,
      upsertToolCall, upsertSession, setCurrentSession])

  return { send }
}
