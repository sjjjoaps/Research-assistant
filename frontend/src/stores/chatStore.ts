import { create } from 'zustand'
import type { ChatMessage, ToolCall, Session } from '../types'

interface ChatState {
  sessions: Session[]
  currentSessionId: string | null
  messages: ChatMessage[]
  isStreaming: boolean
  activeToolCalls: ToolCall[]

  setCurrentSession: (id: string | null) => void
  setSessions: (sessions: Session[]) => void
  addMessage: (msg: ChatMessage) => void
  setMessages: (msgs: ChatMessage[]) => void
  appendDelta: (delta: string) => void
  setStreaming: (v: boolean) => void
  upsertToolCall: (tc: ToolCall) => void
  clearToolCalls: () => void
}

export const useChatStore = create<ChatState>((set) => ({
  sessions: [],
  currentSessionId: null,
  messages: [],
  isStreaming: false,
  activeToolCalls: [],

  setCurrentSession: (id) => set({ currentSessionId: id }),
  setSessions: (sessions) => set({ sessions }),
  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  setMessages: (msgs) => set({ messages: msgs }),
  appendDelta: (delta) =>
    set((s) => {
      const msgs = [...s.messages]
      const last = msgs[msgs.length - 1]
      if (last?.role === 'assistant') {
        msgs[msgs.length - 1] = { ...last, content: last.content + delta }
      } else {
        msgs.push({ role: 'assistant', content: delta })
      }
      return { messages: msgs }
    }),
  setStreaming: (v) => set({ isStreaming: v }),
  upsertToolCall: (tc) =>
    set((s) => {
      const idx = s.activeToolCalls.findIndex(t => t.tool_name === tc.tool_name && t.status === 'loading')
      if (idx >= 0) {
        const updated = [...s.activeToolCalls]
        updated[idx] = tc
        return { activeToolCalls: updated }
      }
      return { activeToolCalls: [...s.activeToolCalls, tc] }
    }),
  clearToolCalls: () => set({ activeToolCalls: [] }),
}))
