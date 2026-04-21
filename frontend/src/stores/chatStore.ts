import { create } from 'zustand'
import type { ChatMessage, ToolCall, Session, TokenUsage } from '../types'

interface ChatState {
  sessions: Session[]
  currentSessionId: string | null
  messages: ChatMessage[]
  isStreaming: boolean
  activeToolCalls: ToolCall[]

  setCurrentSession: (id: string | null) => void
  setSessions: (sessions: Session[]) => void
  upsertSession: (session: Session) => void
  removeSession: (id: string) => void
  addMessage: (msg: ChatMessage) => void
  setMessages: (msgs: ChatMessage[]) => void
  appendDelta: (delta: string) => void
  setSources: (sources: import('../types').Source[]) => void
  setLastUsage: (usage: TokenUsage) => void
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
  upsertSession: (session) =>
    set((s) => {
      const idx = s.sessions.findIndex(x => x.session_id === session.session_id)
      if (idx >= 0) {
        const updated = [...s.sessions]
        updated[idx] = session
        return { sessions: updated }
      }
      return { sessions: [session, ...s.sessions] }
    }),
  removeSession: (id) =>
    set((s) => ({ sessions: s.sessions.filter(x => x.session_id !== id) })),
  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  setMessages: (msgs) => set({ messages: msgs }),
  appendDelta: (delta) =>
    set((s) => {
      const msgs = [...s.messages]
      const last = msgs[msgs.length - 1]
      if (last?.role === 'assistant') {
        msgs[msgs.length - 1] = { ...last, content: last.content + delta }
      } else {
        msgs.push({ role: 'assistant', content: delta, sources: [] })
      }
      return { messages: msgs }
    }),
  setSources: (sources) =>
    set((s) => {
      const msgs = [...s.messages]
      const last = msgs[msgs.length - 1]
      if (last?.role === 'assistant') {
        msgs[msgs.length - 1] = { ...last, sources }
      }
      return { messages: msgs }
    }),
  setLastUsage: (usage) =>
    set((s) => {
      const msgs = [...s.messages]
      const last = msgs[msgs.length - 1]
      if (last?.role === 'assistant') {
        msgs[msgs.length - 1] = { ...last, usage }
      }
      return { messages: msgs }
    }),
  setStreaming: (v) => set({ isStreaming: v }),
  upsertToolCall: (tc) =>
    set((s) => {
      const idx = s.activeToolCalls.findIndex(
        t => t.tool_name === tc.tool_name && t.status === 'loading',
      )
      if (idx >= 0) {
        const updated = [...s.activeToolCalls]
        updated[idx] = tc
        return { activeToolCalls: updated }
      }
      return { activeToolCalls: [...s.activeToolCalls, tc] }
    }),
  clearToolCalls: () => set({ activeToolCalls: [] }),
}))
