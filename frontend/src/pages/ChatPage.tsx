import { useState, useRef, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import { MessageSquare, Plus, Trash2, Paperclip, Send, Loader2, ChevronDown, ChevronRight } from 'lucide-react'
import { useChatStore } from '../stores/chatStore'
import { useSSE } from '../hooks/useSSE'
import ToolCallCard from '../components/ToolCallCard/ToolCallCard'
import FileUploader from '../components/FileUploader/FileUploader'
import { listSessions, deleteSession, getSessionHistory } from '../api/client'
import type { Session, Source } from '../types'
import { sourceFilePath, sourceContent, sourceSectionType } from '../types'

// ── Session list ──────────────────────────────────────────────────────────────
function SessionList({
  sessions,
  currentId,
  onSelect,
  onNew,
  onDelete,
}: {
  sessions: Session[]
  currentId: string | null
  onSelect: (id: string) => void
  onNew: () => void
  onDelete: (id: string) => void
}) {
  return (
    <div className="w-72 border-r shrink-0 flex flex-col" style={{ background: 'var(--panel)', borderColor: 'var(--border)' }}>
      <div className="p-4 border-b" style={{ borderColor: 'var(--border)' }}>
        <button
          onClick={onNew}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-colors"
          style={{ background: 'var(--accent)', color: '#fff' }}
          onMouseEnter={e => (e.currentTarget.style.background = 'var(--accent-hover)')}
          onMouseLeave={e => (e.currentTarget.style.background = 'var(--accent)')}
        >
          <Plus size={15} />
          新会话
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-1">
        {sessions.length === 0 && (
          <p className="text-xs px-3 py-2" style={{ color: 'var(--text-dim)' }}>暂无会话</p>
        )}
        {sessions.map(s => (
          <div
            key={s.session_id}
            onClick={() => onSelect(s.session_id)}
            className="group flex items-center justify-between rounded-xl px-3 py-2.5 cursor-pointer transition-colors"
            style={{
              background: currentId === s.session_id ? 'rgba(124,58,237,0.18)' : 'transparent',
              color: currentId === s.session_id ? '#c4b5fd' : 'var(--text-muted)',
            }}
            onMouseEnter={e => {
              if (currentId !== s.session_id)
                e.currentTarget.style.background = 'rgba(255,255,255,0.04)'
            }}
            onMouseLeave={e => {
              if (currentId !== s.session_id)
                e.currentTarget.style.background = 'transparent'
            }}
          >
            <div className="flex items-start gap-2.5 min-w-0">
              <MessageSquare size={13} className="mt-0.5 shrink-0 opacity-60" />
              <div className="min-w-0">
                <p className="text-sm truncate">{s.title || '新会话'}</p>
                <p className="text-xs mt-0.5 opacity-50">{s.turn_count} 轮对话</p>
              </div>
            </div>
            <button
              onClick={e => { e.stopPropagation(); onDelete(s.session_id) }}
              className="opacity-0 group-hover:opacity-100 p-1 rounded-lg transition-all hover:text-red-400"
              style={{ color: 'var(--text-dim)' }}
            >
              <Trash2 size={13} />
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Sources accordion ─────────────────────────────────────────────────────────
function SourcesAccordion({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false)
  if (!sources.length) return null
  return (
    <div className="mt-3">
      <button
        onClick={() => setOpen(v => !v)}
        className="flex items-center gap-1.5 text-xs transition-colors"
        style={{ color: 'var(--text-dim)' }}
        onMouseEnter={e => (e.currentTarget.style.color = 'var(--text-muted)')}
        onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-dim)')}
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        引用来源 ({sources.length})
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="overflow-hidden"
          >
            <div className="mt-2 space-y-1.5">
              {sources.map((src, i) => (
                <div
                  key={i}
                  className="rounded-xl px-3 py-2.5 text-xs"
                  style={{ background: 'var(--surface)', color: 'var(--text-muted)' }}
                >
                  <p className="font-mono truncate" style={{ color: 'var(--text)' }}>
                    {sourceFilePath(src)}
                  </p>
                  {sourceSectionType(src) && (
                    <span className="text-violet-400/70 text-[10px]">[{sourceSectionType(src)}]</span>
                  )}
                  {sourceContent(src) && (
                    <p className="mt-1 line-clamp-2 text-[11px]" style={{ color: 'var(--text-dim)' }}>
                      {sourceContent(src)}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ── Upload drawer ─────────────────────────────────────────────────────────────
function UploadDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ y: '100%' }}
          animate={{ y: 0 }}
          exit={{ y: '100%' }}
          transition={{ type: 'spring', damping: 30, stiffness: 300 }}
          className="absolute bottom-0 left-0 right-0 border-t z-20 rounded-t-2xl p-5"
          style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold" style={{ color: 'var(--text)' }}>上传文献</h3>
            <button
              onClick={onClose}
              className="w-7 h-7 flex items-center justify-center rounded-lg text-lg leading-none transition-colors"
              style={{ color: 'var(--text-dim)' }}
              onMouseEnter={e => (e.currentTarget.style.color = 'var(--text)')}
              onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-dim)')}
            >
              ×
            </button>
          </div>
          <FileUploader />
        </motion.div>
      )}
    </AnimatePresence>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function ChatPage() {
  const [input, setInput] = useState('')
  const [showUpload, setShowUpload] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const {
    sessions, currentSessionId, messages, isStreaming, activeToolCalls,
    setSessions, setCurrentSession, removeSession, setMessages, upsertSession,
  } = useChatStore()
  const { send } = useSSE()
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    listSessions()
      .then((data: unknown) => {
        const list: Session[] = Array.isArray(data)
          ? data
          : (data as { sessions?: Session[] }).sessions ?? []
        setSessions(list)
      })
      .catch(console.error)
  }, [setSessions])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`
  }, [input])

  const newSession = useCallback(() => {
    const id = `session_${Date.now()}`
    setCurrentSession(id)
    setMessages([])
  }, [setCurrentSession, setMessages])

  const selectSession = useCallback(async (id: string) => {
    setCurrentSession(id)
    try {
      const data = await getSessionHistory(id)
      const history: Array<{ role: string; content: string }> = data.messages ?? data.history ?? []
      setMessages(
        history
          .filter(m => m.role === 'human' || m.role === 'user' || m.role === 'assistant')
          .map(m => ({
            role: (m.role === 'human' || m.role === 'user') ? 'user' : 'assistant' as 'user' | 'assistant',
            content: m.content,
          })),
      )
    } catch {
      setMessages([])
    }
  }, [setCurrentSession, setMessages])

  const handleDeleteSession = useCallback(async (id: string) => {
    try {
      await deleteSession(id)
    } catch { /* ignore */ }
    removeSession(id)
    if (currentSessionId === id) {
      setCurrentSession(null)
      setMessages([])
    }
  }, [currentSessionId, removeSession, setCurrentSession, setMessages])

  const handleSend = useCallback(async () => {
    const text = input.trim()
    if (!text || isStreaming) return
    const sid = currentSessionId ?? `session_${Date.now()}`
    if (!currentSessionId) {
      setCurrentSession(sid)
      upsertSession({
        session_id: sid,
        title: text.slice(0, 30),
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        turn_count: 0,
        total_tokens: 0,
        total_cost_cny: 0,
      })
    }
    setInput('')
    await send(sid, text)
  }, [input, isStreaming, currentSessionId, setCurrentSession, upsertSession, send])

  return (
    <div className="flex h-full relative">
      {/* Session list */}
      <SessionList
        sessions={sessions}
        currentId={currentSessionId}
        onSelect={selectSession}
        onNew={newSession}
        onDelete={handleDeleteSession}
      />

      {/* Messages */}
      <div className="flex-1 flex flex-col min-w-0" style={{ background: 'var(--bg)' }}>
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-5">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full gap-3" style={{ color: 'var(--text-dim)' }}>
              <MessageSquare size={36} className="opacity-30" />
              <p className="text-sm">选择或新建会话，开始提问</p>
            </div>
          )}
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className="rounded-2xl px-5 py-4 leading-relaxed"
                style={{
                  fontSize: '25px',
                  maxWidth: '72%',
                  background: msg.role === 'user' ? 'var(--accent)' : 'var(--surface2)',
                  color: msg.role === 'user' ? '#fff' : 'var(--text)',
                }}
              >
                {msg.role === 'assistant' ? (
                  <>
                    <ReactMarkdown
                      rehypePlugins={[rehypeHighlight]}
                      components={{
                        pre: ({ children }) => (
                          <pre className="overflow-x-auto rounded-xl p-4 my-3 text-[13px]" style={{ background: 'var(--panel)' }}>
                            {children}
                          </pre>
                        ),
                        code: ({ className, children }) =>
                          className ? (
                            <code className={className}>{children}</code>
                          ) : (
                            <code className="px-1.5 py-0.5 rounded-md text-[13px] text-violet-300" style={{ background: 'var(--panel)' }}>
                              {children}
                            </code>
                          ),
                        p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                      }}
                    >
                      {msg.content}
                    </ReactMarkdown>
                    {msg.sources && <SourcesAccordion sources={msg.sources} />}
                  </>
                ) : (
                  <span className="whitespace-pre-wrap">{msg.content}</span>
                )}
              </div>
            </div>
          ))}
          {isStreaming && (
            <div className="flex justify-start">
              <div className="rounded-2xl px-5 py-4" style={{ background: 'var(--surface2)' }}>
                <Loader2 size={16} className="animate-spin" style={{ color: 'var(--accent)' }} />
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Composer */}
        <div className="px-6 pb-6 pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
          <div
            className="flex items-end gap-3 rounded-2xl border px-5 py-4 transition-colors focus-within:border-violet-500/60"
            style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}
          >
            <button
              onClick={() => setShowUpload(v => !v)}
              title="上传文献"
              className="p-5 rounded-xl transition-colors shrink-0 mb-0.5"
              style={{ color: 'var(--text-dim)' }}
              onMouseEnter={e => (e.currentTarget.style.color = 'var(--text)')}
              onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-dim)')}
            >
              <Paperclip size={22} />
            </button>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSend()
                }
              }}
              placeholder="输入问题... (Shift+Enter 换行)"
              disabled={isStreaming}
              rows={1}
              className="flex-1 bg-transparent outline-none resize-none disabled:opacity-50"
              style={{
                color: 'var(--text)',
                fontSize: '16px',
                lineHeight: '25px',
                minHeight: '32px',
                maxHeight: '200px',
              }}
            />
            <button
              onClick={handleSend}
              disabled={isStreaming || !input.trim()}
              className="p-2.5 rounded-xl transition-colors shrink-0 mb-0.5 disabled:opacity-30"
              style={{ background: 'var(--accent)', color: '#fff' }}
              onMouseEnter={e => {
                if (!isStreaming && input.trim())
                  e.currentTarget.style.background = 'var(--accent-hover)'
              }}
              onMouseLeave={e => (e.currentTarget.style.background = 'var(--accent)')}
            >
              {isStreaming ? <Loader2 size={22} className="animate-spin" /> : <Send size={22} />}
            </button>
          </div>
          <p className="text-xs mt-2 text-center" style={{ color: 'var(--text-dim)' }}>
            Enter 发送 · Shift+Enter 换行
          </p>
        </div>
      </div>

      {/* Tool call panel */}
      <div className="w-80 border-l overflow-y-auto shrink-0" style={{ background: 'var(--panel)', borderColor: 'var(--border)' }}>
        <ToolCallCard toolCalls={activeToolCalls} />
      </div>

      <UploadDrawer open={showUpload} onClose={() => setShowUpload(false)} />
    </div>
  )
}
