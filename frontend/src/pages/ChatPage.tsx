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
  sessions, currentId, onSelect, onNew, onDelete,
}: {
  sessions: Session[]
  currentId: string | null
  onSelect: (id: string) => void
  onNew: () => void
  onDelete: (id: string) => void
}) {
  return (
    <div className="flex flex-col shrink-0" style={{
      width: 240,
      background: 'var(--panel)',
      borderRight: '1px solid var(--border)',
    }}>
      <div className="p-3" style={{ borderBottom: '1px solid var(--border)' }}>
        <button onClick={onNew} className="btn btn-primary w-full" style={{ fontSize: 13 }}>
          <Plus size={14} />
          新建会话
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-2" style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {sessions.length === 0 && (
          <p className="px-3 py-4 text-center" style={{ fontSize: 12, color: 'var(--text-dim)' }}>暂无会话</p>
        )}
        {sessions.map(s => (
          <div
            key={s.session_id}
            onClick={() => onSelect(s.session_id)}
            className="group"
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              borderRadius: 'var(--r-md)',
              padding: '8px 10px',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
              background: currentId === s.session_id ? 'var(--accent-subtle)' : 'transparent',
              border: `1px solid ${currentId === s.session_id ? 'rgba(91,110,245,0.2)' : 'transparent'}`,
            }}
            onMouseEnter={e => {
              if (currentId !== s.session_id)
                (e.currentTarget as HTMLElement).style.background = 'var(--surface)'
            }}
            onMouseLeave={e => {
              if (currentId !== s.session_id)
                (e.currentTarget as HTMLElement).style.background = 'transparent'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8, minWidth: 0 }}>
              <MessageSquare size={12} style={{
                marginTop: 2,
                flexShrink: 0,
                color: currentId === s.session_id ? '#a5b4fc' : 'var(--text-dim)',
              }} />
              <div style={{ minWidth: 0 }}>
                <p style={{
                  fontSize: 13,
                  fontWeight: 500,
                  color: currentId === s.session_id ? '#c4b5fd' : 'var(--text-muted)',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}>
                  {s.title || '新会话'}
                </p>
                <p style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 1 }}>
                  {s.turn_count} 轮对话
                </p>
              </div>
            </div>
            <button
              onClick={e => { e.stopPropagation(); onDelete(s.session_id) }}
              className="session-delete-btn"
              onMouseEnter={e => (e.currentTarget.style.color = 'var(--danger)')}
              onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-dim)')}
            >
              <Trash2 size={12} />
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
    <div style={{ marginTop: 12 }}>
      <button
        onClick={() => setOpen(v => !v)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 5,
          fontSize: 11.5,
          color: 'var(--text-dim)',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          transition: 'color 0.15s',
        }}
        onMouseEnter={e => (e.currentTarget.style.color = 'var(--text-muted)')}
        onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-dim)')}
      >
        {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
        引用来源 ({sources.length})
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
            style={{ overflow: 'hidden' }}
          >
            <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
              {sources.map((src, i) => (
                <div key={i} style={{
                  borderRadius: 'var(--r-md)',
                  padding: '8px 12px',
                  background: 'var(--surface)',
                  border: '1px solid var(--border-light)',
                  fontSize: 12,
                }}>
                  <p style={{ fontFamily: 'var(--font-mono)', color: 'var(--text)', fontSize: 11.5, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {sourceFilePath(src)}
                  </p>
                  {sourceSectionType(src) && (
                    <span style={{ color: '#a5b4fc', fontSize: 10.5, opacity: 0.8 }}>[{sourceSectionType(src)}]</span>
                  )}
                  {sourceContent(src) && (
                    <p style={{ marginTop: 4, fontSize: 11.5, color: 'var(--text-dim)', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
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
          style={{
            position: 'absolute',
            bottom: 0, left: 0, right: 0,
            borderTop: '1px solid var(--border)',
            zIndex: 20,
            borderRadius: '20px 20px 0 0',
            padding: 20,
            background: 'var(--surface)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
            <h3 style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--text)' }}>上传文献</h3>
            <button
              onClick={onClose}
              style={{
                width: 26, height: 26,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                borderRadius: 'var(--r-sm)',
                fontSize: 16,
                color: 'var(--text-dim)',
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                transition: 'color 0.15s',
              }}
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
    // Don't reload if already on this session — avoids wiping in-progress messages
    if (id === currentSessionId) return
    setCurrentSession(id)
    try {
      const data = await getSessionHistory(id)
      const history: Array<{ role: string; content: string }> = data.messages ?? data.history ?? []
      setMessages(
        history
          .filter(m => m.role === 'human' || m.role === 'user' || m.role === 'assistant')
          .filter(m => m.content)  // skip empty-content assistant tool-call stubs
          .map(m => ({
            role: (m.role === 'human' || m.role === 'user') ? 'user' : 'assistant' as 'user' | 'assistant',
            content: m.content,
          })),
      )
    } catch {
      setMessages([])
    }
  }, [currentSessionId, setCurrentSession, setMessages])

  const handleDeleteSession = useCallback(async (id: string) => {
    try { await deleteSession(id) } catch { /* ignore */ }
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
    <div style={{ display: 'flex', height: '100%', position: 'relative' }}>
      <SessionList
        sessions={sessions}
        currentId={currentSessionId}
        onSelect={selectSession}
        onNew={newSession}
        onDelete={handleDeleteSession}
      />

      {/* Messages area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, background: 'var(--bg-mid)' }}>
        <div style={{ flex: 1, overflowY: 'auto', padding: '24px 28px', display: 'flex', flexDirection: 'column', gap: 16 }}>
          {messages.length === 0 && (
            <div style={{
              display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
              height: '100%', gap: 12, color: 'var(--text-dim)',
            }}>
              <div style={{
                width: 56, height: 56,
                borderRadius: 'var(--r-xl)',
                background: 'var(--surface)',
                border: '1px solid var(--border-light)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <MessageSquare size={24} style={{ opacity: 0.4 }} />
              </div>
              <p style={{ fontSize: 13.5 }}>选择或新建会话，开始提问</p>
            </div>
          )}

          {messages.map((msg, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              style={{ display: 'flex', justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start' }}
            >
              <div
                className={msg.role === 'user' ? 'msg-user' : 'msg-assistant'}
                style={{
                  maxWidth: '72%',
                  padding: '12px 18px',
                  fontSize: 14.5,
                  lineHeight: 1.7,
                }}
              >
                {msg.role === 'assistant' ? (
                  <>
                    <ReactMarkdown
                      rehypePlugins={[rehypeHighlight]}
                      components={{
                        pre: ({ children }) => (
                          <pre style={{
                            overflowX: 'auto',
                            borderRadius: 'var(--r-md)',
                            padding: '14px 16px',
                            margin: '10px 0',
                            fontSize: 12.5,
                            background: 'var(--panel)',
                            border: '1px solid var(--border-light)',
                            fontFamily: 'var(--font-mono)',
                          }}>
                            {children}
                          </pre>
                        ),
                        code: ({ className, children }) =>
                          className ? (
                            <code className={className}>{children}</code>
                          ) : (
                            <code style={{
                              padding: '2px 6px',
                              borderRadius: 'var(--r-sm)',
                              fontSize: 12.5,
                              color: '#a5b4fc',
                              background: 'var(--panel)',
                              fontFamily: 'var(--font-mono)',
                            }}>
                              {children}
                            </code>
                          ),
                        p: ({ children }) => <p style={{ marginBottom: 8 }}>{children}</p>,
                      }}
                    >
                      {msg.content}
                    </ReactMarkdown>
                    {msg.sources && <SourcesAccordion sources={msg.sources} />}
                  </>
                ) : (
                  <span style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</span>
                )}
              </div>
            </motion.div>
          ))}

          {isStreaming && (
            <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
              <div className="msg-assistant" style={{ padding: '12px 18px', maxWidth: '72%' }}>
                {activeToolCalls.length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
                    {activeToolCalls.map((tc, i) => (
                      <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        {tc.status === 'loading' ? (
                          <motion.span
                            style={{
                              display: 'inline-block',
                              width: 7, height: 7,
                              borderRadius: '50%',
                              background: 'var(--accent)',
                              flexShrink: 0,
                            }}
                            animate={{ opacity: [0.3, 1, 0.3] }}
                            transition={{ duration: 1.2, repeat: Infinity }}
                          />
                        ) : (
                          <span style={{
                            display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                            width: 14, height: 14, borderRadius: '50%',
                            background: 'var(--success-subtle)', color: 'var(--success)',
                            fontSize: 9, fontWeight: 700, flexShrink: 0,
                          }}>✓</span>
                        )}
                        <span style={{ fontSize: 12.5, color: 'var(--text-muted)', flex: 1 }}>
                          {tc.display_message || tc.tool_name}
                        </span>
                        {tc.elapsed_ms !== undefined && (
                          <span style={{ fontSize: 11, color: 'var(--text-dim)', flexShrink: 0 }}>
                            {tc.elapsed_ms}ms
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ display: 'flex', gap: 5, alignItems: 'center' }}>
                    {[0, 1, 2].map(i => (
                      <motion.div
                        key={i}
                        style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent)' }}
                        animate={{ opacity: [0.3, 1, 0.3] }}
                        transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }}
                      />
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Composer */}
        <div style={{ padding: '12px 20px 20px', borderTop: '1px solid var(--border)' }}>
          <div className="composer" style={{ display: 'flex', alignItems: 'flex-end', gap: 10, padding: '10px 14px' }}>
            <button
              onClick={() => setShowUpload(v => !v)}
              title="上传文献"
              style={{
                padding: '6px',
                borderRadius: 'var(--r-sm)',
                color: showUpload ? 'var(--accent)' : 'var(--text-dim)',
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                transition: 'color 0.15s',
                flexShrink: 0,
                marginBottom: 2,
              }}
              onMouseEnter={e => { if (!showUpload) (e.currentTarget as HTMLElement).style.color = 'var(--text-muted)' }}
              onMouseLeave={e => { if (!showUpload) (e.currentTarget as HTMLElement).style.color = 'var(--text-dim)' }}
            >
              <Paperclip size={18} />
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
              style={{
                flex: 1,
                background: 'transparent',
                outline: 'none',
                resize: 'none',
                color: 'var(--text)',
                fontSize: 14,
                lineHeight: 1.6,
                minHeight: 28,
                maxHeight: 180,
                fontFamily: 'var(--font-sans)',
                opacity: isStreaming ? 0.5 : 1,
              }}
            />
            <button
              onClick={handleSend}
              disabled={isStreaming || !input.trim()}
              className="btn btn-primary"
              style={{ padding: '7px 10px', flexShrink: 0, marginBottom: 2 }}
            >
              {isStreaming ? <Loader2 size={16} className="spin" /> : <Send size={16} />}
            </button>
          </div>
          <p style={{ fontSize: 11, textAlign: 'center', marginTop: 6, color: 'var(--text-faint)' }}>
            Enter 发送 · Shift+Enter 换行
          </p>
        </div>
      </div>

      {/* Tool call panel */}
      <div style={{
        width: 288,
        borderLeft: '1px solid var(--border)',
        overflowY: 'auto',
        flexShrink: 0,
        background: 'var(--panel)',
      }}>
        <ToolCallCard toolCalls={activeToolCalls} />
      </div>

      <UploadDrawer open={showUpload} onClose={() => setShowUpload(false)} />
    </div>
  )
}
