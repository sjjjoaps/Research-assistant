import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import { useChatStore } from '../stores/chatStore'
import { useSSE } from '../hooks/useSSE'
import ToolCallCard from '../components/ToolCallCard/ToolCallCard'

export default function ChatPage() {
  const [input, setInput] = useState('')
  const [sessionId] = useState(() => `session_${Date.now()}`)
  const { messages, isStreaming, activeToolCalls } = useChatStore()
  const { send } = useSSE()
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = async () => {
    const text = input.trim()
    if (!text || isStreaming) return
    setInput('')
    await send(sessionId, text)
  }

  return (
    <div className="flex h-full">
      {/* Messages */}
      <div className="flex-1 flex flex-col">
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && (
            <div className="flex items-center justify-center h-full text-slate-500">
              <p>发送消息开始对话</p>
            </div>
          )}
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm ${
                  msg.role === 'user'
                    ? 'bg-violet-600 text-white'
                    : 'bg-[#1e2130] text-slate-200'
                }`}
              >
                {msg.role === 'assistant' ? (
                  <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
                    {msg.content}
                  </ReactMarkdown>
                ) : (
                  msg.content
                )}
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="p-4 border-t border-slate-700/50">
          <div className="flex gap-2">
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleSend()}
              placeholder="输入问题..."
              disabled={isStreaming}
              className="flex-1 bg-[#1e2130] border border-slate-600 rounded-xl px-4 py-2.5 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-violet-500 disabled:opacity-50"
            />
            <button
              onClick={handleSend}
              disabled={isStreaming || !input.trim()}
              className="px-4 py-2.5 bg-violet-600 hover:bg-violet-500 disabled:opacity-40 rounded-xl text-sm font-medium transition-colors"
            >
              {isStreaming ? '...' : '发送'}
            </button>
          </div>
        </div>
      </div>

      {/* Tool calls panel */}
      <div className="w-64 border-l border-slate-700/50 overflow-y-auto shrink-0">
        <ToolCallCard toolCalls={activeToolCalls} />
      </div>
    </div>
  )
}
