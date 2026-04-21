import { motion, AnimatePresence } from 'framer-motion'
import type { ToolCall } from '../../types'

interface Props {
  toolCalls: ToolCall[]
}

function StatusDot({ status }: { status: ToolCall['status'] }) {
  if (status === 'loading') {
    return (
      <span style={{
        display: 'inline-block',
        width: 7, height: 7,
        borderRadius: '50%',
        background: 'var(--accent)',
        boxShadow: '0 0 6px var(--accent-glow)',
        animation: 'glow-pulse 1.4s ease-in-out infinite',
        flexShrink: 0,
      }} />
    )
  }
  if (status === 'done') {
    return (
      <span style={{
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        width: 14, height: 14,
        borderRadius: '50%',
        background: 'var(--success-subtle)',
        color: 'var(--success)',
        fontSize: 9,
        fontWeight: 700,
        flexShrink: 0,
      }}>✓</span>
    )
  }
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
      width: 14, height: 14,
      borderRadius: '50%',
      background: 'var(--danger-subtle)',
      color: 'var(--danger)',
      fontSize: 9,
      fontWeight: 700,
      flexShrink: 0,
    }}>✗</span>
  )
}

export default function ToolCallCard({ toolCalls }: Props) {
  if (toolCalls.length === 0) {
    return (
      <div style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        height: '100%', gap: 8, padding: 20,
        color: 'var(--text-faint)',
      }}>
        <div style={{
          width: 36, height: 36,
          borderRadius: 'var(--r-md)',
          background: 'var(--surface)',
          border: '1px solid var(--border-light)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 16,
          opacity: 0.5,
        }}>
          ⚙
        </div>
        <p style={{ fontSize: 11.5, textAlign: 'center', lineHeight: 1.5 }}>工具调用<br />将在此显示</p>
      </div>
    )
  }

  return (
    <div style={{ padding: '14px 12px', display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{
        fontSize: 10.5, fontWeight: 600, textTransform: 'uppercase',
        letterSpacing: '0.1em', color: 'var(--text-dim)', marginBottom: 2,
      }}>
        工具调用
      </div>
      <AnimatePresence>
        {toolCalls.map((tc, i) => (
          <motion.div
            key={`${tc.tool_name}-${i}`}
            initial={{ opacity: 0, x: 16, scale: 0.97 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 16, scale: 0.97 }}
            transition={{ duration: 0.2 }}
            style={{
              borderRadius: 'var(--r-md)',
              border: `1px solid ${
                tc.status === 'loading' ? 'rgba(91,110,245,0.25)' :
                tc.status === 'done'    ? 'rgba(16,185,129,0.2)' :
                                          'rgba(244,63,94,0.2)'
              }`,
              background: tc.status === 'loading' ? 'var(--accent-subtle)' :
                          tc.status === 'done'    ? 'var(--success-subtle)' :
                                                    'var(--danger-subtle)',
              padding: '10px 12px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              <StatusDot status={tc.status} />
              <span style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 11.5,
                color: 'var(--text)',
                flex: 1,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}>
                {tc.tool_name}
              </span>
              {tc.elapsed_ms !== undefined && (
                <span style={{ fontSize: 10.5, color: 'var(--text-dim)', flexShrink: 0 }}>
                  {tc.elapsed_ms}ms
                </span>
              )}
            </div>
            {tc.display_message && (
              <p style={{ marginTop: 5, fontSize: 11.5, color: 'var(--text-muted)', lineHeight: 1.5 }}>
                {tc.display_message}
              </p>
            )}
            {tc.result_summary && (
              <p style={{ marginTop: 4, fontSize: 11.5, color: 'var(--text)', lineHeight: 1.5 }}>
                {tc.result_summary}
              </p>
            )}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  )
}
