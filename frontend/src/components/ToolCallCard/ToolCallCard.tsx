import { motion, AnimatePresence } from 'framer-motion'
import type { ToolCall } from '../../types'

interface Props {
  toolCalls: ToolCall[]
}

const statusIcon = (status: ToolCall['status']) => {
  if (status === 'loading') return <span className="animate-spin">⟳</span>
  if (status === 'done') return '✓'
  return '✗'
}

const statusColor = (status: ToolCall['status']) => {
  if (status === 'loading') return 'border-violet-500/50 bg-violet-500/10'
  if (status === 'done') return 'border-emerald-500/50 bg-emerald-500/10'
  return 'border-red-500/50 bg-red-500/10'
}

export default function ToolCallCard({ toolCalls }: Props) {
  if (toolCalls.length === 0) return null

  return (
    <div className="flex flex-col gap-2 p-3">
      <p className="text-xs text-slate-500 uppercase tracking-wider">工具调用</p>
      <AnimatePresence>
        {toolCalls.map((tc, i) => (
          <motion.div
            key={`${tc.tool_name}-${i}`}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            className={`rounded-lg border p-3 text-sm ${statusColor(tc.status)}`}
          >
            <div className="flex items-center gap-2 font-mono text-xs">
              <span className="text-base">{statusIcon(tc.status)}</span>
              <span className="text-slate-300">{tc.tool_name}</span>
              {tc.elapsed_ms !== undefined && (
                <span className="ml-auto text-slate-500">{tc.elapsed_ms}ms</span>
              )}
            </div>
            {tc.display_message && (
              <p className="mt-1 text-slate-400 text-xs">{tc.display_message}</p>
            )}
            {tc.result_summary && (
              <p className="mt-1 text-slate-300 text-xs">{tc.result_summary}</p>
            )}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  )
}
