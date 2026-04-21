import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { getDocumentStatus } from '../../api/client'

interface Props {
  docId: string
  fileName: string
  onDone?: () => void
}

// Ordered backend status values
const STATUS_ORDER = [
  'pending', 'parsing', 'chunking', 'metadata',
  'indexing', 'graph', 'citations', 'extracting', 'processed',
]

const STATUS_LABELS: Record<string, string> = {
  pending: '等待入库',
  parsing: '文档解析中',
  chunking: '分块处理中',
  metadata: '元数据提取中',
  indexing: '向量化中',
  graph: '知识图谱构建中',
  citations: '引用关系提取中',
  extracting: '实体提取中',
  processed: '入库完成',
  failed: '入库失败',
}

function statusProgress(status: string | null): number {
  if (!status) return 0
  const idx = STATUS_ORDER.indexOf(status)
  return idx >= 0 ? idx : 0
}

export default function ParseProgress({ docId, fileName, onDone }: Props) {
  const [status, setStatus] = useState<string | null>(null)
  const [currentStep, setCurrentStep] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const poll = async () => {
      try {
        const data = await getDocumentStatus(docId)
        setStatus(data.status)
        setCurrentStep(data.current_step ?? null)
        setError(data.error_message ?? null)
        if (data.status === 'processed') onDone?.()
      } catch {
        // ignore transient errors
      }
    }
    poll()
    const timer = setInterval(() => {
      if (status === 'processed' || status === 'failed') {
        clearInterval(timer)
        return
      }
      poll()
    }, 2500)
    return () => clearInterval(timer)
  }, [docId, status, onDone])

  const done = status === 'processed'
  const failed = status === 'failed'
  const progress = done ? STATUS_ORDER.length : statusProgress(status)
  const label = failed
    ? (error ?? '入库失败')
    : (currentStep ?? STATUS_LABELS[status ?? ''] ?? '等待中...')

  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      className={`rounded-xl border p-3 text-xs ${
        failed
          ? 'border-red-500/40 bg-red-500/10'
          : done
          ? 'border-emerald-500/40 bg-emerald-500/10'
          : 'border-violet-500/30 bg-violet-500/10'
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-slate-300 truncate max-w-[180px]">{fileName}</span>
        {done && <span className="text-emerald-400">✓ 完成</span>}
        {failed && <span className="text-red-400">✗ 失败</span>}
        {!done && !failed && (
          <span className="animate-spin text-violet-400">⟳</span>
        )}
      </div>

      <div className="h-1 rounded-full bg-slate-700 overflow-hidden mb-2">
        <motion.div
          className={`h-full rounded-full ${done ? 'bg-emerald-500' : failed ? 'bg-red-500' : 'bg-violet-500'}`}
          animate={{ width: `${Math.round((progress / STATUS_ORDER.length) * 100)}%` }}
          transition={{ duration: 0.4 }}
        />
      </div>

      <p className="text-slate-500">{label}</p>
    </motion.div>
  )
}
