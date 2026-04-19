import { useEffect, useState, useRef, useCallback } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Search, Upload, Trash2, RefreshCw, ChevronUp, ChevronDown, ChevronsUpDown, X, FileText, MoreHorizontal } from 'lucide-react'
import { listDocuments, deleteDocument, getDocumentStatus } from '../api/client'
import { useDocumentStore } from '../stores/documentStore'
import FileUploader from '../components/FileUploader/FileUploader'
import type { Document } from '../types'

const RUNNING_STATUS_SET = new Set([
  'pending', 'parsing', 'chunking', 'metadata',
  'indexing', 'graph', 'citations', 'extracting', 'processing',
])

function StatusBadge({ status }: { status: string | null }) {
  const s = status ?? 'unknown'
  let cls = 'bg-slate-500/20 text-slate-400'
  if (s === 'processed') cls = 'bg-emerald-500/20 text-emerald-400'
  else if (s === 'failed') cls = 'bg-red-500/20 text-red-400'
  else if (RUNNING_STATUS_SET.has(s)) cls = 'bg-amber-500/20 text-amber-400'
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${cls}`}>
      {RUNNING_STATUS_SET.has(s) && <span className="animate-spin text-[10px]">⟳</span>}
      {s}
    </span>
  )
}

type SortDir = 'asc' | 'desc' | null
function SortIcon({ dir }: { dir: SortDir }) {
  if (dir === 'asc') return <ChevronUp size={12} />
  if (dir === 'desc') return <ChevronDown size={12} />
  return <ChevronsUpDown size={12} className="opacity-30" />
}
type SortKey = 'title' | 'authors' | 'year' | 'status' | 'chunk_count'

function DetailPanel({ doc, onClose }: { doc: Document; onClose: () => void }) {
  return (
    <motion.div
      initial={{ x: 40, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 40, opacity: 0 }}
      transition={{ duration: 0.18 }}
      className="w-96 border-l overflow-y-auto shrink-0 flex flex-col"
      style={{ background: 'var(--panel)', borderColor: 'var(--border)' }}
    >
      <div className="flex items-center justify-between px-5 py-3.5 border-b" style={{ borderColor: 'var(--border)' }}>
        <div className="flex items-center gap-2" style={{ color: 'var(--text-muted)' }}>
          <FileText size={15} />
          <span className="text-sm font-medium">文献详情</span>
        </div>
        <button onClick={onClose} style={{ color: 'var(--text-dim)' }} className="hover:text-white transition-colors">
          <X size={16} />
        </button>
      </div>

      <div className="p-5 flex-1 space-y-5">
        <div>
          <p className="text-xs font-medium mb-1.5" style={{ color: 'var(--text-dim)' }}>标题</p>
          <p className="text-sm leading-6" style={{ color: 'var(--text)' }}>{doc.title || doc.file_path}</p>
        </div>

        {doc.authors && (
          <div>
            <p className="text-xs font-medium mb-1" style={{ color: 'var(--text-dim)' }}>作者</p>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{doc.authors}</p>
          </div>
        )}

        {doc.institution && (
          <div>
            <p className="text-xs font-medium mb-1" style={{ color: 'var(--text-dim)' }}>机构</p>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{doc.institution}</p>
          </div>
        )}

        <div className="flex gap-6">
          {doc.year && (
            <div>
              <p className="text-xs font-medium mb-1" style={{ color: 'var(--text-dim)' }}>年份</p>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{doc.year}</p>
            </div>
          )}
          <div>
            <p className="text-xs font-medium mb-1" style={{ color: 'var(--text-dim)' }}>状态</p>
            <StatusBadge status={doc.status} />
          </div>
        </div>

        {doc.abstract && (
          <div>
            <p className="text-xs font-medium mb-1.5" style={{ color: 'var(--text-dim)' }}>摘要</p>
            <p className="text-sm leading-6" style={{ color: 'var(--text-muted)' }}>{doc.abstract}</p>
          </div>
        )}

        {doc.keywords && (
          <div>
            <p className="text-xs font-medium mb-2" style={{ color: 'var(--text-dim)' }}>关键词</p>
            <div className="flex flex-wrap gap-1.5">
              {doc.keywords.split(/[,;，；]/).map(k => k.trim()).filter(Boolean).map((k, i) => (
                <span key={i} className="px-2 py-0.5 rounded-lg text-xs bg-violet-500/15 text-violet-300">{k}</span>
              ))}
            </div>
          </div>
        )}

        <div className="grid grid-cols-3 gap-2">
          {([
            { label: 'Chunks', value: doc.chunk_count },
            { label: '实体', value: doc.entity_count },
            { label: '关系', value: doc.relation_count },
          ] as const).map(({ label, value }) => (
            <div key={label} className="rounded-xl p-3 text-center" style={{ background: 'var(--surface)' }}>
              <p className="text-base font-semibold" style={{ color: 'var(--text)' }}>{value ?? '-'}</p>
              <p className="text-xs mt-0.5" style={{ color: 'var(--text-dim)' }}>{label}</p>
            </div>
          ))}
        </div>

        {doc.error_message && (
          <div className="rounded-xl p-3 bg-red-500/10 border border-red-500/30">
            <p className="text-sm text-red-400">{doc.error_message}</p>
          </div>
        )}

        <p className="text-[10px] font-mono break-all" style={{ color: 'var(--text-dim)' }}>{doc.doc_id}</p>
      </div>
    </motion.div>
  )
}

interface CtxMenu { x: number; y: number; doc: Document }

function ContextMenu({ menu, onDelete, onRefresh, onClose }: {
  menu: CtxMenu; onDelete: (d: Document) => void; onRefresh: (d: Document) => void; onClose: () => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) onClose() }
    document.addEventListener('mousedown', h)
    return () => document.removeEventListener('mousedown', h)
  }, [onClose])

  return (
    <div ref={ref} className="fixed z-50 rounded-xl shadow-xl border py-1 min-w-[148px]"
      style={{ top: menu.y, left: menu.x, background: 'var(--surface)', borderColor: 'var(--border)' }}>
      {[
        { label: '刷新状态', icon: <RefreshCw size={13} />, action: () => { onRefresh(menu.doc); onClose() } },
        { label: '删除文献', icon: <Trash2 size={13} />, action: () => { onDelete(menu.doc); onClose() }, danger: true },
      ].map(item => (
        <button key={item.label} onClick={item.action}
          className={`w-full flex items-center gap-2.5 px-3 py-2 text-sm transition-colors ${item.danger ? 'text-red-400 hover:bg-red-500/10' : 'hover:bg-white/5'}`}
          style={{ color: item.danger ? undefined : 'var(--text-muted)' }}>
          {item.icon}{item.label}
        </button>
      ))}
    </div>
  )
}

export default function LibraryPage() {
  const { documents, setDocuments, loading, setLoading, removeDocument, upsertDocument } = useDocumentStore()
  // Store only the ID; derive the live doc from the store so poll updates auto-reflect in panel
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null)
  const selected = selectedDocId ? (documents.find(d => (d.doc_id ?? String(d.id)) === selectedDocId) ?? null) : null

  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState<'all' | 'running' | 'processed' | 'failed'>('all')
  const [sortKey, setSortKey] = useState<SortKey>('title')
  const [sortDir, setSortDir] = useState<SortDir>('asc')
  const [showUpload, setShowUpload] = useState(false)
  const [ctxMenu, setCtxMenu] = useState<CtxMenu | null>(null)

  useEffect(() => {
    setLoading(true)
    listDocuments()
      .then(data => setDocuments(data.documents ?? data ?? []))
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [setDocuments, setLoading])

  // Poll running docs — merge status fields into existing doc to preserve metadata
  useEffect(() => {
    const running = documents.filter(d => RUNNING_STATUS_SET.has(d.status ?? ''))
    if (running.length === 0) return
    const timer = setInterval(async () => {
      for (const doc of running) {
        try {
          const statusData = await getDocumentStatus(doc.doc_id ?? String(doc.id))
          // Merge: keep all existing doc fields, only overwrite status-related fields
          upsertDocument({ ...doc, ...statusData })
        } catch { /* ignore */ }
      }
    }, 3000)
    return () => clearInterval(timer)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documents, upsertDocument])

  const handleDelete = useCallback(async (doc: Document) => {
    if (!confirm(`确认删除 ${doc.title || doc.file_path}？`)) return
    try { await deleteDocument(doc.doc_id ?? String(doc.id)) } catch { /* ignore */ }
    removeDocument(doc.doc_id ?? String(doc.id))
    if (selectedDocId === (doc.doc_id ?? String(doc.id))) setSelectedDocId(null)
  }, [selectedDocId, removeDocument])

  const handleRefresh = useCallback(async (doc: Document) => {
    try {
      const statusData = await getDocumentStatus(doc.doc_id ?? String(doc.id))
      upsertDocument({ ...doc, ...statusData })
    } catch { /* ignore */ }
  }, [upsertDocument])

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : d === 'desc' ? null : 'asc')
    else { setSortKey(key); setSortDir('asc') }
  }

  // Status distribution for chips
  const counts = {
    all: documents.length,
    running: documents.filter(d => RUNNING_STATUS_SET.has(d.status ?? '')).length,
    processed: documents.filter(d => d.status === 'processed').length,
    failed: documents.filter(d => d.status === 'failed').length,
  }

  const filtered = documents.filter(d => {
    if (statusFilter === 'running' && !RUNNING_STATUS_SET.has(d.status ?? '')) return false
    if (statusFilter === 'processed' && d.status !== 'processed') return false
    if (statusFilter === 'failed' && d.status !== 'failed') return false
    if (!search) return true
    const q = search.toLowerCase()
    return (d.title || d.file_path).toLowerCase().includes(q)
      || (d.authors ?? '').toLowerCase().includes(q)
      || String(d.year ?? '').includes(q)
  })

  const sorted = [...filtered].sort((a, b) => {
    if (!sortDir) return 0
    let av: string | number = '', bv: string | number = ''
    if (sortKey === 'title') { av = (a.title || a.file_path).toLowerCase(); bv = (b.title || b.file_path).toLowerCase() }
    else if (sortKey === 'authors') { av = (a.authors ?? '').toLowerCase(); bv = (b.authors ?? '').toLowerCase() }
    else if (sortKey === 'year') { av = a.year ?? 0; bv = b.year ?? 0 }
    else if (sortKey === 'status') { av = a.status ?? ''; bv = b.status ?? '' }
    else if (sortKey === 'chunk_count') { av = a.chunk_count ?? 0; bv = b.chunk_count ?? 0 }
    if (av < bv) return sortDir === 'asc' ? -1 : 1
    if (av > bv) return sortDir === 'asc' ? 1 : -1
    return 0
  })

  const cols: { key: SortKey; label: string; cls?: string }[] = [
    { key: 'title', label: '标题' },
    { key: 'authors', label: '作者', cls: 'hidden md:table-cell' },
    { key: 'year', label: '年份', cls: 'hidden lg:table-cell w-16' },
    { key: 'status', label: '状态', cls: 'w-40' },
    { key: 'chunk_count', label: 'Chunks', cls: 'hidden xl:table-cell w-16 text-right' },
  ]

  const chipStyles = (active: boolean) => ({
    background: active ? 'rgba(124,58,237,0.2)' : 'var(--surface)',
    color: active ? '#c4b5fd' : 'var(--text-dim)',
    border: `1px solid ${active ? 'rgba(124,58,237,0.4)' : 'var(--border)'}`,
  })

  return (
    <div className="flex h-full relative">
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Toolbar */}
        <div className="flex items-center gap-3 px-5 py-3 border-b shrink-0" style={{ borderColor: 'var(--border)', background: 'var(--panel)' }}>
          <div className="relative flex-1 max-w-xs">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-dim)' }} />
            <input value={search} onChange={e => setSearch(e.target.value)} placeholder="搜索标题/作者..."
              className="w-full pl-8 pr-3 py-2 rounded-xl text-sm outline-none"
              style={{ background: 'var(--surface)', color: 'var(--text)', border: '1px solid var(--border)' }}
              onFocus={e => (e.currentTarget.style.borderColor = 'rgba(124,58,237,0.6)')}
              onBlur={e => (e.currentTarget.style.borderColor = 'var(--border)')} />
          </div>

          {/* Status filter chips */}
          <div className="flex gap-1.5">
            {([
              { key: 'all', label: `全部 ${counts.all}` },
              { key: 'running', label: `处理中 ${counts.running}` },
              { key: 'processed', label: `完成 ${counts.processed}` },
              { key: 'failed', label: `失败 ${counts.failed}` },
            ] as const).map(chip => (
              <button key={chip.key} onClick={() => setStatusFilter(chip.key)}
                className="px-2.5 py-1 rounded-lg text-xs font-medium transition-colors whitespace-nowrap"
                style={chipStyles(statusFilter === chip.key)}>
                {chip.label}
              </button>
            ))}
          </div>

          <button onClick={() => setShowUpload(v => !v)}
            className="flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-medium transition-colors ml-auto"
            style={{ background: 'var(--accent)', color: '#fff' }}
            onMouseEnter={e => (e.currentTarget.style.background = 'var(--accent-hover)')}
            onMouseLeave={e => (e.currentTarget.style.background = 'var(--accent)')}>
            <Upload size={14} />上传
          </button>
        </div>

        {/* Upload panel */}
        <AnimatePresence>
          {showUpload && (
            <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.18 }}
              className="overflow-hidden border-b shrink-0 px-5 py-4"
              style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}>
              <FileUploader />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Table */}
        <div className="flex-1 overflow-y-auto">
          {loading ? (
            <div className="flex items-center justify-center h-full" style={{ color: 'var(--text-dim)' }}>
              <span className="animate-spin mr-2 text-lg">⟳</span>加载中...
            </div>
          ) : sorted.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full gap-3" style={{ color: 'var(--text-dim)' }}>
              <FileText size={36} className="opacity-30" />
              <p className="text-sm">{search || statusFilter !== 'all' ? '无匹配文献' : '暂无文献，请上传 PDF'}</p>
            </div>
          ) : (
            <table className="w-full text-sm border-collapse">
              <thead className="sticky top-0 z-10" style={{ background: 'var(--panel)' }}>
                <tr>
                  {cols.map(col => (
                    <th key={col.key} onClick={() => handleSort(col.key)}
                      className={`text-left px-4 py-3 text-sm font-medium cursor-pointer select-none border-b ${col.cls ?? ''}`}
                      style={{ color: 'var(--text-dim)', borderColor: 'var(--border)' }}>
                      <span className="flex items-center gap-1">
                        {col.label}<SortIcon dir={sortKey === col.key ? sortDir : null} />
                      </span>
                    </th>
                  ))}
                  <th className="px-4 py-3 border-b w-10" style={{ borderColor: 'var(--border)' }} />
                </tr>
              </thead>
              <tbody>
                {sorted.map(doc => {
                  const docKey = doc.doc_id ?? String(doc.id)
                  const isSelected = selectedDocId === docKey
                  return (
                    <tr key={docKey}
                      onClick={() => setSelectedDocId(id => id === docKey ? null : docKey)}
                      onContextMenu={e => { e.preventDefault(); setCtxMenu({ x: e.clientX, y: e.clientY, doc }) }}
                      className="group border-b cursor-pointer transition-colors"
                      style={{ borderColor: 'var(--border)', background: isSelected ? 'rgba(124,58,237,0.1)' : 'transparent' }}
                      onMouseEnter={e => { if (!isSelected) e.currentTarget.style.background = 'rgba(255,255,255,0.03)' }}
                      onMouseLeave={e => { if (!isSelected) e.currentTarget.style.background = 'transparent' }}>
                      <td className="px-4 py-3.5" style={{ color: 'var(--text)' }}>
                        <p className="truncate max-w-xs">{doc.title || doc.file_path}</p>
                      </td>
                      <td className="px-4 py-3.5 hidden md:table-cell truncate max-w-[140px]" style={{ color: 'var(--text-muted)' }}>
                        {doc.authors || '-'}
                      </td>
                      <td className="px-4 py-3.5 hidden lg:table-cell w-16" style={{ color: 'var(--text-muted)' }}>
                        {doc.year ?? '-'}
                      </td>
                      <td className="px-4 py-3.5 w-40"><StatusBadge status={doc.status} /></td>
                      <td className="px-4 py-3.5 hidden xl:table-cell w-16 text-right" style={{ color: 'var(--text-dim)' }}>
                        {doc.chunk_count ?? '-'}
                      </td>
                      <td className="px-4 py-3.5 w-10">
                        <button
                          onClick={e => { e.stopPropagation(); setCtxMenu({ x: e.currentTarget.getBoundingClientRect().left, y: e.currentTarget.getBoundingClientRect().bottom, doc }) }}
                          className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded-lg hover:bg-white/10"
                          style={{ color: 'var(--text-dim)' }}>
                          <MoreHorizontal size={15} />
                        </button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <AnimatePresence>
        {selected && <DetailPanel doc={selected} onClose={() => setSelectedDocId(null)} />}
      </AnimatePresence>

      {ctxMenu && (
        <ContextMenu menu={ctxMenu} onDelete={handleDelete} onRefresh={handleRefresh} onClose={() => setCtxMenu(null)} />
      )}
    </div>
  )
}
