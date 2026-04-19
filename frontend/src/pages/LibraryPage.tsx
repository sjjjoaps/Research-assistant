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
  let cls = 'badge badge-default'
  if (s === 'processed') cls = 'badge badge-success'
  else if (s === 'failed') cls = 'badge badge-danger'
  else if (RUNNING_STATUS_SET.has(s)) cls = 'badge badge-warning'
  return (
    <span className={cls}>
      {RUNNING_STATUS_SET.has(s) && (
        <span style={{ display: 'inline-block', animation: 'spin 0.8s linear infinite' }}>⟳</span>
      )}
      {s}
    </span>
  )
}

type SortDir = 'asc' | 'desc' | null
function SortIcon({ dir }: { dir: SortDir }) {
  if (dir === 'asc') return <ChevronUp size={11} />
  if (dir === 'desc') return <ChevronDown size={11} />
  return <ChevronsUpDown size={11} style={{ opacity: 0.3 }} />
}
type SortKey = 'title' | 'authors' | 'year' | 'status' | 'chunk_count'

function DetailPanel({ doc, onClose }: { doc: Document; onClose: () => void }) {
  return (
    <motion.div
      initial={{ x: 40, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 40, opacity: 0 }}
      transition={{ duration: 0.2 }}
      style={{
        width: 360,
        borderLeft: '1px solid var(--border)',
        overflowY: 'auto',
        flexShrink: 0,
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--panel)',
      }}
    >
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '14px 18px',
        borderBottom: '1px solid var(--border)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text-muted)' }}>
          <FileText size={14} />
          <span style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--text)' }}>文献详情</span>
        </div>
        <button
          onClick={onClose}
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-dim)', transition: 'color 0.15s' }}
          onMouseEnter={e => (e.currentTarget.style.color = 'var(--text)')}
          onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-dim)')}
        >
          <X size={15} />
        </button>
      </div>

      <div style={{ padding: '18px', flex: 1, display: 'flex', flexDirection: 'column', gap: 18 }}>
        <div>
          <div className="section-label">标题</div>
          <p style={{ fontSize: 13.5, lineHeight: 1.6, color: 'var(--text)' }}>{doc.title || doc.file_path}</p>
        </div>

        {doc.authors && (
          <div>
            <div className="section-label">作者</div>
            <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>{doc.authors}</p>
          </div>
        )}

        {doc.institution && (
          <div>
            <div className="section-label">机构</div>
            <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>{doc.institution}</p>
          </div>
        )}

        <div style={{ display: 'flex', gap: 20 }}>
          {doc.year && (
            <div>
              <div className="section-label">年份</div>
              <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>{doc.year}</p>
            </div>
          )}
          <div>
            <div className="section-label">状态</div>
            <StatusBadge status={doc.status} />
          </div>
        </div>

        {doc.abstract && (
          <div>
            <div className="section-label">摘要</div>
            <p style={{ fontSize: 12.5, lineHeight: 1.7, color: 'var(--text-muted)' }}>{doc.abstract}</p>
          </div>
        )}

        {doc.keywords && (
          <div>
            <div className="section-label">关键词</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
              {doc.keywords.split(/[,;，；]/).map(k => k.trim()).filter(Boolean).map((k, i) => (
                <span key={i} className="tag">{k}</span>
              ))}
            </div>
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
          {([
            { label: 'Chunks', value: doc.chunk_count },
            { label: '实体',   value: doc.entity_count },
            { label: '关系',   value: doc.relation_count },
          ] as const).map(({ label, value }) => (
            <div key={label} className="stat-tile">
              <div className="stat-tile-value">{value ?? '—'}</div>
              <div className="stat-tile-label">{label}</div>
            </div>
          ))}
        </div>

        {doc.error_message && (
          <div style={{
            borderRadius: 'var(--r-md)',
            padding: '10px 12px',
            background: 'var(--danger-subtle)',
            border: '1px solid rgba(244,63,94,0.2)',
          }}>
            <p style={{ fontSize: 12.5, color: 'var(--danger)' }}>{doc.error_message}</p>
          </div>
        )}

        <p style={{ fontSize: 10, fontFamily: 'var(--font-mono)', wordBreak: 'break-all', color: 'var(--text-faint)' }}>
          {doc.doc_id}
        </p>
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
    const h = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose()
    }
    document.addEventListener('mousedown', h)
    return () => document.removeEventListener('mousedown', h)
  }, [onClose])

  return (
    <div ref={ref} style={{
      position: 'fixed', zIndex: 50,
      top: menu.y, left: menu.x,
      background: 'var(--surface2)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--r-md)',
      boxShadow: 'var(--shadow-lg)',
      padding: '4px',
      minWidth: 156,
    }}>
      {[
        { label: '刷新状态', icon: <RefreshCw size={13} />, action: () => { onRefresh(menu.doc); onClose() }, danger: false },
        { label: '删除文献', icon: <Trash2 size={13} />, action: () => { onDelete(menu.doc); onClose() }, danger: true },
      ].map(item => (
        <button
          key={item.label}
          onClick={item.action}
          style={{
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '7px 10px',
            fontSize: 13,
            borderRadius: 'var(--r-sm)',
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            color: item.danger ? 'var(--danger)' : 'var(--text-muted)',
            transition: 'background 0.12s',
          }}
          onMouseEnter={e => (e.currentTarget.style.background = item.danger ? 'var(--danger-subtle)' : 'var(--surface3)')}
          onMouseLeave={e => (e.currentTarget.style.background = 'none')}
        >
          {item.icon}{item.label}
        </button>
      ))}
    </div>
  )
}

export default function LibraryPage() {
  const { documents, setDocuments, loading, setLoading, removeDocument, upsertDocument } = useDocumentStore()
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

  useEffect(() => {
    const running = documents.filter(d => RUNNING_STATUS_SET.has(d.status ?? ''))
    if (running.length === 0) return
    const timer = setInterval(async () => {
      for (const doc of running) {
        try {
          const statusData = await getDocumentStatus(doc.doc_id ?? String(doc.id))
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
    { key: 'year', label: '年份', cls: 'hidden lg:table-cell' },
    { key: 'status', label: '状态' },
    { key: 'chunk_count', label: 'Chunks', cls: 'hidden xl:table-cell' },
  ]

  const filterChips = [
    { key: 'all'       as const, label: '全部',   count: counts.all      },
    { key: 'running'   as const, label: '处理中', count: counts.running  },
    { key: 'processed' as const, label: '已完成', count: counts.processed },
    { key: 'failed'    as const, label: '失败',   count: counts.failed   },
  ]

  return (
    <div style={{ display: 'flex', height: '100%', position: 'relative' }}>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, overflow: 'hidden' }}>
        {/* Toolbar */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10,
          padding: '10px 16px',
          borderBottom: '1px solid var(--border)',
          background: 'var(--panel)',
          flexShrink: 0,
        }}>
          <div style={{ position: 'relative', flex: 1, maxWidth: 280 }}>
            <Search size={13} style={{
              position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)',
              color: 'var(--text-dim)', pointerEvents: 'none',
            }} />
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="搜索标题 / 作者..."
              className="input"
              style={{ paddingLeft: 32, fontSize: 13 }}
              onFocus={e => {
                e.currentTarget.style.borderColor = 'var(--border-focus)'
                e.currentTarget.style.boxShadow = '0 0 0 3px var(--accent-subtle)'
              }}
              onBlur={e => {
                e.currentTarget.style.borderColor = 'var(--border-light)'
                e.currentTarget.style.boxShadow = 'none'
              }}
            />
          </div>

          <div style={{ display: 'flex', gap: 4 }}>
            {filterChips.map(chip => {
              const isActive = statusFilter === chip.key
              return (
                <button
                  key={chip.key}
                  onClick={() => setStatusFilter(chip.key)}
                  style={{
                    padding: '5px 10px',
                    borderRadius: 'var(--r-sm)',
                    fontSize: 12,
                    fontWeight: 500,
                    cursor: 'pointer',
                    border: `1px solid ${isActive ? 'rgba(91,110,245,0.3)' : 'var(--border-light)'}`,
                    background: isActive ? 'var(--accent-subtle)' : 'transparent',
                    color: isActive ? '#a5b4fc' : 'var(--text-dim)',
                    transition: 'all 0.15s',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {chip.label} <span style={{ opacity: 0.7, marginLeft: 2 }}>{chip.count}</span>
                </button>
              )
            })}
          </div>

          <button
            onClick={() => setShowUpload(v => !v)}
            className="btn btn-primary"
            style={{ fontSize: 13, padding: '6px 14px', marginLeft: 'auto' }}
          >
            <Upload size={13} />
            上传
          </button>
        </div>

        {/* Upload panel */}
        <AnimatePresence>
          {showUpload && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              style={{
                overflow: 'hidden',
                borderBottom: '1px solid var(--border)',
                flexShrink: 0,
                background: 'var(--surface)',
              }}
            >
              <div style={{ padding: '14px 16px' }}>
                <FileUploader />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Table */}
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {loading ? (
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              height: '100%', color: 'var(--text-dim)', gap: 8,
            }}>
              <span style={{ animation: 'spin 0.8s linear infinite', fontSize: 18 }}>⟳</span>
              加载中...
            </div>
          ) : sorted.length === 0 ? (
            <div style={{
              display: 'flex', flexDirection: 'column', alignItems: 'center',
              justifyContent: 'center', height: '100%', gap: 12, color: 'var(--text-dim)',
            }}>
              <div style={{
                width: 56, height: 56,
                borderRadius: 'var(--r-xl)',
                background: 'var(--surface)',
                border: '1px solid var(--border-light)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <FileText size={24} style={{ opacity: 0.3 }} />
              </div>
              <p style={{ fontSize: 13.5 }}>
                {search || statusFilter !== 'all' ? '无匹配文献' : '暂无文献，请上传 PDF'}
              </p>
            </div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  {cols.map(col => (
                    <th
                      key={col.key}
                      onClick={() => handleSort(col.key)}
                      className={col.cls}
                    >
                      <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        {col.label}
                        <SortIcon dir={sortKey === col.key ? sortDir : null} />
                      </span>
                    </th>
                  ))}
                  <th style={{ width: 36 }} />
                </tr>
              </thead>
              <tbody>
                {sorted.map(doc => {
                  const docKey = doc.doc_id ?? String(doc.id)
                  const isSelected = selectedDocId === docKey
                  return (
                    <tr
                      key={docKey}
                      onClick={() => setSelectedDocId(id => id === docKey ? null : docKey)}
                      onContextMenu={e => { e.preventDefault(); setCtxMenu({ x: e.clientX, y: e.clientY, doc }) }}
                      className={`group${isSelected ? ' selected' : ''}`}
                    >
                      <td>
                        <p style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 320 }}>
                          {doc.title || doc.file_path}
                        </p>
                      </td>
                      <td className="hidden md:table-cell" style={{ maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {doc.authors || '—'}
                      </td>
                      <td className="hidden lg:table-cell">{doc.year ?? '—'}</td>
                      <td><StatusBadge status={doc.status} /></td>
                      <td className="hidden xl:table-cell" style={{ textAlign: 'right' }}>
                        {doc.chunk_count ?? '—'}
                      </td>
                      <td>
                        <button
                          onClick={e => {
                            e.stopPropagation()
                            const r = e.currentTarget.getBoundingClientRect()
                            setCtxMenu({ x: r.left, y: r.bottom, doc })
                          }}
                          className="group-hover:opacity-100"
                          style={{
                            opacity: 0,
                            padding: '3px 4px',
                            borderRadius: 'var(--r-sm)',
                            color: 'var(--text-dim)',
                            background: 'none',
                            border: 'none',
                            cursor: 'pointer',
                            transition: 'all 0.15s',
                          }}
                          onMouseEnter={e => {
                            e.currentTarget.style.opacity = '1'
                            e.currentTarget.style.background = 'var(--surface2)'
                          }}
                          onMouseLeave={e => {
                            e.currentTarget.style.opacity = '0'
                            e.currentTarget.style.background = 'none'
                          }}
                        >
                          <MoreHorizontal size={14} />
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
