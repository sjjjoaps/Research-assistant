import { useEffect, useState } from 'react'
import { listDocuments, deleteDocument, getDocumentStatus } from '../api/client'
import { useDocumentStore } from '../stores/documentStore'
import FileUploader from '../components/FileUploader/FileUploader'
import type { Document } from '../types'

const statusBadge = (status: string) => {
  const map: Record<string, string> = {
    processed: 'bg-emerald-500/20 text-emerald-400',
    processing: 'bg-amber-500/20 text-amber-400',
    failed: 'bg-red-500/20 text-red-400',
    pending: 'bg-slate-500/20 text-slate-400',
  }
  return map[status] ?? 'bg-slate-500/20 text-slate-400'
}

export default function LibraryPage() {
  const { documents, setDocuments, loading, setLoading, removeDocument, upsertDocument } = useDocumentStore()
  const [selected, setSelected] = useState<Document | null>(null)

  useEffect(() => {
    setLoading(true)
    listDocuments()
      .then(data => setDocuments(data.documents ?? data ?? []))
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [setDocuments, setLoading])

  // Poll processing documents
  useEffect(() => {
    const processing = documents.filter(d => d.status === 'processing')
    if (processing.length === 0) return
    const timer = setInterval(async () => {
      for (const doc of processing) {
        try {
          const updated = await getDocumentStatus(doc.doc_id ?? String(doc.id))
          upsertDocument(updated)
        } catch {
          // ignore
        }
      }
    }, 3000)
    return () => clearInterval(timer)
  }, [documents, upsertDocument])

  const handleDelete = async (doc: Document) => {
    if (!confirm(`确认删除 ${doc.title || doc.file_path}？`)) return
    await deleteDocument(doc.doc_id ?? String(doc.id))
    removeDocument(doc.doc_id ?? String(doc.id))
    if (selected?.doc_id === doc.doc_id) setSelected(null)
  }

  return (
    <div className="flex h-full">
      <div className="flex-1 flex flex-col p-4 overflow-hidden">
        <div className="mb-4">
          <FileUploader />
        </div>

        {loading ? (
          <p className="text-slate-500 text-sm">加载中...</p>
        ) : documents.length === 0 ? (
          <p className="text-slate-500 text-sm">暂无文献，请上传 PDF 文件</p>
        ) : (
          <div className="overflow-y-auto flex-1">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-slate-500 uppercase border-b border-slate-700/50">
                  <th className="pb-2 pr-4">标题</th>
                  <th className="pb-2 pr-4">状态</th>
                  <th className="pb-2 pr-4">当前步骤</th>
                  <th className="pb-2"></th>
                </tr>
              </thead>
              <tbody>
                {documents.map(doc => (
                  <tr
                    key={doc.doc_id ?? doc.id}
                    onClick={() => setSelected(doc)}
                    className={`border-b border-slate-700/30 hover:bg-slate-700/20 cursor-pointer transition-colors ${
                      selected?.doc_id === doc.doc_id ? 'bg-slate-700/30' : ''
                    }`}
                  >
                    <td className="py-2.5 pr-4 text-slate-200 truncate max-w-xs">
                      {doc.title || doc.file_path}
                    </td>
                    <td className="py-2.5 pr-4">
                      <span className={`px-2 py-0.5 rounded-full text-xs ${statusBadge(doc.status ?? 'unknown')}`}>
                        {doc.status ?? 'unknown'}
                        {doc.status === 'processing' && (
                          <span className="ml-1 inline-block animate-spin">⟳</span>
                        )}
                      </span>
                    </td>
                    <td className="py-2.5 pr-4 text-slate-500 text-xs">{doc.current_step ?? '-'}</td>
                    <td className="py-2.5">
                      <button
                        onClick={e => { e.stopPropagation(); handleDelete(doc) }}
                        className="text-red-400/60 hover:text-red-400 text-xs"
                      >
                        删除
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Detail panel */}
      {selected && (
        <div className="w-72 border-l border-slate-700/50 p-4 overflow-y-auto shrink-0">
          <button onClick={() => setSelected(null)} className="text-slate-500 text-xs mb-3">← 关闭</button>
          <h2 className="text-sm font-semibold text-slate-200 mb-3">
            {selected.title ?? selected.file_path}
          </h2>
          <dl className="space-y-2 text-xs">
            {selected.authors && (
              <>
                <dt className="text-slate-500">作者</dt>
                <dd className="text-slate-300">{selected.authors}</dd>
              </>
            )}
            {selected.year && (
              <>
                <dt className="text-slate-500">年份</dt>
                <dd className="text-slate-300">{selected.year}</dd>
              </>
            )}
            <dt className="text-slate-500">状态</dt>
            <dd>
              <span className={`px-2 py-0.5 rounded-full ${statusBadge(selected.status ?? 'unknown')}`}>
                {selected.status ?? 'unknown'}
              </span>
            </dd>
            <dt className="text-slate-500">Chunk 数</dt>
            <dd className="text-slate-300">{selected.chunk_count ?? '-'}</dd>
            <dt className="text-slate-500 font-mono break-all">Doc ID</dt>
            <dd className="text-slate-400 font-mono break-all text-xs">{selected.doc_id}</dd>
          </dl>
        </div>
      )}
    </div>
  )
}
