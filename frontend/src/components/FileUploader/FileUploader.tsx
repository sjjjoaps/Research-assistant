import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { AnimatePresence } from 'framer-motion'
import { uploadDocument } from '../../api/client'
import { useDocumentStore } from '../../stores/documentStore'
import ParseProgress from '../ParseProgress/ParseProgress'
import type { Document } from '../../types'

interface UploadEntry {
  docId: string
  fileName: string
}

export default function FileUploader() {
  const { upsertDocument } = useDocumentStore()
  const [uploads, setUploads] = useState<UploadEntry[]>([])
  const [uploading, setUploading] = useState(false)

  const onDrop = useCallback(async (files: File[]) => {
    setUploading(true)
    for (const file of files) {
      try {
        const result = await uploadDocument(file)
        if (result.doc_id) {
          upsertDocument({
            id: 0,
            doc_id: result.doc_id,
            file_path: result.file_path ?? file.name,
            title: file.name,
            authors: '',
            institution: null,
            year: null,
            abstract: null,
            keywords: '',
            status: 'processing',
            current_step: '上传成功，等待解析',
            chunk_count: 0,
            entity_count: 0,
            relation_count: 0,
            error_message: null,
            updated_at: null,
          } satisfies Document)
          setUploads(prev => [...prev, { docId: result.doc_id, fileName: file.name }])
        }
      } catch (e) {
        console.error('Upload failed:', e)
      }
    }
    setUploading(false)
  }, [upsertDocument])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'], 'text/plain': ['.txt'], 'text/markdown': ['.md'] },
  })

  const removeEntry = (docId: string) =>
    setUploads(prev => prev.filter(u => u.docId !== docId))

  return (
    <div className="flex flex-col gap-3">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-colors ${
          isDragActive
            ? 'border-violet-500 bg-violet-500/10'
            : 'border-slate-600 hover:border-slate-400'
        }`}
      >
        <input {...getInputProps()} />
        <p className="text-slate-400 text-sm">
          {uploading
            ? '上传中...'
            : isDragActive
            ? '松开以上传文件'
            : '拖拽文件到此处，或点击选择（PDF / TXT / MD）'}
        </p>
      </div>

      <AnimatePresence>
        {uploads.map(u => (
          <ParseProgress
            key={u.docId}
            docId={u.docId}
            fileName={u.fileName}
            onDone={() => removeEntry(u.docId)}
          />
        ))}
      </AnimatePresence>
    </div>
  )
}
