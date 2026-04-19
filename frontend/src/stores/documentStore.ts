import { create } from 'zustand'
import type { Document } from '../types'

interface DocumentState {
  documents: Document[]
  loading: boolean
  setDocuments: (docs: Document[]) => void
  setLoading: (v: boolean) => void
  upsertDocument: (doc: Document) => void
  removeDocument: (docId: string) => void
}

export const useDocumentStore = create<DocumentState>((set) => ({
  documents: [],
  loading: false,
  setDocuments: (docs) => set({ documents: docs }),
  setLoading: (v) => set({ loading: v }),
  upsertDocument: (doc) =>
    set((s) => {
      const idx = s.documents.findIndex(d => d.doc_id === doc.doc_id)
      if (idx >= 0) {
        const updated = [...s.documents]
        updated[idx] = doc
        return { documents: updated }
      }
      return { documents: [...s.documents, doc] }
    }),
  removeDocument: (docId) =>
    set((s) => ({ documents: s.documents.filter(d => d.doc_id !== docId) })),
}))
