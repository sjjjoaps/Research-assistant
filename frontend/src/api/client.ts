import type { SSEEvent } from '../types'

const BASE = ''  // proxied by Vite dev server

// ── Agent / Chat ──────────────────────────────────────────────────────────────

export async function* streamChat(
  sessionId: string,
  message: string,
): AsyncGenerator<SSEEvent> {
  const res = await fetch(`${BASE}/agent/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ session_id: sessionId, user_input: message }),
  })
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`)

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''

  const parseBlock = function* (block: string): Generator<SSEEvent> {
    const lines = block.split('\n')
    let eventType = ''
    for (const line of lines) {
      if (line.startsWith('event:')) {
        eventType = line.slice(6).trim()
      } else if (line.startsWith('data:')) {
        try {
          const parsed = JSON.parse(line.slice(5).trim()) as SSEEvent
          // event: line overrides type field in data
          if (eventType) (parsed as Record<string, unknown>).type = eventType
          yield parsed
        } catch {
          // skip malformed
        }
      }
    }
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) {
      // flush remaining buffer
      if (buf.trim()) yield* parseBlock(buf)
      break
    }
    buf += decoder.decode(value, { stream: true })
    const parts = buf.split('\n\n')
    buf = parts.pop() ?? ''
    for (const part of parts) {
      if (part.trim()) yield* parseBlock(part)
    }
  }
}

export async function listSessions() {
  const res = await fetch(`${BASE}/agent/sessions`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function deleteSession(sessionId: string) {
  const res = await fetch(`${BASE}/agent/sessions/${sessionId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function getSessionHistory(sessionId: string) {
  const res = await fetch(`${BASE}/agent/sessions/${sessionId}/history`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

// ── Documents ─────────────────────────────────────────────────────────────────

export async function listDocuments(params?: { page?: number; page_size?: number }) {
  const qs = new URLSearchParams()
  if (params?.page) qs.set('page', String(params.page))
  if (params?.page_size) qs.set('page_size', String(params.page_size))
  const res = await fetch(`${BASE}/documents?${qs}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function uploadDocument(file: File) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/documents/upload`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function getDocumentStatus(docId: string) {
  const res = await fetch(`${BASE}/documents/${docId}/status`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function deleteDocument(docId: string) {
  const res = await fetch(`${BASE}/documents/${docId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

// ── Graph ─────────────────────────────────────────────────────────────────────

export async function getGraphStats() {
  const res = await fetch(`${BASE}/graph/stats`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function getSubgraph(params?: {
  limit?: number
  node_types?: string[]
  search?: string
}) {
  const qs = new URLSearchParams()
  if (params?.limit) qs.set('limit', String(params.limit))
  if (params?.search) qs.set('search', params.search)
  params?.node_types?.forEach(t => qs.append('node_types', t))
  const res = await fetch(`${BASE}/graph/subgraph?${qs}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function getGraphNodes(params?: { node_types?: string[]; search?: string; limit?: number }) {
  const qs = new URLSearchParams()
  if (params?.search) qs.set('search', params.search)
  if (params?.limit) qs.set('limit', String(params.limit))
  params?.node_types?.forEach(t => qs.append('node_types', t))
  const res = await fetch(`${BASE}/graph/nodes?${qs}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function getGraphEdges(params?: { node_id?: string; node_types?: string[]; limit?: number }) {
  const qs = new URLSearchParams()
  if (params?.node_id) qs.set('node_id', params.node_id)
  if (params?.limit) qs.set('limit', String(params.limit))
  params?.node_types?.forEach(t => qs.append('node_types', t))
  const res = await fetch(`${BASE}/graph/edges?${qs}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

// ── Community ─────────────────────────────────────────────────────────────────

export async function detectCommunities(minCommunitySize: number) {
  const res = await fetch(`${BASE}/community/detect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ min_community_size: minCommunitySize }),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}
