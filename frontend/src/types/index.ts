// SSE event types matching backend src/agents/events.py
export type EventType =
  | 'session_start'
  | 'thinking'
  | 'tool_start'
  | 'tool_end'
  | 'text_delta'
  | 'sources'
  | 'usage'
  | 'done'
  | 'error'

export interface BaseEvent {
  type: EventType
}

export interface SessionStartEvent extends BaseEvent {
  type: 'session_start'
  session_id: string
}

export interface ThinkingEvent extends BaseEvent {
  type: 'thinking'
  iteration: number
}

export interface ToolStartEvent extends BaseEvent {
  type: 'tool_start'
  tool_name: string
  display_message: string
}

export interface ToolEndEvent extends BaseEvent {
  type: 'tool_end'
  tool_name: string
  result_summary: string
  elapsed_ms: number
}

export interface TextDeltaEvent extends BaseEvent {
  type: 'text_delta'
  delta: string
}

export interface SourcesEvent extends BaseEvent {
  type: 'sources'
  sources: Source[]
}

export interface UsageEvent extends BaseEvent {
  type: 'usage'
  total_tokens: number
  prompt_tokens: number
  completion_tokens: number
  estimated_cost_cny: number
}

export interface DoneEvent extends BaseEvent {
  type: 'done'
  session_id: string
}

export interface ErrorEvent extends BaseEvent {
  type: 'error'
  message: string
  code: string
}

export type SSEEvent =
  | SessionStartEvent
  | ThinkingEvent
  | ToolStartEvent
  | ToolEndEvent
  | TextDeltaEvent
  | SourcesEvent
  | UsageEvent
  | DoneEvent
  | ErrorEvent

export interface Source {
  file_path: string
  chunk_index: number
  content?: string
  section_type?: string
}

// Matches backend DocumentItem schema
export interface Document {
  id: number
  doc_id: string | null
  file_path: string
  title: string
  authors: string        // comma-separated string, not array
  institution: string | null
  year: number | null
  abstract: string | null
  keywords: string
  status: string | null
  current_step: string | null
  chunk_count: number
  entity_count: number
  relation_count: number
  error_message: string | null
  updated_at: string | null
}

// Matches backend GraphNode schema
export interface GraphNode {
  id: string
  label: string
  type: string
  labels: string[]
  properties: Record<string, unknown>
}

// Matches backend GraphEdge schema
export interface GraphEdge {
  id: string
  source: string
  target: string
  type: string
  label: string
  properties: Record<string, unknown>
}

// Matches backend GraphStatsResponse schema
export interface GraphCountItem {
  label: string
  count: number
}

export interface GraphStats {
  node_count: number
  relationship_count: number
  node_labels: GraphCountItem[]
  relationship_types: GraphCountItem[]
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  tool_calls?: ToolCall[]
  sources?: Source[]
}

export interface ToolCall {
  tool_name: string
  status: 'loading' | 'done' | 'error'
  display_message?: string
  result_summary?: string
  elapsed_ms?: number
}

// Matches backend SessionMeta schema
export interface Session {
  session_id: string
  title: string
  created_at: string
  updated_at: string
  turn_count: number
  total_tokens: number
  total_cost_cny: number
}
