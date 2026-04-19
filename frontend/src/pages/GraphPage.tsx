import { useEffect, useRef, useState, useCallback } from 'react'
import { Graph } from '@antv/g6'
import type { IElementEvent } from '@antv/g6'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, RefreshCw, X, Network } from 'lucide-react'
import { getSubgraph, getGraphStats } from '../api/client'
import type { GraphStats, GraphNode, GraphEdge } from '../types'

const PALETTE = ['#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#f97316', '#a78bfa']

type DetailItem = { kind: 'node'; data: GraphNode } | { kind: 'edge'; data: GraphEdge }

function PropList({ obj }: { obj: Record<string, unknown> }) {
  const entries = Object.entries(obj).filter(([, v]) => v !== null && v !== undefined && v !== '')
  if (!entries.length) return null
  return (
    <dl className="space-y-2">
      {entries.map(([k, v]) => (
        <div key={k}>
          <dt className="text-xs uppercase tracking-wide" style={{ color: 'var(--text-dim)' }}>{k}</dt>
          <dd className="text-sm mt-0.5 break-all" style={{ color: 'var(--text-muted)' }}>{String(v)}</dd>
        </div>
      ))}
    </dl>
  )
}

function DetailPanel({ item, onClose }: { item: DetailItem; onClose: () => void }) {
  return (
    <motion.div initial={{ x: 40, opacity: 0 }} animate={{ x: 0, opacity: 1 }}
      exit={{ x: 40, opacity: 0 }} transition={{ duration: 0.16 }}
      className="absolute right-0 top-0 bottom-0 w-72 border-l overflow-y-auto z-10 flex flex-col"
      style={{ background: 'var(--panel)', borderColor: 'var(--border)' }}>
      <div className="flex items-center justify-between px-4 py-3 border-b shrink-0" style={{ borderColor: 'var(--border)' }}>
        <span className="text-sm font-medium" style={{ color: 'var(--text-muted)' }}>
          {item.kind === 'node' ? '节点详情' : '关系详情'}
        </span>
        <button onClick={onClose} style={{ color: 'var(--text-dim)' }} className="hover:text-white transition-colors">
          <X size={15} />
        </button>
      </div>
      <div className="p-4 space-y-4 flex-1">
        {item.kind === 'node' ? (
          <>
            <div>
              <p className="text-xs uppercase tracking-wide mb-1" style={{ color: 'var(--text-dim)' }}>标签</p>
              <p className="text-sm font-semibold" style={{ color: 'var(--text)' }}>{item.data.label ?? item.data.id}</p>
            </div>
            {item.data.type && (
              <div>
                <p className="text-xs uppercase tracking-wide mb-1" style={{ color: 'var(--text-dim)' }}>类型</p>
                <span className="px-2 py-0.5 rounded-lg text-xs bg-violet-500/15 text-violet-300">{item.data.type}</span>
              </div>
            )}
            {item.data.labels?.length > 0 && (
              <div>
                <p className="text-xs uppercase tracking-wide mb-1.5" style={{ color: 'var(--text-dim)' }}>标签组</p>
                <div className="flex flex-wrap gap-1">
                  {item.data.labels.map(l => (
                    <span key={l} className="px-2 py-0.5 rounded-lg text-xs" style={{ background: 'var(--surface)', color: 'var(--text-muted)' }}>{l}</span>
                  ))}
                </div>
              </div>
            )}
            {item.data.properties && Object.keys(item.data.properties).length > 0 && (
              <div>
                <p className="text-xs uppercase tracking-wide mb-2" style={{ color: 'var(--text-dim)' }}>属性</p>
                <PropList obj={item.data.properties as Record<string, unknown>} />
              </div>
            )}
          </>
        ) : (
          <>
            <div>
              <p className="text-xs uppercase tracking-wide mb-1" style={{ color: 'var(--text-dim)' }}>关系类型</p>
              <p className="text-sm font-semibold" style={{ color: 'var(--text)' }}>{item.data.label ?? item.data.type ?? '-'}</p>
            </div>
            <div>
              <p className="text-xs uppercase tracking-wide mb-1" style={{ color: 'var(--text-dim)' }}>起点 → 终点</p>
              <p className="text-sm font-mono break-all" style={{ color: 'var(--text-muted)' }}>
                {item.data.source} → {item.data.target}
              </p>
            </div>
            {item.data.properties && Object.keys(item.data.properties).length > 0 && (
              <div>
                <p className="text-xs uppercase tracking-wide mb-2" style={{ color: 'var(--text-dim)' }}>属性</p>
                <PropList obj={item.data.properties as Record<string, unknown>} />
              </div>
            )}
          </>
        )}
      </div>
    </motion.div>
  )
}

export default function GraphPage() {
  const containerRef = useRef<HTMLDivElement>(null)
  const graphRef = useRef<Graph | null>(null)
  const [stats, setStats] = useState<GraphStats | null>(null)
  // All known types come from stats (stable), not from current graph data
  const [allNodeTypes, setAllNodeTypes] = useState<string[]>([])
  const [selectedTypes, setSelectedTypes] = useState<string[]>([])
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(false)
  const [detail, setDetail] = useState<DetailItem | null>(null)

  const rawNodesRef = useRef<GraphNode[]>([])
  const rawEdgesRef = useRef<GraphEdge[]>([])
  const colorMapRef = useRef<Record<string, string>>({})

  // Build color map from all known types (from stats) for visual consistency
  const buildColorMap = useCallback((types: string[]) => {
    const map: Record<string, string> = {}
    types.forEach((t, i) => { map[t] = PALETTE[i % PALETTE.length] })
    colorMapRef.current = map
    return map
  }, [])

  const loadGraph = useCallback(async (types: string[], srch: string) => {
    if (!containerRef.current) return
    setLoading(true)
    setDetail(null)
    try {
      const data = await getSubgraph({
        limit: 300,
        search: srch || undefined,
        node_types: types.length > 0 ? types : undefined,
      })

      const rawNodes: GraphNode[] = data.nodes ?? []
      const rawEdges: GraphEdge[] = data.edges ?? []
      rawNodesRef.current = rawNodes
      rawEdgesRef.current = rawEdges

      const colorMap = colorMapRef.current

      if (graphRef.current) { graphRef.current.destroy(); graphRef.current = null }

      const graph = new Graph({
        container: containerRef.current,
        width: containerRef.current.clientWidth,
        height: containerRef.current.clientHeight,
        autoFit: 'view',
        layout: { type: 'force', preventOverlap: true, nodeSpacing: 25, linkDistance: 80 },
        node: {
          style: {
            size: 32,
            labelText: (d: Record<string, unknown>) => String(d.label ?? d.id ?? ''),
            labelFill: '#cbd5e1',
            labelFontSize: 12,
            fill: (d: Record<string, unknown>) => colorMap[String(d.type ?? 'Entity')] ?? '#8b5cf6',
            stroke: 'transparent',
            cursor: 'pointer',
          },
          state: {
            highlight: { stroke: '#8b5cf6', lineWidth: 3, shadowColor: '#8b5cf6', shadowBlur: 12 },
            dim: { fillOpacity: 0.12, labelOpacity: 0.15 },
          },
        },
        edge: {
          style: {
            stroke: '#334155',
            lineWidth: 1.5,
            labelText: (d: Record<string, unknown>) => String(d.label ?? ''),
            labelFill: '#475569',
            labelFontSize: 10,
            cursor: 'pointer',
          },
          state: {
            highlight: { stroke: '#8b5cf6', lineWidth: 2.5 },
            dim: { strokeOpacity: 0.08 },
          },
        },
        behaviors: ['drag-canvas', 'zoom-canvas', 'drag-element'],
        data: {
          nodes: rawNodes.map(n => ({ id: n.id, label: n.label ?? n.id, type: n.type })),
          edges: rawEdges.map((e, i) => ({
            id: e.id ?? `e${i}`,
            source: e.source,
            target: e.target,
            label: e.label ?? e.type ?? '',
          })),
        },
      })

      // G6 v5: event.target.id is the element ID
      graph.on('node:click', (evt: IElementEvent) => {
        const nodeId = evt.target.id
        const raw = rawNodesRef.current.find(n => n.id === nodeId)
        if (raw) setDetail({ kind: 'node', data: raw })
        const neighborIds = new Set<string>()
        rawEdgesRef.current.forEach(e => {
          if (e.source === nodeId) neighborIds.add(e.target)
          if (e.target === nodeId) neighborIds.add(e.source)
        })
        rawNodesRef.current.forEach(n => {
          graph.setElementState(n.id, n.id === nodeId || neighborIds.has(n.id) ? 'highlight' : 'dim')
        })
      })

      graph.on('edge:click', (evt: IElementEvent) => {
        const edgeId = evt.target.id
        const raw = rawEdgesRef.current.find(e => (e.id ?? '') === edgeId)
          ?? rawEdgesRef.current.find((_, i) => `e${i}` === edgeId)
        if (raw) setDetail({ kind: 'edge', data: raw })
      })

      graph.on('canvas:click', () => {
        rawNodesRef.current.forEach(n => graph.setElementState(n.id, []))
        setDetail(null)
      })

      await graph.render()
      graphRef.current = graph
    } catch (e) {
      console.error('Graph load error:', e)
    } finally {
      setLoading(false)
    }
  }, [])

  // On mount: load stats first to get stable type list, then load graph
  useEffect(() => {
    getGraphStats().then(s => {
      setStats(s)
      const types = (s.node_labels ?? []).map((l: { label: string }) => l.label)
      setAllNodeTypes(types)
      buildColorMap(types)
      loadGraph([], '')
    }).catch(() => loadGraph([], ''))
    return () => { graphRef.current?.destroy() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Auto-reload when type filter changes
  useEffect(() => {
    if (allNodeTypes.length === 0) return  // skip initial empty state before stats loaded
    loadGraph(selectedTypes, search)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTypes])

  const handleSearch = () => loadGraph(selectedTypes, search)

  const toggleType = (type: string) =>
    setSelectedTypes(prev => prev.includes(type) ? prev.filter(t => t !== type) : [...prev, type])

  return (
    <div className="flex h-full relative">
      {/* Control panel */}
      <div className="w-60 border-r shrink-0 flex flex-col gap-4 p-4 overflow-y-auto"
        style={{ background: 'var(--panel)', borderColor: 'var(--border)' }}>
        <div>
          <h3 className="text-xs uppercase tracking-wider mb-2 font-medium" style={{ color: 'var(--text-dim)' }}>图谱统计</h3>
          {stats ? (
            <dl className="grid grid-cols-2 gap-x-3 gap-y-2 text-sm">
              <dt style={{ color: 'var(--text-dim)' }}>节点</dt>
              <dd style={{ color: 'var(--text)' }}>{stats.node_count.toLocaleString()}</dd>
              <dt style={{ color: 'var(--text-dim)' }}>关系</dt>
              <dd style={{ color: 'var(--text)' }}>{stats.relationship_count.toLocaleString()}</dd>
            </dl>
          ) : <p className="text-sm" style={{ color: 'var(--text-dim)' }}>加载中...</p>}
        </div>

        <div>
          <h3 className="text-xs uppercase tracking-wider mb-2 font-medium" style={{ color: 'var(--text-dim)' }}>节点搜索</h3>
          <div className="relative">
            <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-dim)' }} />
            <input value={search} onChange={e => setSearch(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSearch()}
              placeholder="输入关键词..."
              className="w-full pl-8 pr-3 py-2 rounded-xl text-sm outline-none"
              style={{ background: 'var(--surface)', color: 'var(--text)', border: '1px solid var(--border)' }}
              onFocus={e => (e.currentTarget.style.borderColor = 'rgba(124,58,237,0.6)')}
              onBlur={e => (e.currentTarget.style.borderColor = 'var(--border)')} />
          </div>
        </div>

        {allNodeTypes.length > 0 && (
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-xs uppercase tracking-wider font-medium" style={{ color: 'var(--text-dim)' }}>节点类型</h3>
              {selectedTypes.length > 0 && (
                <button onClick={() => setSelectedTypes([])} className="text-xs transition-colors"
                  style={{ color: 'var(--text-dim)' }}
                  onMouseEnter={e => (e.currentTarget.style.color = 'var(--text)')}
                  onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-dim)')}>清除</button>
              )}
            </div>
            <div className="flex flex-col gap-2">
              {allNodeTypes.map((type, i) => {
                const active = selectedTypes.length === 0 || selectedTypes.includes(type)
                return (
                  <button key={type} onClick={() => toggleType(type)}
                    className="flex items-center gap-2.5 text-left transition-colors"
                    style={{ color: active ? 'var(--text)' : 'var(--text-dim)' }}>
                    <span className="w-3 h-3 rounded-full shrink-0 transition-opacity"
                      style={{ background: PALETTE[i % PALETTE.length], opacity: active ? 1 : 0.3 }} />
                    <span className="text-sm">{type}</span>
                    {selectedTypes.includes(type) && (
                      <span className="ml-auto text-xs text-violet-400">✓</span>
                    )}
                  </button>
                )
              })}
            </div>
          </div>
        )}

        <button onClick={handleSearch} disabled={loading}
          className="mt-auto flex items-center justify-center gap-2 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors disabled:opacity-40"
          style={{ background: 'var(--accent)', color: '#fff' }}
          onMouseEnter={e => { if (!loading) e.currentTarget.style.background = 'var(--accent-hover)' }}
          onMouseLeave={e => (e.currentTarget.style.background = 'var(--accent)')}>
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          {loading ? '加载中...' : '刷新图谱'}
        </button>
      </div>

      {/* Graph canvas */}
      <div className="flex-1 relative overflow-hidden">
        <div ref={containerRef} className="w-full h-full"
          style={{ background: 'radial-gradient(ellipse at center, #0d1020 0%, #080b13 100%)' }} />

        {!loading && rawNodesRef.current.length === 0 && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 pointer-events-none" style={{ color: 'var(--text-dim)' }}>
            <Network size={40} className="opacity-20" />
            <p className="text-sm">暂无图谱数据，请先上传并解析文献</p>
          </div>
        )}

        <AnimatePresence>
          {detail && (
            <DetailPanel item={detail} onClose={() => {
              setDetail(null)
              if (graphRef.current) rawNodesRef.current.forEach(n => graphRef.current!.setElementState(n.id, []))
            }} />
          )}
        </AnimatePresence>

        <div className="absolute bottom-3 left-3 text-xs px-2.5 py-1.5 rounded-lg pointer-events-none"
          style={{ background: 'rgba(0,0,0,0.55)', color: 'var(--text-dim)' }}>
          点击节点/关系查看详情 · 滚轮缩放 · 拖拽平移
        </div>
      </div>
    </div>
  )
}
