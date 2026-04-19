import { useEffect, useRef, useState } from 'react'
import { Graph } from '@antv/g6'
import { getSubgraph, getGraphStats } from '../api/client'
import type { GraphStats } from '../types'

export default function GraphPage() {
  const containerRef = useRef<HTMLDivElement>(null)
  const graphRef = useRef<Graph | null>(null)
  const [stats, setStats] = useState<GraphStats | null>(null)
  const [search, setSearch] = useState('')
  const [nodeTypes, setNodeTypes] = useState<string[]>([])
  const [selectedTypes, setSelectedTypes] = useState<string[]>([])
  const [loading, setLoading] = useState(false)

  const palette = ['#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#ec4899']

  const loadGraph = async () => {
    if (!containerRef.current) return
    setLoading(true)
    try {
      const data = await getSubgraph({
        limit: 200,
        search: search || undefined,
        node_types: selectedTypes.length > 0 ? selectedTypes : undefined,
      })

      const rawNodes: { id: string; label?: string; name?: string; type?: string }[] = data.nodes ?? []
      const rawEdges: { source: string; target: string; label?: string; relation_type?: string }[] = data.edges ?? []

      const types = Array.from(new Set(rawNodes.map(n => n.type ?? 'Entity').filter(Boolean))) as string[]
      setNodeTypes(types)

      const colorMap: Record<string, string> = {}
      types.forEach((t, i) => { colorMap[t] = palette[i % palette.length] })

      if (graphRef.current) {
        graphRef.current.destroy()
        graphRef.current = null
      }

      const graph = new Graph({
        container: containerRef.current,
        width: containerRef.current.clientWidth,
        height: containerRef.current.clientHeight,
        autoFit: 'view',
        layout: { type: 'force', preventOverlap: true, nodeSpacing: 20 },
        node: {
          style: {
            size: 28,
            labelText: (d: { id: string; label?: string; name?: string }) => d.label ?? d.name ?? d.id,
            labelFill: '#cbd5e1',
            labelFontSize: 10,
            fill: (d: { type?: string }) => colorMap[d.type ?? 'Entity'] ?? '#8b5cf6',
            stroke: 'transparent',
          },
        },
        edge: {
          style: {
            stroke: '#334155',
            lineWidth: 1,
            labelText: (d: Record<string, unknown>) => String((d.label ?? d.relation_type) ?? ''),
            labelFill: '#64748b',
            labelFontSize: 9,
          },
        },
        behaviors: ['drag-canvas', 'zoom-canvas', 'drag-element'],
        data: {
          nodes: rawNodes.map(n => ({ id: n.id, label: n.label ?? n.name ?? n.id, type: n.type })),
          edges: rawEdges.map((e, i) => ({
            id: `e${i}`,
            source: e.source,
            target: e.target,
            label: e.label ?? e.relation_type ?? '',
          })),
        },
      })

      await graph.render()
      graphRef.current = graph
    } catch (e) {
      console.error('Failed to load graph:', e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    getGraphStats().then(setStats).catch(console.error)
    loadGraph()
    return () => { graphRef.current?.destroy() }
  }, [])

  const toggleType = (type: string) => {
    setSelectedTypes(prev =>
      prev.includes(type) ? prev.filter(t => t !== type) : [...prev, type]
    )
  }

  return (
    <div className="flex h-full">
      <div className="w-60 border-r border-slate-700/50 p-4 flex flex-col gap-4 shrink-0 overflow-y-auto">
        <div>
          <h3 className="text-xs text-slate-500 uppercase tracking-wider mb-2">图谱统计</h3>
          {stats && (
            <dl className="grid grid-cols-2 gap-1 text-xs">
              <dt className="text-slate-500">节点</dt>
              <dd className="text-slate-200">{stats.node_count}</dd>
              <dt className="text-slate-500">关系</dt>
              <dd className="text-slate-200">{stats.relationship_count}</dd>
            </dl>
          )}
        </div>

        <div>
          <h3 className="text-xs text-slate-500 uppercase tracking-wider mb-2">搜索</h3>
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="节点名称..."
            className="w-full bg-[#1e2130] border border-slate-600 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-violet-500"
          />
        </div>

        {nodeTypes.length > 0 && (
          <div>
            <h3 className="text-xs text-slate-500 uppercase tracking-wider mb-2">节点类型</h3>
            <div className="flex flex-col gap-1">
              {nodeTypes.map(type => (
                <label key={type} className="flex items-center gap-2 text-xs cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedTypes.includes(type)}
                    onChange={() => toggleType(type)}
                    className="accent-violet-500"
                  />
                  <span className="text-slate-300">{type}</span>
                </label>
              ))}
            </div>
          </div>
        )}

        <button
          onClick={loadGraph}
          disabled={loading}
          className="mt-auto px-3 py-2 bg-violet-600 hover:bg-violet-500 disabled:opacity-40 rounded-lg text-xs font-medium transition-colors"
        >
          {loading ? '加载中...' : '刷新图谱'}
        </button>
      </div>

      <div ref={containerRef} className="flex-1 bg-[#0a0d16]" />
    </div>
  )
}
