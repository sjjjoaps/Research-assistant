import { useEffect, useRef, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, RefreshCw, X, Network, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react'
import { getSubgraph, getGraphStats } from '../api/client'
import type { GraphStats, GraphNode, GraphEdge } from '../types'

// ── Palette ───────────────────────────────────────────────────────────────────
const PALETTE = ['#5b6ef5', '#22d3ee', '#10b981', '#f59e0b', '#f43f5e', '#ec4899', '#f97316', '#a78bfa']

// ── Physics ───────────────────────────────────────────────────────────────────
const REPULSION   = 4800
const SPRING_K    = 0.028
const SPRING_LEN  = 100
const DAMPING     = 0.86
const GRAVITY     = 0.01
const MAX_VEL     = 10
const NODE_R      = 7   // base node radius

interface SimNode extends GraphNode {
  x: number; y: number; vx: number; vy: number
}

type DetailItem = { kind: 'node'; data: GraphNode } | { kind: 'edge'; data: GraphEdge }

// ── PropList ──────────────────────────────────────────────────────────────────
function PropList({ obj }: { obj: Record<string, unknown> }) {
  const entries = Object.entries(obj).filter(([, v]) => v !== null && v !== undefined && v !== '')
  if (!entries.length) return null
  return (
    <dl style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {entries.map(([k, v]) => (
        <div key={k}>
          <dt style={{ fontSize: 10.5, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-dim)', marginBottom: 2 }}>{k}</dt>
          <dd style={{ fontSize: 12.5, color: 'var(--text-muted)', wordBreak: 'break-all' }}>{String(v)}</dd>
        </div>
      ))}
    </dl>
  )
}

// ── useForceGraph (custom Canvas force-directed graph) ────────────────────────
function useForceGraph(canvasRef: React.RefObject<HTMLCanvasElement | null>, wrapperRef: React.RefObject<HTMLDivElement | null>) {
  const rafRef       = useRef(0)
  const simNodes     = useRef<SimNode[]>([])
  const simEdges     = useRef<GraphEdge[]>([])
  const colorMap     = useRef<Record<string, string>>({})
  const scale        = useRef(1)
  const offset       = useRef({ x: 0, y: 0 })
  const dragging     = useRef<{ nodeIdx: number } | null>(null)
  const panning      = useRef<{ sx: number; sy: number; ox: number; oy: number } | null>(null)
  const hoveredNode       = useRef(-1)
  const selectedNode      = useRef(-1)
  const selectedEdge      = useRef(-1)
  const gravityCenterRef  = useRef({ x: 0, y: 0 })  // fixed world-space gravity anchor

  const toWorld = useCallback((cx: number, cy: number) => ({
    x: (cx - offset.current.x) / scale.current,
    y: (cy - offset.current.y) / scale.current,
  }), [])

  const hitNode = useCallback((cx: number, cy: number) => {
    const { x, y } = toWorld(cx, cy)
    for (let i = simNodes.current.length - 1; i >= 0; i--) {
      const n = simNodes.current[i]
      if ((n.x - x) ** 2 + (n.y - y) ** 2 <= (NODE_R + 4) ** 2) return i
    }
    return -1
  }, [toWorld])

  const hitEdge = useCallback((cx: number, cy: number) => {
    const { x, y } = toWorld(cx, cy)
    const nodes = simNodes.current, edges = simEdges.current
    for (let i = 0; i < edges.length; i++) {
      const e = edges[i]
      const s = nodes.find(n => n.id === e.source)
      const t = nodes.find(n => n.id === e.target)
      if (!s || !t) continue
      const dx = t.x - s.x, dy = t.y - s.y
      const len2 = dx * dx + dy * dy
      if (len2 === 0) continue
      const tc = Math.max(0, Math.min(1, ((x - s.x) * dx + (y - s.y) * dy) / len2))
      const px = s.x + tc * dx - x, py = s.y + tc * dy - y
      if (px * px + py * py < 36) return i
    }
    return -1
  }, [toWorld])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!
    const W = canvas.width, H = canvas.height
    ctx.clearRect(0, 0, W, H)

    // background gradient
    const grad = ctx.createRadialGradient(W * 0.38, H * 0.32, 0, W * 0.5, H * 0.5, Math.max(W, H) * 0.75)
    grad.addColorStop(0, '#0d1628')
    grad.addColorStop(1, '#060a10')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, W, H)

    ctx.save()
    ctx.translate(offset.current.x, offset.current.y)
    ctx.scale(scale.current, scale.current)

    const nodes = simNodes.current, edges = simEdges.current
    const sIdx = selectedNode.current, hIdx = hoveredNode.current, sEIdx = selectedEdge.current
    const cmap = colorMap.current

    // neighbors of selected node
    const neighborSet = new Set<string>()
    if (sIdx >= 0) {
      const sid = nodes[sIdx]?.id
      if (sid) edges.forEach(e => { if (e.source === sid) neighborSet.add(e.target); if (e.target === sid) neighborSet.add(e.source) })
    }
    const dimActive = sIdx >= 0

    // ── draw edges ─────────────────────────────────────────────────────────────
    edges.forEach((e, ei) => {
      const s = nodes.find(n => n.id === e.source)
      const t = nodes.find(n => n.id === e.target)
      if (!s || !t) return
      const isHighlight = dimActive && (nodes[sIdx]?.id === e.source || nodes[sIdx]?.id === e.target)
      const isSelected  = ei === sEIdx
      const dim = dimActive && !isHighlight

      ctx.beginPath()
      ctx.moveTo(s.x, s.y)
      ctx.lineTo(t.x, t.y)
      if (isSelected)        { ctx.strokeStyle = 'rgba(91,110,245,0.95)'; ctx.lineWidth = 2 }
      else if (isHighlight)  { ctx.strokeStyle = 'rgba(91,110,245,0.6)';  ctx.lineWidth = 1.5 }
      else if (dim)          { ctx.strokeStyle = 'rgba(255,255,255,0.03)'; ctx.lineWidth = 0.8 }
      else                   { ctx.strokeStyle = 'rgba(255,255,255,0.13)'; ctx.lineWidth = 1 }
      ctx.stroke()

      if ((isHighlight || isSelected) && e.label) {
        const mx = (s.x + t.x) / 2, my = (s.y + t.y) / 2
        const fs = 10 / scale.current
        ctx.font = `${fs}px JetBrains Mono, monospace`
        ctx.fillStyle = 'rgba(148,163,184,0.8)'
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
        ctx.fillText(e.label.length > 22 ? e.label.slice(0, 20) + '\u2026' : e.label, mx, my - 9 / scale.current)
      }
    })

    // ── draw nodes ─────────────────────────────────────────────────────────────
    nodes.forEach((n, i) => {
      const color      = cmap[n.type] ?? '#5b6ef5'
      const isHovered  = i === hIdx
      const isSelected = i === sIdx
      const dim        = dimActive && i !== sIdx && !neighborSet.has(n.id)

      // outer glow
      if (isSelected || isHovered) {
        const g = ctx.createRadialGradient(n.x, n.y, NODE_R, n.x, n.y, NODE_R + 14)
        g.addColorStop(0, color + '55')
        g.addColorStop(1, 'transparent')
        ctx.beginPath(); ctx.arc(n.x, n.y, NODE_R + 14, 0, Math.PI * 2)
        ctx.fillStyle = g; ctx.fill()
      }

      ctx.globalAlpha = dim ? 0.18 : 1

      // node fill
      ctx.beginPath(); ctx.arc(n.x, n.y, NODE_R, 0, Math.PI * 2)
      ctx.fillStyle = color; ctx.fill()

      // ring
      if (isSelected) {
        ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2; ctx.stroke()
      } else if (isHovered) {
        ctx.strokeStyle = color;   ctx.lineWidth = 1.5; ctx.stroke()
      }

      ctx.globalAlpha = 1

      // label: 悬停/选中/邻居节点/缩放足够大时显示
      const isNeighbor = dimActive && neighborSet.has(n.id)
      const showLabel = isHovered || isSelected || isNeighbor || (!dimActive && scale.current > 0.6)
      if (showLabel && !dim) {
        const lbl = (n.label || n.id)
        const fs  = Math.max(9, Math.min(12, 11 / scale.current))
        ctx.font = `${isSelected ? 600 : 400} ${fs}px DM Sans, system-ui`
        ctx.fillStyle = isSelected ? '#ffffff' : 'rgba(226,232,240,0.82)'
        ctx.textAlign = 'center'; ctx.textBaseline = 'top'
        ctx.fillText(lbl.length > 22 ? lbl.slice(0, 20) + '\u2026' : lbl, n.x, n.y + NODE_R + 4)
      }
    })

    ctx.restore()
  }, [canvasRef])

  const tick = useCallback(() => {
    const nodes = simNodes.current, edges = simEdges.current
    // 重力锚点固定为世界坐标原点附近，不随平移变化
    const cx0 = gravityCenterRef.current.x
    const cy0 = gravityCenterRef.current.y

    if (nodes.length > 0) {
      // repulsion
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j]
          let dx = b.x - a.x, dy = b.y - a.y
          const d2 = dx * dx + dy * dy + 0.01
          const d  = Math.sqrt(d2)
          const f  = REPULSION / d2
          dx /= d; dy /= d
          a.vx -= f * dx; a.vy -= f * dy
          b.vx += f * dx; b.vy += f * dy
        }
      }
      // springs
      const nodeMap = new Map(nodes.map(n => [n.id, n]))
      edges.forEach(e => {
        const s = nodeMap.get(e.source), t = nodeMap.get(e.target)
        if (!s || !t) return
        const dx = t.x - s.x, dy = t.y - s.y
        const d  = Math.sqrt(dx * dx + dy * dy) || 1
        const f  = SPRING_K * (d - SPRING_LEN)
        const fx = (dx / d) * f, fy = (dy / d) * f
        s.vx += fx; s.vy += fy; t.vx -= fx; t.vy -= fy
      })
      // gravity + integrate
      nodes.forEach((n, i) => {
        n.vx += (cx0 - n.x) * GRAVITY
        n.vy += (cy0 - n.y) * GRAVITY
        if (dragging.current?.nodeIdx === i) return
        n.vx = Math.max(-MAX_VEL, Math.min(MAX_VEL, n.vx)) * DAMPING
        n.vy = Math.max(-MAX_VEL, Math.min(MAX_VEL, n.vy)) * DAMPING
        n.x += n.vx; n.y += n.vy
      })
    }

    draw()
    rafRef.current = requestAnimationFrame(tick)
  }, [draw, canvasRef])

  const startLoop = useCallback(() => {
    cancelAnimationFrame(rafRef.current)
    rafRef.current = requestAnimationFrame(tick)
  }, [tick])

  const stopLoop = useCallback(() => cancelAnimationFrame(rafRef.current), [])

  const fitView = useCallback(() => {
    const nodes = simNodes.current, canvas = canvasRef.current
    if (!nodes.length || !canvas) return
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
    nodes.forEach(n => { minX = Math.min(minX, n.x); maxX = Math.max(maxX, n.x); minY = Math.min(minY, n.y); maxY = Math.max(maxY, n.y) })
    const pad = 80
    const sx = (canvas.width  - pad * 2) / (maxX - minX || 1)
    const sy = (canvas.height - pad * 2) / (maxY - minY || 1)
    scale.current    = Math.max(0.15, Math.min(2, Math.min(sx, sy)))
    offset.current.x = pad - minX * scale.current
    offset.current.y = pad - minY * scale.current
  }, [canvasRef])

  const loadData = useCallback((nodes: GraphNode[], edges: GraphEdge[]) => {
    const canvas = canvasRef.current
    const W = canvas?.width ?? 800, H = canvas?.height ?? 600
    const cx = W / 2 / scale.current, cy = H / 2 / scale.current
    // 固定重力锚点为本次布局的世界坐标中心，之后平移不影响它
    gravityCenterRef.current = { x: cx, y: cy }
    simNodes.current = nodes.map((n, i) => {
      const angle = (i / nodes.length) * Math.PI * 2
      const r = Math.min(W, H) * 0.25
      return { ...n, x: cx + Math.cos(angle) * r, y: cy + Math.sin(angle) * r, vx: 0, vy: 0 }
    })
    simEdges.current = edges
    selectedNode.current = -1
    selectedEdge.current = -1
    hoveredNode.current  = -1
  }, [canvasRef])

  // mouse helpers
  const onMouseDown = useCallback((e: React.MouseEvent, onSelectNode: (i: number) => void, onSelectEdge: (i: number) => void, onDeselect: () => void) => {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect()
    const cx = e.clientX - r.left, cy = e.clientY - r.top
    const ni = hitNode(cx, cy)
    if (ni >= 0) {
      dragging.current = { nodeIdx: ni }
      selectedNode.current = ni; selectedEdge.current = -1
      onSelectNode(ni)
    } else {
      const ei = hitEdge(cx, cy)
      if (ei >= 0) {
        selectedEdge.current = ei; selectedNode.current = -1
        onSelectEdge(ei)
      } else {
        panning.current = { sx: cx, sy: cy, ox: offset.current.x, oy: offset.current.y }
        selectedNode.current = -1; selectedEdge.current = -1
        onDeselect()
      }
    }
    e.preventDefault()
  }, [hitNode, hitEdge])

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect()
    const cx = e.clientX - r.left, cy = e.clientY - r.top
    if (dragging.current) {
      const nd = simNodes.current[dragging.current.nodeIdx]
      const w = toWorld(cx, cy); nd.x = w.x; nd.y = w.y; nd.vx = 0; nd.vy = 0
    } else if (panning.current) {
      offset.current.x = panning.current.ox + (cx - panning.current.sx)
      offset.current.y = panning.current.oy + (cy - panning.current.sy)
    } else {
      hoveredNode.current = hitNode(cx, cy)
    }
  }, [toWorld, hitNode])

  const onMouseUp = useCallback(() => { dragging.current = null; panning.current = null }, [])

  const onWheel = useCallback((e: React.WheelEvent) => {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect()
    const cx = e.clientX - r.left, cy = e.clientY - r.top
    const factor = e.deltaY < 0 ? 1.12 : 0.88
    const wx = (cx - offset.current.x) / scale.current
    const wy = (cy - offset.current.y) / scale.current
    scale.current    = Math.max(0.12, Math.min(6, scale.current * factor))
    offset.current.x = cx - wx * scale.current
    offset.current.y = cy - wy * scale.current
    e.preventDefault()
  }, [])

  return { simNodes, simEdges, colorMap, selectedNode, selectedEdge, loadData, startLoop, stopLoop, fitView, onMouseDown, onMouseMove, onMouseUp, onWheel, scale, offset }
}

// ── DetailPanel ───────────────────────────────────────────────────────────────
function DetailPanel({ item, onClose }: { item: DetailItem; onClose: () => void }) {
  return (
    <motion.div
      initial={{ x: 40, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 40, opacity: 0 }}
      transition={{ duration: 0.18 }}
      style={{
        position: 'absolute', right: 0, top: 0, bottom: 0,
        width: 268, borderLeft: '1px solid var(--border)',
        overflowY: 'auto', zIndex: 10,
        display: 'flex', flexDirection: 'column',
        background: 'rgba(14,20,34,0.92)',
        backdropFilter: 'blur(12px)',
        boxShadow: 'var(--shadow-lg)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 14px', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>
          {item.kind === 'node' ? '节点详情' : '关系详情'}
        </span>
        <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-dim)', transition: 'color 0.15s' }}
          onMouseEnter={e => (e.currentTarget.style.color = 'var(--text)')}
          onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-dim)')}>
          <X size={14} />
        </button>
      </div>
      <div style={{ padding: '14px', flex: 1, display: 'flex', flexDirection: 'column', gap: 14 }}>
        {item.kind === 'node' ? (
          <>
            <div>
              <div className="section-label">标签</div>
              <p style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--text)' }}>{item.data.label ?? item.data.id}</p>
            </div>
            {item.data.type && (
              <div>
                <div className="section-label">类型</div>
                <span className="badge badge-accent">{item.data.type}</span>
              </div>
            )}
            {item.data.labels?.length > 0 && (
              <div>
                <div className="section-label">标签组</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginTop: 4 }}>
                  {item.data.labels.map(l => <span key={l} className="badge badge-default">{l}</span>)}
                </div>
              </div>
            )}
            {item.data.properties && Object.keys(item.data.properties).length > 0 && (
              <div>
                <div className="section-label">属性</div>
                <PropList obj={item.data.properties as Record<string, unknown>} />
              </div>
            )}
          </>
        ) : (
          <>
            <div>
              <div className="section-label">关系类型</div>
              <p style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--text)' }}>{item.data.label ?? item.data.type ?? '—'}</p>
            </div>
            <div>
              <div className="section-label">起点 → 终点</div>
              <p style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', wordBreak: 'break-all' }}>
                {item.data.source} → {item.data.target}
              </p>
            </div>
            {item.data.properties && Object.keys(item.data.properties).length > 0 && (
              <div>
                <div className="section-label">属性</div>
                <PropList obj={item.data.properties as Record<string, unknown>} />
              </div>
            )}
          </>
        )}
      </div>
    </motion.div>
  )
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function GraphPage() {
  const canvasRef  = useRef<HTMLCanvasElement>(null)
  const wrapperRef = useRef<HTMLDivElement>(null)

  const [stats, setStats]               = useState<GraphStats | null>(null)
  const [allNodeTypes, setAllNodeTypes] = useState<string[]>([])
  const [selectedTypes, setSelectedTypes] = useState<string[]>([])
  const [search, setSearch]             = useState('')
  const [loading, setLoading]           = useState(false)
  const [detail, setDetail]             = useState<DetailItem | null>(null)
  const [nodeCount, setNodeCount]       = useState(0)

  const graph = useForceGraph(canvasRef, wrapperRef)

  const loadGraph = useCallback(async (types: string[], srch: string) => {
    setLoading(true)
    setDetail(null)
    try {
      const data = await getSubgraph({ limit: 300, search: srch || undefined, node_types: types.length ? types : undefined })
      const rawNodes: GraphNode[] = data.nodes ?? []
      const rawEdges: GraphEdge[] = data.edges ?? []
      graph.loadData(rawNodes, rawEdges)
      setNodeCount(rawNodes.length)
    } catch (e) { console.error(e) }
    setLoading(false)
  }, [graph])

  useEffect(() => {
    const resize = () => {
      const canvas = canvasRef.current, wrapper = wrapperRef.current
      if (!canvas || !wrapper) return
      canvas.width  = wrapper.clientWidth
      canvas.height = wrapper.clientHeight
    }
    resize()
    const ro = new ResizeObserver(resize)
    if (wrapperRef.current) ro.observe(wrapperRef.current)
    graph.startLoop()

    getGraphStats().then(s => {
      setStats(s)
      const types = (s.node_labels ?? []).map((l: { label: string }) => l.label)
      setAllNodeTypes(types)
      const map: Record<string, string> = {}
      types.forEach((t: string, i: number) => { map[t] = PALETTE[i % PALETTE.length] })
      graph.colorMap.current = map
      loadGraph([], '')
    }).catch(() => loadGraph([], ''))

    return () => { ro.disconnect(); graph.stopLoop() }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (allNodeTypes.length === 0) return
    loadGraph(selectedTypes, search)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTypes])

  const handleSearch = () => loadGraph(selectedTypes, search)
  const toggleType   = (type: string) => setSelectedTypes(prev => prev.includes(type) ? prev.filter(t => t !== type) : [...prev, type])

  return (
    <div style={{ display: 'flex', height: '100%', position: 'relative' }}>
      <div style={{ width: 220, borderRight: '1px solid var(--border)', flexShrink: 0, display: 'flex', flexDirection: 'column', background: 'var(--panel)', overflow: 'hidden' }}>
        <div style={{ padding: '16px 14px', borderBottom: '1px solid var(--border)' }}>
          <div className="section-label">图谱统计</div>
          {stats ? (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <div className="stat-tile"><div className="stat-tile-value">{stats.node_count.toLocaleString()}</div><div className="stat-tile-label">节点</div></div>
              <div className="stat-tile"><div className="stat-tile-value">{stats.relationship_count.toLocaleString()}</div><div className="stat-tile-label">关系</div></div>
            </div>
          ) : <p style={{ fontSize: 12, color: 'var(--text-dim)' }}>加载中...</p>}
        </div>

        <div style={{ padding: '14px', borderBottom: '1px solid var(--border)' }}>
          <div className="section-label">节点搜索</div>
          <div style={{ position: 'relative' }}>
            <Search size={12} style={{ position: 'absolute', left: 9, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-dim)', pointerEvents: 'none' }} />
            <input value={search} onChange={e => setSearch(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleSearch()}
              placeholder="输入关键词..." className="input" style={{ paddingLeft: 28, fontSize: 12.5 }}
              onFocus={e => { e.currentTarget.style.borderColor = 'var(--border-focus)'; e.currentTarget.style.boxShadow = '0 0 0 3px var(--accent-subtle)' }}
              onBlur={e => { e.currentTarget.style.borderColor = 'var(--border-light)'; e.currentTarget.style.boxShadow = 'none' }}
            />
          </div>
        </div>

        {allNodeTypes.length > 0 && (
          <div style={{ padding: '14px', flex: 1, overflowY: 'auto' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
              <div className="section-label" style={{ marginBottom: 0 }}>节点类型</div>
              {selectedTypes.length > 0 && (
                <button onClick={() => setSelectedTypes([])} style={{ fontSize: 11, color: 'var(--text-dim)', background: 'none', border: 'none', cursor: 'pointer' }}
                  onMouseEnter={e => (e.currentTarget.style.color = 'var(--accent)')} onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-dim)')}>清除</button>
              )}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {allNodeTypes.map((type, i) => {
                const isFiltered = selectedTypes.length > 0 && !selectedTypes.includes(type)
                return (
                  <button key={type} onClick={() => toggleType(type)}
                    style={{ display: 'flex', alignItems: 'center', gap: 8, background: 'none', border: 'none', cursor: 'pointer', padding: '5px 6px', borderRadius: 'var(--r-sm)', transition: 'background 0.12s', color: isFiltered ? 'var(--text-dim)' : 'var(--text)', fontSize: 12.5, textAlign: 'left' }}
                    onMouseEnter={e => (e.currentTarget.style.background = 'var(--surface2)')} onMouseLeave={e => (e.currentTarget.style.background = 'none')}>
                    <span style={{ width: 10, height: 10, borderRadius: '50%', flexShrink: 0, background: PALETTE[i % PALETTE.length], opacity: isFiltered ? 0.2 : 1, boxShadow: !isFiltered ? `0 0 5px ${PALETTE[i % PALETTE.length]}88` : 'none' }} />
                    <span style={{ flex: 1 }}>{type}</span>
                    {selectedTypes.includes(type) && <span style={{ fontSize: 10, color: 'var(--accent)', fontWeight: 600 }}>✓</span>}
                  </button>
                )
              })}
            </div>
          </div>
        )}

        <div style={{ padding: '12px 14px', borderTop: '1px solid var(--border)' }}>
          <button onClick={handleSearch} disabled={loading} className="btn btn-primary" style={{ width: '100%', fontSize: 13 }}>
            <RefreshCw size={13} style={{ animation: loading ? 'spin 0.8s linear infinite' : 'none' }} />
            {loading ? '加载中...' : '刷新图谱'}
          </button>
        </div>
      </div>

      <div ref={wrapperRef} style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        <canvas
          ref={canvasRef}
          style={{ display: 'block', width: '100%', height: '100%', cursor: 'default' }}
          onMouseDown={e => graph.onMouseDown(e,
            (i) => setDetail({ kind: 'node', data: graph.simNodes.current[i] }),
            (i) => setDetail({ kind: 'edge', data: graph.simEdges.current[i] }),
            ()  => setDetail(null),
          )}
          onMouseMove={graph.onMouseMove}
          onMouseUp={graph.onMouseUp}
          onMouseLeave={graph.onMouseUp}
          onWheel={graph.onWheel}
        />

        {!loading && nodeCount === 0 && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12, color: 'var(--text-dim)', pointerEvents: 'none' }}>
            <div style={{ width: 64, height: 64, borderRadius: 'var(--r-xl)', background: 'var(--surface)', border: '1px solid var(--border-light)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Network size={28} style={{ opacity: 0.2 }} />
            </div>
            <p style={{ fontSize: 13.5 }}>暂无图谱数据，请先上传并解析文献</p>
          </div>
        )}

        {loading && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(6,10,16,0.6)', backdropFilter: 'blur(4px)' }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
              <RefreshCw size={24} style={{ color: 'var(--accent)', animation: 'spin 0.8s linear infinite' }} />
              <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>加载图谱...</p>
            </div>
          </div>
        )}

        <div style={{ position: 'absolute', bottom: 16, left: 16, display: 'flex', gap: 6 }}>
          {([
            { icon: <ZoomIn size={14} />, title: '放大',     fn: () => { graph.scale.current = Math.min(6, graph.scale.current * 1.25) } },
            { icon: <ZoomOut size={14} />, title: '缩小',    fn: () => { graph.scale.current = Math.max(0.12, graph.scale.current * 0.8) } },
            { icon: <Maximize2 size={14} />, title: '适应视图', fn: graph.fitView },
          ]).map((btn, i) => (
            <button key={i} onClick={btn.fn} title={btn.title}
              style={{ width: 32, height: 32, borderRadius: 'var(--r-sm)', background: 'rgba(14,20,34,0.88)', backdropFilter: 'blur(8px)', border: '1px solid var(--border-light)', color: 'var(--text-muted)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.15s' }}
              onMouseEnter={e => { e.currentTarget.style.background = 'var(--surface2)'; e.currentTarget.style.color = 'var(--text)' }}
              onMouseLeave={e => { e.currentTarget.style.background = 'rgba(14,20,34,0.88)'; e.currentTarget.style.color = 'var(--text-muted)' }}>
              {btn.icon}
            </button>
          ))}
        </div>

        <div className="hint-pill" style={{ position: 'absolute', bottom: 16, right: detail ? 284 : 16, pointerEvents: 'none', transition: 'right 0.2s' }}>
          拖拽节点 · 滚轮缩放 · 点击查看详情
        </div>

        <AnimatePresence>
          {detail && (
            <DetailPanel item={detail} onClose={() => {
              setDetail(null)
              graph.selectedNode.current = -1
              graph.selectedEdge.current = -1
            }} />
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
