import { NavLink, Outlet } from 'react-router-dom'
import { MessageSquare, BookOpen, Share2 } from 'lucide-react'

const navItems = [
  { to: '/chat',    icon: MessageSquare, label: '对话助手',  title: 'Chat'    },
  { to: '/library', icon: BookOpen,      label: '文献库',    title: 'Library' },
  { to: '/graph',   icon: Share2,        label: '知识图谱',  title: 'Graph'   },
]

export default function AppLayout() {
  return (
    <div className="flex h-screen overflow-hidden" style={{ background: 'var(--bg)' }}>
      <nav className="sidebar">
        <div className="sidebar-logo">
          <div className="sidebar-logo-mark">G</div>
          <div>
            <div className="sidebar-logo-text" style={{
              fontFamily: 'var(--font-display)',
              fontWeight: 700,
              fontSize: 15,
              background: 'linear-gradient(135deg, #e8edf5 30%, #a5b4fc)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              letterSpacing: '-0.02em',
            }}>
              GraphAssist
            </div>
            <div className="sidebar-logo-sub">学术知识引擎</div>
          </div>
        </div>

        <div className="sidebar-nav">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
            >
              <Icon size={16} className="nav-item-icon" />
              {label}
            </NavLink>
          ))}
        </div>

        <div className="sidebar-footer">
          <div className="sidebar-version">v0.1.0 · alpha</div>
        </div>
      </nav>

      <main className="flex-1 overflow-hidden">
        <Outlet />
      </main>
    </div>
  )
}
