import { NavLink, Outlet } from 'react-router-dom'

const navItems = [
  { to: '/chat', label: '💬 对话', title: 'Chat' },
  { to: '/library', label: '📚 文献库', title: 'Library' },
  { to: '/graph', label: '🕸 知识图谱', title: 'Graph' },
]

export default function AppLayout() {
  return (
    <div className="flex h-screen bg-[#0f1117] text-slate-200 overflow-hidden">
      <nav className="w-14 flex flex-col items-center py-4 gap-3 bg-[#1a1d27] border-r border-slate-700/50 shrink-0">
        <div className="w-8 h-8 rounded-lg bg-violet-600 flex items-center justify-center text-sm font-bold mb-4">
          G
        </div>
        {navItems.map(({ to, label, title }) => (
          <NavLink
            key={to}
            to={to}
            title={title}
            className={({ isActive }) =>
              `w-10 h-10 flex items-center justify-center rounded-lg text-lg transition-colors ${
                isActive
                  ? 'bg-violet-600/30 text-violet-400'
                  : 'text-slate-500 hover:text-slate-200 hover:bg-slate-700/50'
              }`
            }
          >
            {label.split(' ')[0]}
          </NavLink>
        ))}
      </nav>
      <main className="flex-1 overflow-hidden">
        <Outlet />
      </main>
    </div>
  )
}
