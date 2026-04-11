import type { ReactNode } from 'react'
import { NavLink } from 'react-router-dom'
import { clearToken } from '../../lib/auth'
import { ConnectionBadge } from '../status/ConnectionBadge'

const links = [
  { to: '/overview',    label: 'Overview',    icon: '⬡',  desc: 'KPIs & alerts' },
  { to: '/telemetry',   label: 'Telemetry',   icon: '📈', desc: 'Live signals' },
  { to: '/predictions', label: 'Predictions', icon: '🤖', desc: 'ML inference' },
  { to: '/twin',        label: '3D Twin',     icon: '🚗', desc: 'Vehicle model' },
  { to: '/laps',        label: 'Laps',        icon: '🏁', desc: 'Lap analytics' },
  { to: '/diagnostics', label: 'Diagnostics', icon: '🔧', desc: 'System health' },
] as const

export function AppShell({ children, onLogout }: { children: ReactNode; onLogout?: () => void }) {
  return (
    <div className="flex min-h-screen bg-[#020617]">

      {/* ── Sidebar ─────────────────────────────────────────────────────── */}
      <aside className="flex w-56 flex-col border-r border-slate-800/60 bg-slate-950/90 backdrop-blur-sm">

        {/* Brand */}
        <div className="border-b border-slate-800/60 px-5 py-5">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-cyan-500/30 bg-cyan-500/10">
              <svg width="18" height="18" viewBox="0 0 40 40" fill="none">
                <rect x="4" y="14" width="32" height="14" rx="3" stroke="#22d3ee" strokeWidth="2"/>
                <rect x="10" y="10" width="20" height="6" rx="2" stroke="#22d3ee" strokeWidth="2"/>
                <circle cx="10" cy="30" r="3.5" stroke="#22d3ee" strokeWidth="2"/>
                <circle cx="30" cy="30" r="3.5" stroke="#22d3ee" strokeWidth="2"/>
              </svg>
            </div>
            <div>
              <div className="text-xs font-bold tracking-tight text-cyan-400">SVT Dashboard</div>
              <div className="text-[10px] text-slate-600">BMW i4 · Digital Twin</div>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex flex-col gap-0.5 p-3 flex-1">
          <p className="mb-2 px-2 text-[10px] font-semibold uppercase tracking-widest text-slate-600">Navigation</p>
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              className={({ isActive }) =>
                `group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-all ${
                  isActive
                    ? 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/20'
                    : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200 border border-transparent'
                }`
              }
            >
              <span className="text-base leading-none">{l.icon}</span>
              <div>
                <div className="font-medium leading-none">{l.label}</div>
                <div className="mt-0.5 text-[10px] text-slate-600 group-hover:text-slate-500">{l.desc}</div>
              </div>
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="border-t border-slate-800/60 p-4 space-y-3">
          <div className="space-y-1">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-600">Stack</p>
            <div className="flex flex-wrap gap-1">
              {['FastAPI', 'React', 'AWS', 'TF'].map(t => (
                <span key={t} className="rounded border border-slate-800 bg-slate-900 px-1.5 py-0.5 text-[10px] text-slate-500">{t}</span>
              ))}
            </div>
          </div>
          <button
            onClick={() => { clearToken(); onLogout?.() }}
            className="flex w-full items-center gap-2 rounded-lg border border-slate-800 px-3 py-2 text-xs text-slate-500 transition-colors hover:border-rose-500/40 hover:text-rose-400"
          >
            <span>⏻</span> Sign out
          </button>
        </div>
      </aside>

      {/* ── Main ────────────────────────────────────────────────────────── */}
      <div className="flex min-w-0 flex-1 flex-col">

        {/* Top bar */}
        <header className="flex items-center justify-between border-b border-slate-800/60 bg-slate-950/60 px-6 py-3 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <div className="h-1.5 w-1.5 rounded-full bg-cyan-400 ping-slow" />
            <span className="text-xs text-slate-500">Cloud AI Developer Portfolio</span>
          </div>
          <div className="flex items-center gap-4">
            <span className="hidden text-xs text-slate-700 sm:block">
              {new Date().toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' })}
            </span>
            <ConnectionBadge />
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-auto p-6 lg:p-8">{children}</main>
      </div>
    </div>
  )
}
