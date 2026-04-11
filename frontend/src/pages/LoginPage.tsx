import { useState } from 'react'
import { login } from '../lib/auth'

export default function LoginPage({ onLogin }: { onLogin: () => void }) {
  const [user, setUser] = useState('admin')
  const [pass, setPass] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      await login(user, pass)
      onLogin()
    } catch {
      setError('Invalid credentials')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-[#020617]">

      {/* Animated background grid */}
      <div className="pointer-events-none absolute inset-0"
        style={{
          backgroundImage: `linear-gradient(rgba(34,211,238,.04) 1px, transparent 1px),
                            linear-gradient(90deg, rgba(34,211,238,.04) 1px, transparent 1px)`,
          backgroundSize: '48px 48px',
        }}
      />

      {/* Radial glow */}
      <div className="pointer-events-none absolute left-1/2 top-1/2 h-[600px] w-[600px] -translate-x-1/2 -translate-y-1/2 rounded-full"
        style={{ background: 'radial-gradient(circle, rgba(34,211,238,.06) 0%, transparent 70%)' }}
      />

      {/* Corner accent lines */}
      <div className="pointer-events-none absolute left-8 top-8 h-16 w-16 border-l-2 border-t-2 border-cyan-500/30" />
      <div className="pointer-events-none absolute right-8 top-8 h-16 w-16 border-r-2 border-t-2 border-cyan-500/30" />
      <div className="pointer-events-none absolute bottom-8 left-8 h-16 w-16 border-b-2 border-l-2 border-cyan-500/30" />
      <div className="pointer-events-none absolute bottom-8 right-8 h-16 w-16 border-b-2 border-r-2 border-cyan-500/30" />

      <div className="relative z-10 w-full max-w-md px-6">

        {/* Logo / brand */}
        <div className="mb-10 text-center">
          <div className="mb-3 inline-flex items-center justify-center rounded-2xl border border-cyan-500/20 bg-cyan-500/5 p-4">
            <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
              <rect x="4" y="14" width="32" height="14" rx="3" stroke="#22d3ee" strokeWidth="1.5"/>
              <rect x="10" y="10" width="20" height="6" rx="2" stroke="#22d3ee" strokeWidth="1.5"/>
              <circle cx="10" cy="30" r="4" stroke="#22d3ee" strokeWidth="1.5"/>
              <circle cx="30" cy="30" r="4" stroke="#22d3ee" strokeWidth="1.5"/>
              <line x1="20" y1="14" x2="20" y2="28" stroke="#22d3ee" strokeWidth="1" strokeDasharray="2 2"/>
            </svg>
          </div>
          <h1 className="gradient-text text-3xl font-bold tracking-tight">Smart Vehicle Twin</h1>
          <p className="mt-2 text-sm text-slate-500">BMW i4 · Real-time Digital Twin · Cloud AI</p>
        </div>

        {/* Card */}
        <div className="glow-cyan rounded-2xl border border-slate-800 bg-slate-900/80 p-8 backdrop-blur-sm">
          <h2 className="mb-6 text-lg font-semibold text-slate-100">Sign in to dashboard</h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="mb-1.5 block text-xs font-medium text-slate-400">Username</label>
              <input
                type="text"
                value={user}
                onChange={e => setUser(e.target.value)}
                className="w-full rounded-lg border border-slate-700 bg-slate-800/80 px-4 py-2.5 text-sm text-slate-100 placeholder-slate-600 transition-colors focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500/30"
                required
              />
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-medium text-slate-400">Password</label>
              <input
                type="password"
                value={pass}
                onChange={e => setPass(e.target.value)}
                placeholder="••••••••"
                className="w-full rounded-lg border border-slate-700 bg-slate-800/80 px-4 py-2.5 text-sm text-slate-100 placeholder-slate-600 transition-colors focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500/30"
                required
              />
            </div>

            {error && (
              <div className="flex items-center gap-2 rounded-lg border border-rose-500/30 bg-rose-950/30 px-3 py-2">
                <span className="text-rose-400">⚠</span>
                <p className="text-xs text-rose-300">{error} — try admin / svtwin2026</p>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="relative w-full overflow-hidden rounded-lg bg-cyan-600 py-2.5 text-sm font-semibold text-white transition-all hover:bg-cyan-500 disabled:opacity-50"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
                  </svg>
                  Authenticating…
                </span>
              ) : 'Sign in'}
            </button>
          </form>

          <div className="mt-6 rounded-lg border border-slate-800 bg-slate-950/50 px-4 py-3">
            <p className="text-xs text-slate-600">Demo credentials</p>
            <p className="mt-0.5 font-mono text-xs text-slate-400">admin / svtwin2026</p>
          </div>
        </div>

        {/* Tech stack badges */}
        <div className="mt-6 flex flex-wrap justify-center gap-2">
          {['FastAPI', 'React 19', 'AWS ECS', 'Terraform', 'scikit-learn'].map(t => (
            <span key={t} className="rounded-full border border-slate-800 bg-slate-900/50 px-3 py-1 text-xs text-slate-500">
              {t}
            </span>
          ))}
        </div>

        {/* Copyright */}
        <p className="mt-6 text-center text-xs text-slate-700">
          © {new Date().getFullYear()} Developed by{' '}
          <a
            href="https://github.com/Anadi99"
            target="_blank"
            rel="noopener noreferrer"
            className="text-cyan-600 transition-colors hover:text-cyan-500"
          >
            Anadi99
          </a>
          {' '}· Cloud AI Developer
        </p>
      </div>
    </div>
  )
}
