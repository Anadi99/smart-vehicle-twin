import { useRealtime } from '../../hooks/useRealtime'

const CONFIG = {
  connecting:   { label: 'Connecting',   dot: 'bg-amber-400',   cls: 'border-amber-500/20 bg-amber-500/10 text-amber-300' },
  open:         { label: 'Live',         dot: 'bg-emerald-400 ping-slow', cls: 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300' },
  reconnecting: { label: 'Reconnecting', dot: 'bg-amber-400 ping-slow',  cls: 'border-amber-500/20 bg-amber-500/10 text-amber-200' },
  closed:       { label: 'Offline',      dot: 'bg-slate-500',   cls: 'border-slate-700 bg-slate-800/50 text-slate-400' },
  error:        { label: 'Error',        dot: 'bg-rose-400',    cls: 'border-rose-500/20 bg-rose-500/10 text-rose-300' },
} as const

export function ConnectionBadge() {
  const { wsStatus } = useRealtime()
  const cfg = CONFIG[wsStatus as keyof typeof CONFIG] ?? CONFIG.closed
  return (
    <span className={`flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium transition-all ${cfg.cls}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${cfg.dot}`} />
      {cfg.label}
    </span>
  )
}
