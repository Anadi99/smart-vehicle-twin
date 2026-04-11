type Props = { latest: Record<string, unknown> }

function num(v: unknown, digits = 1): string {
  if (typeof v === 'number' && Number.isFinite(v)) return v.toFixed(digits)
  return '—'
}

type CardProps = {
  label: string
  value: string
  icon: string
  sub?: string
  accent?: 'cyan' | 'rose' | 'amber' | 'green' | 'violet' | 'slate'
  alert?: boolean
}

const ACCENT_MAP = {
  cyan:   { border: 'border-cyan-500/20',   bg: 'bg-cyan-500/5',   text: 'text-cyan-300',   icon: 'text-cyan-400' },
  rose:   { border: 'border-rose-500/30',   bg: 'bg-rose-500/5',   text: 'text-rose-300',   icon: 'text-rose-400' },
  amber:  { border: 'border-amber-500/25',  bg: 'bg-amber-500/5',  text: 'text-amber-300',  icon: 'text-amber-400' },
  green:  { border: 'border-emerald-500/20',bg: 'bg-emerald-500/5',text: 'text-emerald-300',icon: 'text-emerald-400' },
  violet: { border: 'border-violet-500/20', bg: 'bg-violet-500/5', text: 'text-violet-300', icon: 'text-violet-400' },
  slate:  { border: 'border-slate-700/50',  bg: 'bg-slate-800/30', text: 'text-slate-200',  icon: 'text-slate-400' },
}

function KpiCard({ label, value, icon, sub, accent = 'slate', alert }: CardProps) {
  const c = ACCENT_MAP[accent]
  return (
    <div className={`card-hover scanlines relative overflow-hidden rounded-xl border ${c.border} ${c.bg} p-4`}>
      {alert && (
        <div className="absolute right-3 top-3 flex h-2 w-2">
          <span className="ping-slow absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-rose-500" />
        </div>
      )}
      <div className="flex items-start justify-between">
        <div className="text-xs font-medium uppercase tracking-wider text-slate-500">{label}</div>
        <span className={`text-lg ${c.icon}`}>{icon}</span>
      </div>
      <div className={`ticker mt-2 text-2xl font-bold tabular-nums ${c.text}`}>{value}</div>
      {sub && <div className="mt-1 truncate text-xs text-slate-600" title={sub}>{sub}</div>}
    </div>
  )
}

function riskAccent(v: unknown): 'rose' | 'amber' | 'green' {
  const r = typeof v === 'number' ? v : -1
  if (r >= 0.7) return 'rose'
  if (r >= 0.4) return 'amber'
  return 'green'
}

function tempAccent(v: unknown): 'rose' | 'amber' | 'cyan' {
  const t = typeof v === 'number' ? v : 0
  if (t > 95) return 'rose'
  if (t > 80) return 'amber'
  return 'cyan'
}

function socAccent(v: unknown): 'rose' | 'amber' | 'green' {
  const s = typeof v === 'number' ? v : 100
  if (s < 15) return 'rose'
  if (s < 25) return 'amber'
  return 'green'
}

export function KpiGrid({ latest }: Props) {
  const risk = typeof latest.risk === 'number' ? latest.risk : -1
  const soc  = typeof latest.battery_soc === 'number' ? latest.battery_soc : 100
  const temp = typeof latest.brake_temp_c === 'number' ? latest.brake_temp_c : 0
  const pad  = typeof latest.brake_pad_frac === 'number' ? latest.brake_pad_frac : 1

  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <KpiCard label="Speed"       value={`${num(latest.speed_kph)} km/h`}  icon="⚡" accent="cyan" />
      <KpiCard label="Brake Temp"  value={`${num(latest.brake_temp_c)}°C`}  icon="🌡" accent={tempAccent(temp)} alert={temp > 95} />
      <KpiCard label="Battery SOC" value={`${num(latest.battery_soc, 1)}%`} icon="🔋" accent={socAccent(soc)}  alert={soc < 15} />
      <KpiCard label="Brake Pad"   value={num(latest.brake_pad_frac, 3)}    icon="🛑" accent={pad < 0.2 ? 'rose' : pad < 0.3 ? 'amber' : 'green'} alert={pad < 0.2} />
      <KpiCard label="Tire Wear"   value={num(latest.tire_wear_frac, 3)}    icon="🔄" accent="violet" />
      <KpiCard label="Accel"       value={`${num(latest.accel_mps2, 2)} m/s²`} icon="📐" accent="slate" />
      <KpiCard
        label="Risk Score"
        value={num(latest.risk, 2)}
        icon="⚠"
        accent={riskAccent(risk)}
        alert={risk >= 0.7}
        sub={typeof latest.reasons === 'string' && latest.reasons !== 'nominal' ? latest.reasons as string : undefined}
      />
      <KpiCard
        label="GPS"
        value={typeof latest.lat === 'number' ? `${num(latest.lat, 4)}` : '—'}
        icon="📍"
        accent="slate"
        sub={typeof latest.lon === 'number' ? `lon ${num(latest.lon, 4)}` : undefined}
      />
    </div>
  )
}
