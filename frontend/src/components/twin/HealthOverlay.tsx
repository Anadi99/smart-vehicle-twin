import { getHealthColor, healthHueToHex } from './healthColors'

type Props = { latest: Record<string, unknown> }

function Bar({ label, value, max, color }: { label: string; value: number; max: number; color: string }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100))
  return (
    <div className="flex items-center gap-2">
      <span className="w-20 shrink-0 text-xs text-slate-500">{label}</span>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-800">
        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="w-12 text-right text-xs tabular-nums text-slate-400">{value.toFixed(1)}</span>
    </div>
  )
}

export function HealthOverlay({ latest }: Props) {
  const brake = typeof latest.brake_temp_c === 'number' ? latest.brake_temp_c : 0
  const soc   = typeof latest.battery_soc  === 'number' ? latest.battery_soc  : 100
  const pad   = typeof latest.brake_pad_frac === 'number' ? latest.brake_pad_frac : 1
  const tire  = typeof latest.tire_wear_frac === 'number' ? latest.tire_wear_frac : 0
  const risk  = typeof latest.risk === 'number' ? latest.risk : 0

  const hue   = getHealthColor({ brake_temp_c: brake, battery_soc: soc })
  const color = healthHueToHex(hue)

  const label = hue === 'red' ? 'CRITICAL' : hue === 'amber' ? 'WARNING' : 'NOMINAL'

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4 space-y-3">
      {/* Status badge */}
      <div className="flex items-center gap-3">
        <div className="h-3 w-3 rounded-full animate-pulse" style={{ backgroundColor: color }} />
        <span className="text-sm font-semibold" style={{ color }}>
          Vehicle Health: {label}
        </span>
        {typeof latest.reasons === 'string' && latest.reasons !== 'nominal' && (
          <span className="text-xs text-slate-500 truncate">{latest.reasons as string}</span>
        )}
      </div>

      {/* Health bars */}
      <div className="space-y-2">
        <Bar label="Brake temp"  value={brake}       max={120}  color={brake > 95 ? '#f87171' : brake > 80 ? '#fbbf24' : '#34d399'} />
        <Bar label="Battery SOC" value={soc}         max={100}  color={soc < 15 ? '#f87171' : soc < 25 ? '#fbbf24' : '#34d399'} />
        <Bar label="Brake pad"   value={pad * 100}   max={100}  color={pad < 0.2 ? '#f87171' : pad < 0.3 ? '#fbbf24' : '#34d399'} />
        <Bar label="Tire wear"   value={tire * 100}  max={100}  color={tire > 0.8 ? '#f87171' : tire > 0.6 ? '#fbbf24' : '#34d399'} />
        <Bar label="Risk score"  value={risk * 100}  max={100}  color={risk >= 0.7 ? '#f87171' : risk >= 0.4 ? '#fbbf24' : '#34d399'} />
      </div>
    </div>
  )
}
