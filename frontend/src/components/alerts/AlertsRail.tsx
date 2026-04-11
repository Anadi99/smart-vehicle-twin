export type AlertItem = { code: string; severity: 'info' | 'warning' | 'critical' }

const SEVERITY_MAP: Record<string, AlertItem['severity']> = {
  BRAKE_OVERHEAT:    'critical',
  BRAKE_PAD_CRITICAL:'critical',
  LOW_SOC:           'critical',
  RISK_HIGH:         'critical',
  OVERSPEED:         'critical',
  BRAKE_TEMP_HIGH:   'warning',
  BRAKE_PAD_LOW:     'warning',
  SOC_WARNING:       'warning',
  RISK_MEDIUM:       'warning',
}

const ICONS: Record<string, string> = {
  BRAKE_OVERHEAT:    '🔥',
  BRAKE_PAD_CRITICAL:'🛑',
  BRAKE_PAD_LOW:     '🟡',
  LOW_SOC:           '🔋',
  SOC_WARNING:       '🟡',
  RISK_HIGH:         '☠',
  RISK_MEDIUM:       '⚠',
  OVERSPEED:         '💨',
  BRAKE_TEMP_HIGH:   '🌡',
}

const LABELS: Record<string, string> = {
  BRAKE_OVERHEAT:    'Brake overheat',
  BRAKE_PAD_CRITICAL:'Brake pad critical',
  BRAKE_PAD_LOW:     'Brake pad low',
  LOW_SOC:           'Battery critically low',
  SOC_WARNING:       'Battery warning',
  RISK_HIGH:         'Risk HIGH',
  RISK_MEDIUM:       'Risk MEDIUM',
  OVERSPEED:         'Overspeed detected',
  BRAKE_TEMP_HIGH:   'Brake temp high',
}

function normalize(raw: string[] | AlertItem[]): AlertItem[] {
  return raw.map(a =>
    typeof a === 'string'
      ? { code: a, severity: SEVERITY_MAP[a] ?? 'warning' }
      : a
  )
}

export function AlertsRail({ alerts }: { alerts: string[] | AlertItem[] }) {
  const items = normalize(alerts)

  if (items.length === 0) {
    return (
      <div className="flex items-center gap-3 rounded-xl border border-emerald-500/20 bg-emerald-500/5 px-4 py-3">
        <span className="text-emerald-400">✓</span>
        <span className="text-sm text-emerald-300">All systems nominal</span>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-2">
      {items.map((a) => {
        const isCritical = a.severity === 'critical'
        return (
          <div
            key={a.code}
            className={`flex items-center gap-3 rounded-xl border px-4 py-3 transition-all ${
              isCritical
                ? 'border-rose-500/30 bg-rose-950/30 glow-rose'
                : 'border-amber-500/25 bg-amber-950/20'
            }`}
          >
            {isCritical && (
              <div className="relative flex h-2 w-2 shrink-0">
                <span className="ping-slow absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-rose-500" />
              </div>
            )}
            <span className="text-base">{ICONS[a.code] ?? '⚠'}</span>
            <div className="flex-1 min-w-0">
              <span className="text-sm font-medium text-slate-100">{LABELS[a.code] ?? a.code}</span>
              <span className="ml-2 font-mono text-xs text-slate-500">{a.code}</span>
            </div>
            <span className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-semibold uppercase ${
              isCritical ? 'bg-rose-500/20 text-rose-300' : 'bg-amber-500/15 text-amber-300'
            }`}>
              {a.severity}
            </span>
          </div>
        )
      })}
    </div>
  )
}
