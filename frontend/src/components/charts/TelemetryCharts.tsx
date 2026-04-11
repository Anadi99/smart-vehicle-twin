import { useQuery } from '@tanstack/react-query'
import {
  CartesianGrid, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { getJson } from '../../lib/api'

type Row = Record<string, number>

function toNum(v: unknown, fallback = 0): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : Number(v ?? fallback)
}

const TOOLTIP_STYLE = {
  contentStyle: {
    background: '#0f172a',
    border: '1px solid #1e293b',
    borderRadius: '8px',
    fontSize: '12px',
    boxShadow: '0 8px 24px rgba(0,0,0,.5)',
  },
  labelStyle: { color: '#94a3b8' },
  itemStyle: { color: '#e2e8f0' },
}

const AXIS_TICK = { fill: '#475569', fontSize: 10 }
const GRID_STROKE = '#1e293b'

type ChartConfig = {
  title: string
  lines: { key: string; name: string; color: string; yAxis?: 'left' | 'right' }[]
  rightDomain?: [number, number]
}

const CHARTS: ChartConfig[] = [
  {
    title: 'Speed & Brake Temperature',
    lines: [
      { key: 'speed_kph',    name: 'Speed (km/h)',    color: '#22d3ee', yAxis: 'left' },
      { key: 'brake_temp_c', name: 'Brake temp (°C)', color: '#f472b6', yAxis: 'right' },
    ],
  },
  {
    title: 'Battery SOC & Brake Pad',
    lines: [
      { key: 'battery_soc',    name: 'Battery SOC (%)',  color: '#34d399', yAxis: 'left' },
      { key: 'brake_pad_frac', name: 'Brake pad (frac)', color: '#fb923c', yAxis: 'right' },
    ],
    rightDomain: [0, 1],
  },
  {
    title: 'Risk Score & Tire Wear',
    lines: [
      { key: 'risk',           name: 'Risk score',  color: '#f87171' },
      { key: 'tire_wear_frac', name: 'Tire wear',   color: '#a78bfa' },
    ],
    rightDomain: [0, 1],
  },
]

function Chart({ config, rows }: { config: ChartConfig; rows: Row[] }) {
  const hasRight = config.lines.some(l => l.yAxis === 'right')
  return (
    <div className="rounded-xl border border-slate-800/60 bg-slate-900/40 p-4">
      <p className="mb-3 text-xs font-semibold uppercase tracking-widest text-slate-500">{config.title}</p>
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
            <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="i" tick={AXIS_TICK} axisLine={false} tickLine={false} />
            <YAxis yAxisId="left"  tick={AXIS_TICK} axisLine={false} tickLine={false} />
            {hasRight && (
              <YAxis yAxisId="right" orientation="right" tick={AXIS_TICK} axisLine={false} tickLine={false}
                domain={config.rightDomain} />
            )}
            <Tooltip {...TOOLTIP_STYLE} />
            <Legend wrapperStyle={{ fontSize: '11px', color: '#64748b' }} />
            {config.lines.map(l => (
              <Line
                key={l.key}
                yAxisId={l.yAxis ?? 'left'}
                type="monotone"
                dataKey={l.key}
                name={l.name}
                stroke={l.color}
                dot={false}
                strokeWidth={2}
                activeDot={{ r: 4, strokeWidth: 0 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export function TelemetryCharts() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['telemetry', 'history'],
    queryFn: () => getJson<{ data: Record<string, unknown>[] }>('/api/telemetry/history?limit=120'),
    refetchInterval: 5_000,
  })

  if (isLoading) return (
    <div className="flex h-52 items-center justify-center rounded-xl border border-slate-800 bg-slate-900/40">
      <div className="flex items-center gap-2 text-sm text-slate-500">
        <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
        </svg>
        Loading telemetry…
      </div>
    </div>
  )

  if (error) return (
    <div className="flex h-52 items-center justify-center rounded-xl border border-rose-500/20 bg-rose-950/10">
      <p className="text-sm text-rose-400">Could not load telemetry history</p>
    </div>
  )

  const rows: Row[] = (data?.data ?? []).map((r, i) => ({
    i,
    speed_kph:      toNum(r.speed_kph),
    brake_temp_c:   toNum(r.brake_temp_c),
    battery_soc:    toNum(r.battery_soc),
    brake_pad_frac: toNum(r.brake_pad_frac),
    tire_wear_frac: toNum(r.tire_wear_frac),
    risk:           toNum(r.risk),
  }))

  return (
    <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-3">
      {CHARTS.map(cfg => <Chart key={cfg.title} config={cfg} rows={rows} />)}
    </div>
  )
}
