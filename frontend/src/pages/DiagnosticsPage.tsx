import { useQuery } from '@tanstack/react-query'
import { getJson } from '../lib/api'
import { useRealtime } from '../hooks/useRealtime'

function JsonCard({ title, data, loading }: { title: string; data: unknown; loading: boolean }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
      <h2 className="text-xs font-medium uppercase tracking-wide text-slate-400">{title}</h2>
      <pre className="mt-2 overflow-x-auto text-xs text-cyan-200">
        {loading ? '…' : JSON.stringify(data, null, 2)}
      </pre>
    </div>
  )
}

export default function DiagnosticsPage() {
  const { wsStatus, latest } = useRealtime()

  const health = useQuery({
    queryKey: ['health'],
    queryFn: () => getJson<{ status: string }>('/api/health'),
    refetchInterval: 5000,
  })

  const status = useQuery({
    queryKey: ['status'],
    queryFn: () => getJson<Record<string, unknown>>('/api/status'),
    refetchInterval: 10_000,
  })

  const models = useQuery({
    queryKey: ['models', 'status'],
    queryFn: () => getJson<Record<string, unknown>>('/api/models/status'),
  })

  const wsBadge =
    wsStatus === 'open'
      ? 'bg-emerald-500/20 text-emerald-300'
      : wsStatus === 'connecting'
        ? 'bg-amber-500/15 text-amber-300'
        : 'bg-rose-500/15 text-rose-300'

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-white">Diagnostics</h1>
        <p className="mt-1 text-sm text-slate-400">System health, ingestion mode, model registry, live WS state.</p>
      </div>

      {/* Live WS snapshot */}
      <div className="flex flex-wrap items-center gap-3">
        <span className={`rounded-full px-3 py-1 text-xs font-medium ${wsBadge}`}>
          WebSocket: {wsStatus}
        </span>
        {typeof latest.speed_kph === 'number' && (
          <span className="rounded-full bg-slate-800 px-3 py-1 text-xs text-slate-300">
            Last tick — speed: {(latest.speed_kph as number).toFixed(1)} km/h &nbsp;|&nbsp;
            SOC: {(latest.battery_soc as number)?.toFixed(1)}% &nbsp;|&nbsp;
            risk: {(latest.risk as number)?.toFixed(2)}
          </span>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <JsonCard title="GET /api/health"        data={health.data}  loading={health.isLoading} />
        <JsonCard title="GET /api/status"        data={status.data}  loading={status.isLoading} />
        <JsonCard title="GET /api/models/status" data={models.data}  loading={models.isLoading} />
        <JsonCard title="Latest telemetry (WS)"  data={latest}       loading={false} />
      </div>
    </div>
  )
}
