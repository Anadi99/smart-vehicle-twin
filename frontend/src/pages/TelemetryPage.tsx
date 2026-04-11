import { TelemetryCharts } from '../components/charts/TelemetryCharts'
import { KpiGrid } from '../components/kpi/KpiGrid'
import { useRealtime } from '../hooks/useRealtime'

export default function TelemetryPage() {
  const { latest, connected } = useRealtime()

  return (
    <div className="space-y-8">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Telemetry</h1>
          <p className="mt-1 text-sm text-slate-500">
            Live vehicle signals streamed via WebSocket at 1 Hz
          </p>
        </div>
        <div className={`flex items-center gap-2 rounded-xl border px-3 py-1.5 text-xs font-medium ${
          connected
            ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300'
            : 'border-slate-700 bg-slate-800/50 text-slate-500'
        }`}>
          <span className={`h-1.5 w-1.5 rounded-full ${connected ? 'bg-emerald-400 ping-slow' : 'bg-slate-600'}`} />
          {connected ? 'Streaming' : 'Disconnected'}
        </div>
      </div>

      <section>
        <p className="mb-3 text-xs font-semibold uppercase tracking-widest text-slate-600">Current Readings</p>
        <KpiGrid latest={latest} />
      </section>

      <section>
        <p className="mb-3 text-xs font-semibold uppercase tracking-widest text-slate-600">Signal History (last 120 samples)</p>
        <TelemetryCharts />
      </section>
    </div>
  )
}
