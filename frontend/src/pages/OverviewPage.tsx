import { useQuery } from '@tanstack/react-query'
import { AlertsRail } from '../components/alerts/AlertsRail'
import { TelemetryCharts } from '../components/charts/TelemetryCharts'
import { KpiGrid } from '../components/kpi/KpiGrid'
import { useRealtime } from '../hooks/useRealtime'
import { getJson } from '../lib/api'

export default function OverviewPage() {
  const { latest, connected } = useRealtime()

  const { data: alertsRes } = useQuery({
    queryKey: ['alerts', 'latest'],
    queryFn: () => getJson<{ data: string[] }>('/api/alerts/latest'),
    refetchInterval: 5000,
  })

  const alerts = alertsRes?.data ?? []
  const criticalCount = alerts.filter(a =>
    ['BRAKE_OVERHEAT','BRAKE_PAD_CRITICAL','LOW_SOC','RISK_HIGH','OVERSPEED'].includes(a)
  ).length

  return (
    <div className="space-y-8">

      {/* Page header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-white">Overview</h1>
            {connected && (
              <span className="flex items-center gap-1.5 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-2.5 py-0.5 text-xs font-medium text-emerald-300">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 ping-slow" />
                Live
              </span>
            )}
          </div>
          <p className="mt-1 text-sm text-slate-500">
            BMW i4 · Real-time telemetry, ML alerts, and trend analysis
          </p>
        </div>

        {criticalCount > 0 && (
          <div className="flex items-center gap-2 rounded-xl border border-rose-500/30 bg-rose-950/30 px-4 py-2">
            <span className="text-rose-400">⚠</span>
            <span className="text-sm font-semibold text-rose-300">{criticalCount} critical alert{criticalCount > 1 ? 's' : ''}</span>
          </div>
        )}
      </div>

      {/* KPI cards */}
      <section>
        <p className="mb-3 text-xs font-semibold uppercase tracking-widest text-slate-600">Live Metrics</p>
        <KpiGrid latest={latest} />
      </section>

      {/* Alerts */}
      <section>
        <p className="mb-3 text-xs font-semibold uppercase tracking-widest text-slate-600">
          Active Alerts
          {alerts.length > 0 && (
            <span className="ml-2 rounded-full bg-rose-500/20 px-2 py-0.5 text-rose-400">{alerts.length}</span>
          )}
        </p>
        <AlertsRail alerts={alerts} />
      </section>

      {/* Charts */}
      <section>
        <p className="mb-3 text-xs font-semibold uppercase tracking-widest text-slate-600">Telemetry Trends</p>
        <TelemetryCharts />
      </section>
    </div>
  )
}
