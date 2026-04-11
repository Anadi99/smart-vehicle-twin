import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { getJson } from '../lib/api'
import { authFetch } from '../lib/auth'

type Preds = {
  rul_laps: number | null
  soc_drop_per_lap: number | null
  brake_wear_per_lap: number | null
  laps_until_pad: number | null
  laps_until_soc: number | null
  recommended_action: string
  rul_inference?: string
}
type PredBody = { predictions: Preds; status: string; models: { loaded: string[]; onnx_available: boolean } }
type ExplainBody = { method: string; base_value: number; contribs: { feature: string; value: number; contribution: number }[] }
type AnomalyBody = { is_anomaly: boolean; score: number; status: string; samples_collected?: number }
type DriftBody = { status: string; max_psi: number; samples: number; features: Record<string, { psi: number; status: string }> }

function StatCard({ label, value, unit, warn }: { label: string; value: number | null; unit?: string; warn?: boolean }) {
  return (
    <div className={`rounded-lg border p-4 ${warn ? 'border-amber-500/40 bg-amber-950/20' : 'border-slate-800 bg-slate-900/50'}`}>
      <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`mt-1 text-2xl font-bold tabular-nums ${warn ? 'text-amber-300' : 'text-slate-100'}`}>
        {value !== null && value !== undefined ? value : '—'}
        {value !== null && unit && <span className="ml-1 text-sm font-normal text-slate-400">{unit}</span>}
      </div>
    </div>
  )
}

async function runInference() {
  const res = await authFetch('/api/predictions/run', { method: 'POST' })
  if (!res.ok) throw new Error(`${res.status}`)
  return res.json() as Promise<PredBody>
}

async function fetchExplain() {
  const res = await authFetch('/api/predictions/explain')
  if (!res.ok) throw new Error(`${res.status}`)
  return res.json() as Promise<ExplainBody>
}

export default function PredictionsPage() {
  const qc = useQueryClient()

  const { data: latestRes, isFetching } = useQuery({
    queryKey: ['predictions', 'latest'],
    queryFn: () => getJson<{ data: PredBody | null }>('/api/predictions/latest'),
  })

  const anomaly = useQuery({
    queryKey: ['anomaly'],
    queryFn: () => getJson<AnomalyBody>('/api/anomaly/latest'),
    refetchInterval: 3_000,
  })

  const drift = useQuery({
    queryKey: ['drift'],
    queryFn: () => getJson<DriftBody>('/api/drift/status'),
    refetchInterval: 10_000,
  })

  const explain = useQuery({
    queryKey: ['explain'],
    queryFn: fetchExplain,
    enabled: false,
  })

  const run = useMutation({
    mutationFn: runInference,
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['predictions', 'latest'] })
      void qc.refetchQueries({ queryKey: ['explain'] })
    },
  })

  const payload = run.data ?? latestRes?.data
  const preds = payload?.predictions
  const ad = anomaly.data
  const dr = drift.data

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold text-white">Predictions</h1>
        <p className="mt-1 text-sm text-slate-400">ML inference · SHAP explainability · Anomaly detection · Data drift</p>
      </div>

      {/* Run button */}
      <div className="flex items-center gap-4 flex-wrap">
        <button
          type="button"
          className="rounded-md bg-cyan-600 px-4 py-2 text-sm font-medium text-white hover:bg-cyan-500 disabled:opacity-50"
          disabled={run.isPending}
          onClick={() => run.mutate()}
        >
          {run.isPending ? 'Running…' : 'Run inference'}
        </button>
        {payload && (
          <span className={`text-xs font-medium px-2 py-1 rounded-full ${payload.status === 'ok' ? 'bg-emerald-500/20 text-emerald-300' : 'bg-amber-500/15 text-amber-300'}`}>
            {payload.status === 'ok'
              ? `Models: ${payload.models.loaded.join(', ')}${payload.models.onnx_available ? ' · ONNX ✓' : ''}`
              : 'Degraded — no models loaded'}
          </span>
        )}
      </div>

      {/* Prediction cards */}
      {isFetching && !preds && <p className="text-sm text-slate-500">Loading…</p>}
      {!isFetching && !preds && <p className="text-sm text-slate-500">No prediction run yet.</p>}
      {preds && (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <StatCard label="Remaining Useful Life" value={preds.rul_laps} unit="laps" warn={(preds.rul_laps ?? 99) <= 5} />
            <StatCard label="SOC drop / lap"        value={preds.soc_drop_per_lap} unit="%" />
            <StatCard label="Brake wear / lap"      value={preds.brake_wear_per_lap} />
            <StatCard label="Laps until pad limit"  value={preds.laps_until_pad} unit="laps" warn={(preds.laps_until_pad ?? 99) <= 3} />
            <StatCard label="Laps until SOC limit"  value={preds.laps_until_soc} unit="laps" warn={(preds.laps_until_soc ?? 99) <= 3} />
          </div>
          <div className="rounded-lg border border-cyan-500/30 bg-cyan-950/20 px-4 py-3">
            <span className="text-xs uppercase tracking-wide text-cyan-500">Recommended action</span>
            <p className="mt-1 text-base font-semibold text-cyan-200">{preds.recommended_action}</p>
            {preds.rul_inference && <p className="mt-0.5 text-xs text-slate-600">Inference engine: {preds.rul_inference}</p>}
          </div>
        </>
      )}

      {/* Anomaly Detection */}
      <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
        <h2 className="text-sm font-medium uppercase tracking-wide text-slate-400 mb-3">Anomaly Detection (Isolation Forest)</h2>
        {ad ? (
          <div className="flex items-center gap-4 flex-wrap">
            <span className={`rounded-full px-3 py-1 text-sm font-semibold ${ad.is_anomaly ? 'bg-rose-500/20 text-rose-300' : ad.status === 'warming_up' ? 'bg-amber-500/15 text-amber-300' : 'bg-emerald-500/20 text-emerald-300'}`}>
              {ad.status === 'warming_up' ? `Warming up (${ad.samples_collected}/${200} samples)` : ad.is_anomaly ? '⚠ ANOMALY DETECTED' : '✓ Normal'}
            </span>
            <span className="text-xs text-slate-500">Score: {ad.score.toFixed(4)}</span>
          </div>
        ) : <p className="text-xs text-slate-500">Loading…</p>}
      </div>

      {/* Data Drift */}
      <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
        <h2 className="text-sm font-medium uppercase tracking-wide text-slate-400 mb-3">Data Drift (PSI vs Training Baseline)</h2>
        {dr ? (
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${dr.status === 'alert' ? 'bg-rose-500/20 text-rose-300' : dr.status === 'warning' ? 'bg-amber-500/15 text-amber-300' : 'bg-emerald-500/20 text-emerald-300'}`}>
                {dr.status.toUpperCase()} — max PSI: {dr.max_psi}
              </span>
              <span className="text-xs text-slate-600">{dr.samples} samples</span>
            </div>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              {Object.entries(dr.features ?? {}).map(([feat, info]) => (
                <div key={feat} className="rounded border border-slate-800 bg-slate-950/50 px-3 py-2">
                  <div className="text-xs text-slate-500">{feat}</div>
                  <div className={`text-sm font-semibold ${info.status === 'alert' ? 'text-rose-400' : info.status === 'warning' ? 'text-amber-400' : 'text-emerald-400'}`}>
                    PSI {info.psi}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : <p className="text-xs text-slate-500">Collecting samples…</p>}
      </div>

      {/* SHAP Explainability */}
      <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-medium uppercase tracking-wide text-slate-400">SHAP Explainability (RUL model)</h2>
          <button
            onClick={() => void qc.refetchQueries({ queryKey: ['explain'] })}
            className="text-xs text-cyan-500 hover:text-cyan-400"
          >
            Refresh
          </button>
        </div>
        {explain.data ? (
          <div className="space-y-2">
            <p className="text-xs text-slate-500">Method: {explain.data.method} · Base value: {explain.data.base_value}</p>
            {explain.data.contribs.length > 0 ? (
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={explain.data.contribs} layout="vertical">
                    <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                    <YAxis type="category" dataKey="feature" tick={{ fill: '#94a3b8', fontSize: 10 }} width={110} />
                    <Tooltip
                      contentStyle={{ background: '#0f172a', border: '1px solid #334155' }}
                    formatter={(v: unknown) => [typeof v === 'number' ? v.toFixed(6) : String(v ?? ''), 'contribution'] as [string, string]}
                    />
                    <Bar dataKey="contribution">
                      {explain.data.contribs.map((entry, i) => (
                        <Cell key={i} fill={entry.contribution >= 0 ? '#22d3ee' : '#f87171'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : <p className="text-xs text-slate-500">No contributions available — run inference first.</p>}
          </div>
        ) : (
          <p className="text-xs text-slate-500">Run inference to generate SHAP explanation.</p>
        )}
      </div>
    </div>
  )
}
