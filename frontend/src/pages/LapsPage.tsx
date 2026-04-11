import { useQuery } from '@tanstack/react-query'
import { getJson } from '../lib/api'

type LapRow = {
  lap: number
  samples: number
  speed_mean: number
  speed_max: number
  brake_temp_max: number
  soc_drop: number
  brake_pad_end: number
  tire_wear_end: number
  risk_max: number
  risk_mean: number
}

function riskColor(v: number) {
  if (v >= 0.7) return 'text-rose-400'
  if (v >= 0.4) return 'text-amber-400'
  return 'text-emerald-400'
}

export default function LapsPage() {
  const { data, isLoading } = useQuery({
    queryKey: ['laps'],
    queryFn: () => getJson<{ data: LapRow[] }>('/api/laps'),
    refetchInterval: 5_000,
  })

  const laps = data?.data ?? []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-white">Laps</h1>
        <p className="mt-1 text-sm text-slate-400">
          Per-lap aggregated analytics — speed, brake temp, SOC drop, wear, risk.
        </p>
      </div>

      {isLoading && <p className="text-sm text-slate-500">Loading…</p>}
      {!isLoading && laps.length === 0 && (
        <p className="text-sm text-slate-500">No lap data yet — simulator is warming up.</p>
      )}

      {laps.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-slate-800">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/80 text-xs uppercase tracking-wide text-slate-500">
                {['Lap','Samples','Speed avg','Speed max','Brake max °C','SOC drop %','Pad end','Tire wear','Risk max','Risk avg'].map(h => (
                  <th key={h} className="px-4 py-3 text-left font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...laps].reverse().map((row) => (
                <tr key={row.lap} className="border-b border-slate-800/50 hover:bg-slate-900/40 transition-colors">
                  <td className="px-4 py-2 font-semibold text-cyan-400">{row.lap}</td>
                  <td className="px-4 py-2 text-slate-400">{row.samples}</td>
                  <td className="px-4 py-2 tabular-nums">{row.speed_mean.toFixed(1)}</td>
                  <td className="px-4 py-2 tabular-nums">{row.speed_max.toFixed(1)}</td>
                  <td className="px-4 py-2 tabular-nums">{row.brake_temp_max.toFixed(1)}</td>
                  <td className="px-4 py-2 tabular-nums">{row.soc_drop.toFixed(2)}</td>
                  <td className="px-4 py-2 tabular-nums">{row.brake_pad_end.toFixed(3)}</td>
                  <td className="px-4 py-2 tabular-nums">{row.tire_wear_end.toFixed(4)}</td>
                  <td className={`px-4 py-2 tabular-nums font-medium ${riskColor(row.risk_max)}`}>{row.risk_max.toFixed(3)}</td>
                  <td className={`px-4 py-2 tabular-nums ${riskColor(row.risk_mean)}`}>{row.risk_mean.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
