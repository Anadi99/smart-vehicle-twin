import { HealthOverlay } from '../components/twin/HealthOverlay'
import { VehicleScene } from '../components/twin/VehicleScene'
import { useRealtime } from '../hooks/useRealtime'

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 px-3 py-2 text-center">
      <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-0.5 text-sm font-semibold tabular-nums text-slate-100">{value}</div>
    </div>
  )
}

function fmt(v: unknown, digits = 1, unit = ''): string {
  if (typeof v === 'number' && Number.isFinite(v)) return `${v.toFixed(digits)}${unit}`
  return '—'
}

export default function Twin3DPage() {
  const { latest } = useRealtime()

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold text-white">3D Twin</h1>
        <p className="mt-1 text-sm text-slate-400">
          Live vehicle proxy — color driven by brake temp &amp; SOC health. Orbits the track at real speed.
        </p>
      </div>

      <HealthOverlay latest={latest} />

      {/* Live stats strip */}
      <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
        <Stat label="Speed"      value={fmt(latest.speed_kph, 1, ' km/h')} />
        <Stat label="Brake °C"   value={fmt(latest.brake_temp_c, 1, '°C')} />
        <Stat label="SOC"        value={fmt(latest.battery_soc, 1, '%')} />
        <Stat label="Pad"        value={fmt(latest.brake_pad_frac, 3)} />
        <Stat label="Tire wear"  value={fmt(latest.tire_wear_frac, 3)} />
        <Stat label="Risk"       value={fmt(latest.risk, 2)} />
      </div>

      <VehicleScene latest={latest} />

      {/* GPS coords */}
      {typeof latest.lat === 'number' && (
        <p className="text-xs text-slate-600">
          GPS: {(latest.lat as number).toFixed(5)}, {(latest.lon as number).toFixed(5)}
        </p>
      )}
    </div>
  )
}
