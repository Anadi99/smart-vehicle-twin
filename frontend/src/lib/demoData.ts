/**
 * Demo mode — generates realistic fake telemetry in the browser.
 * Activates automatically when VITE_DEMO_MODE=true or when the backend
 * is unreachable (WS stays closed after first attempt).
 */

let _t = 0

export function isDemoMode(): boolean {
  return (import.meta.env.VITE_DEMO_MODE as string | undefined) === 'true'
}

export function demoTick(): Record<string, unknown> {
  _t += 1
  const speed = 60 + 40 * Math.sin(_t / 30) + (Math.random() - 0.5) * 5
  const brakeTemp = 30 + speed * 0.7 + (Math.random() - 0.5) * 4
  const soc = Math.max(20, 82 - _t * 0.015 + (Math.random() - 0.5) * 0.1)
  const pad = Math.max(0.15, 0.28 - _t * 0.00005)
  const tire = Math.min(0.4, 0.05 + _t * 0.0002)
  const risk = brakeTemp > 90 ? 0.75 : brakeTemp > 75 ? 0.45 : soc < 20 ? 0.6 : 0.1
  const reasons = risk > 0.7 ? 'high brake temp' : risk > 0.4 ? 'brake temp warning' : 'nominal'

  return {
    speed_kph:      Math.round(speed * 10) / 10,
    brake_temp_c:   Math.round(brakeTemp * 10) / 10,
    battery_soc:    Math.round(soc * 100) / 100,
    brake_pad_frac: Math.round(pad * 1000) / 1000,
    tire_wear_frac: Math.round(tire * 10000) / 10000,
    accel_mps2:     Math.round((Math.random() - 0.5) * 4 * 100) / 100,
    lat:            50.3357 + Math.random() * 0.001,
    lon:            6.9470  + Math.random() * 0.001,
    risk:           Math.round(risk * 1000) / 1000,
    reasons,
    ts:             Date.now() / 1000,
  }
}
