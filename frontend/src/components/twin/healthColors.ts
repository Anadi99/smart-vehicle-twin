export type HealthHue = 'red' | 'amber' | 'green'

const hueHex: Record<HealthHue, string> = {
  red: '#f87171',
  amber: '#fbbf24',
  green: '#34d399',
}

export function getHealthColor(latest: { brake_temp_c?: number; battery_soc?: number }): HealthHue {
  const temp = latest.brake_temp_c ?? 0
  const soc = latest.battery_soc ?? 100
  if (temp > 95 || soc < 20) {
    return 'red'
  }
  if (temp > 80 || soc < 35) {
    return 'amber'
  }
  return 'green'
}

export function healthHueToHex(hue: HealthHue): string {
  return hueHex[hue]
}
