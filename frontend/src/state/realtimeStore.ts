export type WsStatus = 'connecting' | 'open' | 'closed' | 'error' | 'reconnecting'

export type RealtimeState = {
  latest: Record<string, unknown>
  connected: boolean
  wsStatus: WsStatus
}

export const initialState: RealtimeState = {
  latest: {},
  connected: false,
  wsStatus: 'closed',
}

export function applyEvent(state: RealtimeState, event: unknown): RealtimeState {
  if (!event || typeof event !== 'object') return state
  const e = event as Record<string, unknown>

  if (e.type === 'snapshot' && e.data && typeof e.data === 'object') {
    return { ...state, latest: { ...state.latest, ...(e.data as Record<string, unknown>) }, connected: true }
  }
  if (e.type === 'telemetry.update' && e.payload && typeof e.payload === 'object') {
    return { ...state, latest: { ...state.latest, ...(e.payload as Record<string, unknown>) }, connected: true }
  }
  return state
}
