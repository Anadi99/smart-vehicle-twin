import { applyEvent, initialState } from '../state/realtimeStore'

test('applies telemetry.update into latest snapshot', () => {
  const next = applyEvent(initialState, {
    type: 'telemetry.update',
    ts: '2026-01-01T00:00:00Z',
    payload: { speed_kph: 100 },
  })
  expect(next.latest.speed_kph).toBe(100)
})
