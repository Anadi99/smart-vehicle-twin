import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'

import { getJson } from '../lib/api'
import { demoTick, isDemoMode } from '../lib/demoData'
import { connectTelemetryWs, type WsStatus } from '../lib/wsClient'
import { RealtimeStateContext } from './realtimeStateContext'
import { applyEvent, initialState, type RealtimeState } from '../state/realtimeStore'

export function RealtimeProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<RealtimeState>(initialState)

  const onStatus = useCallback((wsStatus: WsStatus) => {
    setState(s => ({ ...s, wsStatus, connected: wsStatus === 'open' }))
  }, [])

  useEffect(() => {
    // ── Demo mode: no backend needed ──────────────────────────────────
    if (isDemoMode()) {
      setState(s => ({ ...s, wsStatus: 'open', connected: true, latest: demoTick() }))
      const id = setInterval(() => {
        const payload = demoTick()
        setState(s => applyEvent(s, { type: 'telemetry.update', payload }))
      }, 1000)
      return () => clearInterval(id)
    }

    // ── Live mode: real backend ────────────────────────────────────────
    getJson<{ data: Record<string, unknown> }>('/api/telemetry/latest')
      .then(res => setState(s => ({ ...s, latest: { ...s.latest, ...res.data } })))
      .catch(() => { /* seed optional */ })

    const cancel = connectTelemetryWs(
      msg => setState(s => applyEvent(s, msg)),
      onStatus,
    )
    return () => cancel()
  }, [onStatus])

  const value = useMemo(() => state, [state])
  return (
    <RealtimeStateContext.Provider value={value}>
      {children}
    </RealtimeStateContext.Provider>
  )
}
