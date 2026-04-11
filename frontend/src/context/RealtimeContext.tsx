import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'

import { getJson } from '../lib/api'
import { connectTelemetryWs, type WsStatus } from '../lib/wsClient'
import { RealtimeStateContext } from './realtimeStateContext'
import { applyEvent, initialState, type RealtimeState } from '../state/realtimeStore'

export function RealtimeProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<RealtimeState>(initialState)

  const onStatus = useCallback((wsStatus: WsStatus) => {
    setState((s) => ({ ...s, wsStatus, connected: wsStatus === 'open' }))
  }, [])

  useEffect(() => {
    let cancel = () => {}
    getJson<{ data: Record<string, unknown> }>('/api/telemetry/latest')
      .then((res) => {
        setState((s) => ({
          ...s,
          latest: { ...s.latest, ...res.data },
        }))
      })
      .catch(() => {
        /* seed optional */
      })
    cancel = connectTelemetryWs(
      (msg) => setState((s) => applyEvent(s, msg)),
      onStatus,
    )
    return () => cancel()
  }, [onStatus])

  const value = useMemo(() => state, [state])
  return <RealtimeStateContext.Provider value={value}>{children}</RealtimeStateContext.Provider>
}
