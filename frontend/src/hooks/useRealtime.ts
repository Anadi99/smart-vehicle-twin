import { useContext } from 'react'

import { RealtimeStateContext } from '../context/realtimeStateContext'
import { initialState, type RealtimeState } from '../state/realtimeStore'

/** Live telemetry state; falls back to defaults outside `RealtimeProvider` (e.g. tests). */
export function useRealtime(): RealtimeState {
  const ctx = useContext(RealtimeStateContext)
  return ctx ?? initialState
}
