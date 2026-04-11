import { createContext } from 'react'

import type { RealtimeState } from '../state/realtimeStore'

export const RealtimeStateContext = createContext<RealtimeState | null>(null)
