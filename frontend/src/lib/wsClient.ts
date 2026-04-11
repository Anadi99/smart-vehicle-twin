export type WsStatus = 'connecting' | 'open' | 'closed' | 'error' | 'reconnecting'

const BASE_DELAY_MS = 1_000
const MAX_DELAY_MS = 30_000
const MAX_RETRIES = 10

function wsUrl(path: string): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}${path}`
}

/**
 * WebSocket client with exponential-backoff auto-reconnect.
 * Returns a cancel function — call it to permanently close the connection.
 */
export function connectTelemetryWs(
  onMessage: (data: unknown) => void,
  onStatus?: (s: WsStatus) => void,
): () => void {
  let socket: WebSocket | null = null
  let retries = 0
  let cancelled = false
  let retryTimer: ReturnType<typeof setTimeout> | null = null

  function connect() {
    if (cancelled) return
    onStatus?.('connecting')
    socket = new WebSocket(wsUrl('/ws/telemetry'))

    socket.onopen = () => {
      retries = 0
      onStatus?.('open')
    }

    socket.onmessage = (ev) => {
      try {
        onMessage(JSON.parse(ev.data as string) as unknown)
      } catch {
        /* ignore malformed frames */
      }
    }

    socket.onerror = () => {
      onStatus?.('error')
    }

    socket.onclose = () => {
      if (cancelled) {
        onStatus?.('closed')
        return
      }
      if (retries >= MAX_RETRIES) {
        onStatus?.('closed')
        return
      }
      const delay = Math.min(BASE_DELAY_MS * 2 ** retries, MAX_DELAY_MS)
      retries++
      onStatus?.('reconnecting')
      retryTimer = setTimeout(connect, delay)
    }
  }

  connect()

  return () => {
    cancelled = true
    if (retryTimer) clearTimeout(retryTimer)
    socket?.close()
  }
}
