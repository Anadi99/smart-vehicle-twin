const TOKEN_KEY = 'svt_token'

const DEMO_USER = 'admin'
const DEMO_PASS = 'svtwin2026'
const DEMO_TOKEN = 'demo-token-admin'

export function isDemoMode(): boolean {
  return (import.meta.env.VITE_DEMO_MODE as string | undefined) === 'true'
}

export function getToken(): string | null {
  return sessionStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string): void {
  sessionStorage.setItem(TOKEN_KEY, token)
}

export function clearToken(): void {
  sessionStorage.removeItem(TOKEN_KEY)
}

export async function login(username: string, password: string): Promise<string> {
  // Demo mode — no backend needed, validate credentials locally
  if (isDemoMode()) {
    if (username === DEMO_USER && password === DEMO_PASS) {
      setToken(DEMO_TOKEN)
      return DEMO_TOKEN
    }
    throw new Error('Invalid credentials')
  }

  // Live mode — call real backend
  const base = (import.meta.env.VITE_API_BASE as string | undefined) ?? ''
  const body = new URLSearchParams({ username, password })
  const res = await fetch(`${base}/api/auth/token`, { method: 'POST', body })
  if (!res.ok) throw new Error('Invalid credentials')
  const data = (await res.json()) as { access_token: string }
  setToken(data.access_token)
  return data.access_token
}

/** Fetch wrapper that injects Bearer token automatically. */
export async function authFetch(input: RequestInfo, init: RequestInit = {}): Promise<Response> {
  const token = getToken()
  const headers = new Headers(init.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return fetch(input, { ...init, headers })
}
