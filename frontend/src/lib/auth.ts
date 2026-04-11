const TOKEN_KEY = 'svt_token'

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
