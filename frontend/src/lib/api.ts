const base = () => (import.meta.env.VITE_API_BASE as string | undefined) ?? ''

export async function getJson<T>(path: string): Promise<T> {
  const r = await fetch(`${base()}${path}`)
  if (!r.ok) {
    throw new Error(`${path}: ${r.status} ${r.statusText}`)
  }
  return r.json() as Promise<T>
}

export async function postJson<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(`${base()}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (!r.ok) {
    throw new Error(`${path}: ${r.status} ${r.statusText}`)
  }
  return r.json() as Promise<T>
}
