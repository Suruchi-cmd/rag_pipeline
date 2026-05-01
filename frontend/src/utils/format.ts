export function parseUTC(dateStr: string): Date {
  return new Date(dateStr.endsWith('Z') ? dateStr : `${dateStr}Z`)
}

export function formatDate(dateStr: string): string {
  return parseUTC(dateStr).toLocaleString('en-CA', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function formatDateFull(dateStr: string): string {
  return parseUTC(dateStr).toLocaleString('en-CA', {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

export function formatDuration(startedAt: string, endedAt: string | null): string {
  if (!endedAt) return 'Active'
  const secs = Math.floor(
    (parseUTC(endedAt).getTime() - parseUTC(startedAt).getTime()) / 1000,
  )
  const m = Math.floor(secs / 60)
  const s = secs % 60
  return m > 0 ? `${m}m ${s}s` : `${s}s`
}

export function truncate(str: string, max: number): string {
  return str.length > max ? `${str.slice(0, max)}…` : str
}

export function formatAvgTurn(ms: number | null | undefined): string {
  if (ms == null) return '—'
  const s = ms / 1000
  return s >= 1 ? `${s.toFixed(2)}s / request` : `${Math.round(ms)}ms / request`
}
