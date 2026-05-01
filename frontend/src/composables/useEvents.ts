/**
 * Shared SSE connection. One EventSource is kept alive for the entire app
 * session; components subscribe to specific event types via useEvents().
 */

import { onMounted, onUnmounted } from 'vue'

type Handler = (data: any) => void

const _handlers = new Map<string, Set<Handler>>()
let _es: EventSource | null = null
let _refCount = 0
let _retryTimer: ReturnType<typeof setTimeout> | null = null

function _connect() {
  if (_es) return
  _es = new EventSource('/api/events')

  _es.onmessage = (e) => {
    try {
      const { type, data } = JSON.parse(e.data)
      _handlers.get(type)?.forEach((h) => h(data))
    } catch {
      // malformed payload — ignore
    }
  }

  _es.onerror = () => {
    _es?.close()
    _es = null
    if (_retryTimer) clearTimeout(_retryTimer)
    _retryTimer = setTimeout(_connect, 4_000)
  }
}

function _disconnect() {
  if (_retryTimer) clearTimeout(_retryTimer)
  _es?.close()
  _es = null
}

export function useEvents(listeners: Record<string, Handler>) {
  onMounted(() => {
    _refCount++
    _connect()
    for (const [type, handler] of Object.entries(listeners)) {
      if (!_handlers.has(type)) _handlers.set(type, new Set())
      _handlers.get(type)!.add(handler)
    }
  })

  onUnmounted(() => {
    for (const [type, handler] of Object.entries(listeners)) {
      _handlers.get(type)?.delete(handler)
    }
    _refCount--
    if (_refCount <= 0) {
      _disconnect()
      _refCount = 0
    }
  })
}
