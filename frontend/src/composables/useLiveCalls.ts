import { ref } from 'vue'

export interface LiveCall {
  call_id: number | null
  call_sid: string
  phone_number: string
  started_at: Date
}

// Module-level singleton so all components share the same list
const activeCalls = ref<LiveCall[]>([])

export function useLiveCalls() {
  function add(data: { call_id: number | null; call_sid: string; phone_number: string }) {
    // Deduplicate by call_sid
    if (activeCalls.value.some((c) => c.call_sid === data.call_sid)) return
    activeCalls.value.push({ ...data, started_at: new Date() })
  }

  function remove(call_sid: string) {
    const i = activeCalls.value.findIndex((c) => c.call_sid === call_sid)
    if (i !== -1) activeCalls.value.splice(i, 1)
  }

  function clear() {
    activeCalls.value = []
  }

  return { activeCalls, add, remove, clear }
}
