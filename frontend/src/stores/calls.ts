import { defineStore } from 'pinia'
import { ref, reactive } from 'vue'
import { listCalls, getStats, type ListCallsParams } from '@/api/calls'
import type { Call, Stats } from '@/types'

export const useCallsStore = defineStore('calls', () => {
  const calls = ref<Call[]>([])
  const stats = ref<Stats | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  const filters = reactive<ListCallsParams>({
    offset: 0,
    limit: 50,
    needs_human: null,
    status: '',
  })

  async function fetchCalls() {
    loading.value = true
    error.value = null
    try {
      calls.value = await listCalls(filters)
    } catch {
      error.value = 'Failed to load calls. Is the API running?'
    } finally {
      loading.value = false
    }
  }

  async function fetchStats() {
    try {
      stats.value = await getStats()
    } catch {
      // non-critical
    }
  }

  function setFilter<K extends keyof ListCallsParams>(key: K, value: ListCallsParams[K]) {
    filters[key] = value
    filters.offset = 0
  }

  function nextPage() {
    filters.offset = (filters.offset ?? 0) + (filters.limit ?? 50)
  }

  function prevPage() {
    filters.offset = Math.max(0, (filters.offset ?? 0) - (filters.limit ?? 50))
  }

  return { calls, stats, loading, error, filters, fetchCalls, fetchStats, setFilter, nextPage, prevPage }
})
