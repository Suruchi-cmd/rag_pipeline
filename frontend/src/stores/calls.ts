import { defineStore } from 'pinia'
import { ref, reactive } from 'vue'
import { bulkDeleteCalls, listCalls, getStats, type ListCallsParams } from '@/api/calls'
import type { Call, Stats } from '@/types'

export const useCallsStore = defineStore('calls', () => {
  const calls = ref<Call[]>([])
  const stats = ref<Stats | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)
  const selectedIds = ref<Set<number>>(new Set())
  const deleting = ref(false)

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
      // Drop selected ids that are no longer in view
      const visible = new Set(calls.value.map((c) => c.id))
      selectedIds.value = new Set([...selectedIds.value].filter((id) => visible.has(id)))
    } catch {
      error.value = 'Failed to load calls. Is the API running?'
    } finally {
      loading.value = false
    }
  }

  function toggleSelected(id: number) {
    const next = new Set(selectedIds.value)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    selectedIds.value = next
  }

  function selectAll() {
    selectedIds.value = new Set(calls.value.map((c) => c.id))
  }

  function clearSelection() {
    selectedIds.value = new Set()
  }

  async function deleteSelected(): Promise<number> {
    const ids = [...selectedIds.value]
    if (ids.length === 0) return 0
    deleting.value = true
    try {
      const { deleted } = await bulkDeleteCalls(ids)
      clearSelection()
      await Promise.all([fetchCalls(), fetchStats()])
      return deleted
    } finally {
      deleting.value = false
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

  return {
    calls,
    stats,
    loading,
    error,
    filters,
    selectedIds,
    deleting,
    fetchCalls,
    fetchStats,
    setFilter,
    nextPage,
    prevPage,
    toggleSelected,
    selectAll,
    clearSelection,
    deleteSelected,
  }
})
