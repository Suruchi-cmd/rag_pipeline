<template>
  <div class="space-y-4">
    <!-- Filter bar -->
    <div class="bg-white rounded-xl border border-slate-200 px-5 py-4 flex items-center gap-4 flex-wrap">
      <!-- Status filter -->
      <div class="flex items-center gap-2">
        <label class="text-xs font-medium text-slate-500">Status</label>
        <select
          v-model="localStatus"
          class="text-sm border border-slate-200 rounded-lg px-3 py-1.5 text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 bg-white"
          @change="applyFilters"
        >
          <option value="">All</option>
          <option value="active">Active</option>
          <option value="completed">Completed</option>
          <option value="abandoned">Abandoned</option>
        </select>
      </div>

      <!-- Needs human toggle -->
      <button
        class="flex items-center gap-1.5 text-sm px-3 py-1.5 rounded-lg border transition-colors"
        :class="localNeedsHuman
          ? 'bg-red-50 border-red-200 text-red-700 font-medium'
          : 'border-slate-200 text-slate-600 hover:bg-slate-50'"
        @click="toggleNeedsHuman"
      >
        <FlagIcon :size="14" />
        Needs Follow-up
      </button>

      <!-- Clear -->
      <button
        v-if="localStatus || localNeedsHuman"
        class="text-sm text-slate-400 hover:text-slate-600 transition-colors ml-auto"
        @click="clearFilters"
      >
        Clear filters
      </button>

      <div class="ml-auto flex items-center gap-2">
        <button
          class="p-1.5 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-colors"
          title="Refresh"
          @click="store.fetchCalls()"
        >
          <RefreshCwIcon :size="15" :class="store.loading ? 'animate-spin' : ''" />
        </button>
      </div>
    </div>

    <!-- Bulk action bar -->
    <div
      v-if="store.selectedIds.size > 0"
      class="bg-indigo-50 border border-indigo-200 rounded-xl px-5 py-3 flex items-center justify-between"
    >
      <div class="text-sm text-indigo-900">
        <span class="font-semibold">{{ store.selectedIds.size }}</span>
        {{ store.selectedIds.size === 1 ? 'call' : 'calls' }} selected
      </div>
      <div class="flex items-center gap-2">
        <button
          class="text-sm text-indigo-700 hover:text-indigo-900 px-3 py-1.5 rounded-lg hover:bg-indigo-100 transition-colors"
          @click="store.clearSelection()"
        >
          Clear
        </button>
        <button
          class="flex items-center gap-1.5 px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50"
          :disabled="store.deleting"
          @click="confirmDelete"
        >
          <Trash2Icon :size="14" />
          {{ store.deleting ? 'Deleting…' : 'Delete' }}
        </button>
      </div>
    </div>

    <!-- Table -->
    <div class="bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div v-if="store.loading" class="py-12">
        <Spinner full-page />
      </div>

      <div v-else-if="store.error" class="py-16 text-center">
        <AlertCircleIcon :size="36" class="text-red-300 mx-auto mb-3" />
        <p class="text-slate-600 text-sm font-medium">{{ store.error }}</p>
      </div>

      <div v-else-if="store.calls.length === 0" class="py-16 text-center">
        <PhoneOffIcon :size="36" class="text-slate-300 mx-auto mb-3" />
        <p class="text-slate-500 text-sm font-medium">No calls found</p>
        <p class="text-slate-400 text-xs mt-1">Try clearing the filters</p>
      </div>

      <template v-else>
        <table class="w-full text-sm">
          <thead class="bg-slate-50 border-b border-slate-100">
            <tr>
              <th class="px-4 py-3 w-10">
                <input
                  type="checkbox"
                  class="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500 cursor-pointer"
                  :checked="allSelected"
                  :indeterminate.prop="someSelected"
                  @change="toggleAll"
                />
              </th>
              <th class="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider px-6 py-3">Caller</th>
              <th class="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 py-3">Status</th>
              <th class="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 py-3">Duration</th>
              <th class="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 py-3">Turns</th>
              <th class="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 py-3">Avg / req</th>
              <th class="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 py-3">Started</th>
              <th class="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 py-3">Categories</th>
              <th class="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 py-3">Summary</th>
              <th class="px-4 py-3" />
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-50">
            <tr
              v-for="call in store.calls"
              :key="call.id"
              class="hover:bg-slate-50 cursor-pointer transition-colors group"
              :class="store.selectedIds.has(call.id) ? 'bg-indigo-50/40' : ''"
              @click="$router.push(`/calls/${call.id}`)"
            >
              <td class="px-4 py-3.5" @click.stop>
                <input
                  type="checkbox"
                  class="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500 cursor-pointer"
                  :checked="store.selectedIds.has(call.id)"
                  @change="store.toggleSelected(call.id)"
                />
              </td>
              <td class="px-6 py-3.5">
                <div class="flex items-center gap-2">
                  <span class="font-medium text-slate-900">{{ call.phone_number }}</span>
                  <FlagIcon v-if="call.needs_human" :size="12" class="text-red-500 flex-shrink-0" />
                </div>
              </td>
              <td class="px-4 py-3.5">
                <StatusBadge :status="call.status" />
              </td>
              <td class="px-4 py-3.5 text-slate-600">{{ formatDuration(call.started_at, call.ended_at) }}</td>
              <td class="px-4 py-3.5 text-slate-600">{{ call.total_turns }}</td>
              <td class="px-4 py-3.5 text-slate-600 whitespace-nowrap">{{ formatAvgTurn(call.avg_turn_ms) }}</td>
              <td class="px-4 py-3.5 text-slate-500 whitespace-nowrap">{{ formatDate(call.started_at) }}</td>
              <td class="px-4 py-3.5">
                <div class="flex flex-wrap gap-1">
                  <CategoryBadge
                    v-for="cat in (call as any).categories ?? []"
                    :key="cat"
                    :name="cat"
                  />
                </div>
              </td>
              <td class="px-4 py-3.5 text-slate-500 max-w-xs">
                <span v-if="call.summary" :title="call.summary">{{ truncate(call.summary, 60) }}</span>
                <span v-else class="text-slate-300 italic">—</span>
              </td>
              <td class="px-4 py-3.5">
                <ChevronRightIcon :size="16" class="text-slate-300 group-hover:text-slate-400 transition-colors" />
              </td>
            </tr>
          </tbody>
        </table>

        <div class="px-6 py-4 border-t border-slate-100">
          <Pagination
            :offset="store.filters.offset ?? 0"
            :limit="store.filters.limit ?? 50"
            :count="store.calls.length"
            @prev="store.prevPage(); store.fetchCalls()"
            @next="store.nextPage(); store.fetchCalls()"
          />
        </div>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useCallsStore } from '@/stores/calls'
import { useToasts } from '@/composables/useToasts'
import { formatAvgTurn, formatDate, formatDuration, truncate } from '@/utils/format'
import { useEvents } from '@/composables/useEvents'
import StatusBadge from '@/components/StatusBadge.vue'
import CategoryBadge from '@/components/CategoryBadge.vue'
import Spinner from '@/components/Spinner.vue'
import Pagination from '@/components/Pagination.vue'
import {
  Flag as FlagIcon,
  ChevronRight as ChevronRightIcon,
  RefreshCw as RefreshCwIcon,
  AlertCircle as AlertCircleIcon,
  PhoneOff as PhoneOffIcon,
  Trash2 as Trash2Icon,
} from 'lucide-vue-next'

const store = useCallsStore()
const route = useRoute()
const { push: pushToast } = useToasts()

const allSelected = computed(
  () => store.calls.length > 0 && store.selectedIds.size === store.calls.length,
)
const someSelected = computed(
  () => store.selectedIds.size > 0 && store.selectedIds.size < store.calls.length,
)

function toggleAll() {
  if (allSelected.value) store.clearSelection()
  else store.selectAll()
}

async function confirmDelete() {
  const n = store.selectedIds.size
  if (n === 0) return
  const ok = window.confirm(
    `Delete ${n} ${n === 1 ? 'call' : 'calls'}? This will also remove all associated messages, RAG retrievals, and booking changes. This cannot be undone.`,
  )
  if (!ok) return
  try {
    const deleted = await store.deleteSelected()
    pushToast({
      kind: 'ended',
      title: 'Calls deleted',
      body: `${deleted} ${deleted === 1 ? 'call' : 'calls'} removed.`,
      ttl: 4000,
    })
  } catch {
    pushToast({
      kind: 'error',
      title: 'Delete failed',
      body: 'Could not delete the selected calls.',
      ttl: 5000,
    })
  }
}

// Refresh live when calls start or end
useEvents({
  call_started() { store.fetchCalls(); store.fetchStats() },
  call_ended() { store.fetchCalls(); store.fetchStats() },
})

const localStatus = ref('')
const localNeedsHuman = ref(false)

function applyFilters() {
  store.setFilter('status', localStatus.value)
  store.fetchCalls()
}

function toggleNeedsHuman() {
  localNeedsHuman.value = !localNeedsHuman.value
  store.setFilter('needs_human', localNeedsHuman.value ? true : null)
  store.fetchCalls()
}

function clearFilters() {
  localStatus.value = ''
  localNeedsHuman.value = false
  store.setFilter('status', '')
  store.setFilter('needs_human', null)
  store.fetchCalls()
}

onMounted(() => {
  // Read query params from sidebar quick-links
  if (route.query.needs_human === 'true') {
    localNeedsHuman.value = true
    store.filters.needs_human = true
  }
  if (typeof route.query.status === 'string') {
    localStatus.value = route.query.status
    store.filters.status = route.query.status
  }
  store.fetchCalls()
})

// Re-apply when route query changes (e.g. clicking sidebar links)
watch(() => route.query, (q) => {
  localNeedsHuman.value = q.needs_human === 'true'
  localStatus.value = typeof q.status === 'string' ? q.status : ''
  store.setFilter('needs_human', localNeedsHuman.value ? true : null)
  store.setFilter('status', localStatus.value)
  store.fetchCalls()
})
</script>
