<template>
  <div class="space-y-6">
    <!-- Stat cards -->
    <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
      <div
        v-for="card in statCards"
        :key="card.label"
        class="bg-white rounded-xl border border-slate-200 p-5 flex items-center gap-4"
      >
        <div class="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0" :class="card.iconBg">
          <component :is="card.icon" :size="20" :class="card.iconColor" />
        </div>
        <div>
          <p class="text-2xl font-bold text-slate-900">
            {{ store.stats ? card.value : '—' }}
          </p>
          <p class="text-sm text-slate-500 mt-0.5">{{ card.label }}</p>
        </div>
      </div>
    </div>

    <!-- Recent calls -->
    <div class="bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div class="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
        <div class="flex items-center gap-2">
          <h2 class="font-semibold text-slate-900 text-sm">Recent Calls</h2>
          <span
            v-if="store.stats?.active_calls"
            class="text-xs bg-green-100 text-green-700 font-medium px-2 py-0.5 rounded-full flex items-center gap-1"
          >
            <span class="relative flex h-1.5 w-1.5">
              <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-500 opacity-75" />
              <span class="relative inline-flex rounded-full h-1.5 w-1.5 bg-green-500" />
            </span>
            {{ store.stats.active_calls }} live
          </span>
        </div>
        <div class="flex items-center gap-3">
          <span class="text-xs text-slate-400 flex items-center gap-1">
            <span
              class="w-1.5 h-1.5 rounded-full inline-block"
              :class="sseConnected ? 'bg-green-400' : 'bg-slate-300'"
            />
            {{ sseConnected ? 'Live' : 'Connecting…' }}
          </span>
          <RouterLink to="/calls" class="text-xs text-indigo-600 font-medium hover:underline">
            View all →
          </RouterLink>
        </div>
      </div>

      <div v-if="loading" class="py-12">
        <Spinner full-page />
      </div>

      <div v-else-if="recentCalls.length === 0" class="py-16 text-center">
        <PhoneOffIcon :size="36" class="text-slate-300 mx-auto mb-3" />
        <p class="text-slate-500 text-sm font-medium">No calls yet</p>
        <p class="text-slate-400 text-xs mt-1">Calls will appear here once the voice server receives one</p>
      </div>

      <table v-else class="w-full text-sm">
        <thead class="bg-slate-50 border-b border-slate-100">
          <tr>
            <th class="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider px-6 py-3">Caller</th>
            <th class="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 py-3">Status</th>
            <th class="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 py-3">Duration</th>
            <th class="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 py-3">Turns</th>
            <th class="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 py-3">Started</th>
            <th class="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 py-3">Flag</th>
            <th class="px-4 py-3" />
          </tr>
        </thead>
        <tbody class="divide-y divide-slate-50">
          <tr
            v-for="call in recentCalls"
            :key="call.id"
            class="hover:bg-slate-50 cursor-pointer transition-colors"
            :class="flashIds.has(call.id ?? -1) ? 'bg-green-50' : ''"
            @click="call.id && $router.push(`/calls/${call.id}`)"
          >
            <td class="px-6 py-3.5 font-medium text-slate-900">
              {{ call.phone_number }}
            </td>
            <td class="px-4 py-3.5">
              <StatusBadge :status="call.status" />
            </td>
            <td class="px-4 py-3.5 text-slate-600">
              {{ formatDuration(call.started_at, call.ended_at) }}
            </td>
            <td class="px-4 py-3.5 text-slate-600">{{ call.total_turns }}</td>
            <td class="px-4 py-3.5 text-slate-500">{{ formatDate(call.started_at) }}</td>
            <td class="px-4 py-3.5">
              <span v-if="call.needs_human" class="inline-flex items-center gap-1 text-xs text-red-600 font-medium">
                <FlagIcon :size="12" />
                Follow-up
              </span>
            </td>
            <td class="px-4 py-3.5">
              <ChevronRightIcon :size="16" class="text-slate-300" />
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onUnmounted } from 'vue'
import { useCallsStore } from '@/stores/calls'
import { listCalls, getCall } from '@/api/calls'
import { formatDate, formatDuration } from '@/utils/format'
import { useEvents } from '@/composables/useEvents'
import StatusBadge from '@/components/StatusBadge.vue'
import Spinner from '@/components/Spinner.vue'
import type { Call } from '@/types'
import {
  Phone as PhoneIcon,
  Flag as FlagIcon,
  Activity as ActivityIcon,
  ChevronRight as ChevronRightIcon,
  PhoneOff as PhoneOffIcon,
} from 'lucide-vue-next'

const store = useCallsStore()
const recentCalls = ref<Call[]>([])
const loading = ref(false)
const sseConnected = ref(false)
const flashIds = reactive(new Set<number>())

const statCards = computed(() => [
  {
    label: 'Total Calls',
    value: store.stats?.total_calls ?? 0,
    icon: PhoneIcon,
    iconBg: 'bg-indigo-50',
    iconColor: 'text-indigo-600',
  },
  {
    label: 'Needs Follow-up',
    value: store.stats?.needs_human ?? 0,
    icon: FlagIcon,
    iconBg: 'bg-red-50',
    iconColor: 'text-red-500',
  },
  {
    label: 'Active Now',
    value: store.stats?.active_calls ?? 0,
    icon: ActivityIcon,
    iconBg: 'bg-green-50',
    iconColor: 'text-green-600',
  },
])

async function load() {
  loading.value = true
  try {
    const [calls] = await Promise.all([listCalls({ limit: 10 }), store.fetchStats()])
    recentCalls.value = calls
  } finally {
    loading.value = false
  }
}

function flashRow(callId: number) {
  flashIds.add(callId)
  setTimeout(() => flashIds.delete(callId), 2_000)
}

// SSE: react immediately to call events — no polling needed
useEvents({
  async call_started(data) {
    sseConnected.value = true
    // Fetch the newly created call record from DB
    if (data.call_id) {
      try {
        const detail = await getCall(data.call_id)
        const exists = recentCalls.value.some((c) => c.id === detail.call.id)
        if (!exists) {
          recentCalls.value = [detail.call, ...recentCalls.value].slice(0, 10)
          flashRow(detail.call.id)
        }
      } catch {
        // DB row may not be committed yet; fall back to a full reload
        await load()
      }
    }
    store.fetchStats()
  },

  call_ended(data) {
    sseConnected.value = true
    if (data.call_id == null) { store.fetchStats(); return }
    const idx = recentCalls.value.findIndex((c) => c.id === data.call_id)
    const existing = idx !== -1 ? recentCalls.value[idx] : undefined
    if (idx !== -1 && existing) {
      recentCalls.value[idx] = {
        ...existing,
        status: data.status as Call['status'],
        needs_human: Boolean(data.needs_human),
        summary: data.summary ?? existing.summary,
        ended_at: new Date().toISOString(),
      }
      flashRow(data.call_id)
    }
    store.fetchStats()
  },
})

// Fallback poll every 60 s to catch anything SSE might miss
let timer: ReturnType<typeof setInterval>
onMounted(() => {
  load()
  timer = setInterval(load, 60_000)
})
onUnmounted(() => clearInterval(timer))
</script>
