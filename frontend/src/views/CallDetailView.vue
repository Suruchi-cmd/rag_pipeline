<template>
  <div v-if="loading" class="py-16">
    <Spinner full-page size="lg" />
  </div>

  <div v-else-if="error" class="py-16 text-center">
    <AlertCircleIcon :size="40" class="text-red-300 mx-auto mb-3" />
    <p class="text-slate-600 font-medium">{{ error }}</p>
    <button class="mt-4 text-sm text-indigo-600 hover:underline" @click="$router.back()">Go back</button>
  </div>

  <div v-else-if="detail" class="space-y-5">
    <!-- Back + header -->
    <div class="flex items-start gap-4">
      <button
        class="mt-0.5 p-1.5 rounded-lg hover:bg-slate-200 text-slate-400 hover:text-slate-700 transition-colors flex-shrink-0"
        @click="$router.push('/calls')"
      >
        <ArrowLeftIcon :size="18" />
      </button>
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-3 flex-wrap">
          <h2 class="text-xl font-bold text-slate-900">{{ detail.call.phone_number }}</h2>
          <StatusBadge :status="detail.call.status" />
          <span
            v-if="detail.call.needs_human"
            class="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold bg-red-100 text-red-700"
          >
            <FlagIcon :size="11" /> Needs Follow-up
          </span>
          <CategoryBadge
            v-for="cat in (detail as any).categories ?? []"
            :key="cat"
            :name="cat"
          />
        </div>
        <div class="flex items-center gap-4 mt-1.5 text-sm text-slate-500">
          <span class="flex items-center gap-1.5">
            <CalendarIcon :size="13" />
            {{ formatDateFull(detail.call.started_at) }}
          </span>
          <span class="flex items-center gap-1.5">
            <ClockIcon :size="13" />
            {{ formatDuration(detail.call.started_at, detail.call.ended_at) }}
          </span>
          <span class="flex items-center gap-1.5">
            <MessageSquareIcon :size="13" />
            {{ detail.call.total_turns }} turns
          </span>
          <span
            v-if="detail.call.avg_turn_ms != null"
            class="flex items-center gap-1.5"
            title="Average time spent generating an answer — from when the caller stopped talking to when the bot starts replying"
          >
            <GaugeIcon :size="13" />
            {{ formatAvgTurn(detail.call.avg_turn_ms) }}
          </span>
        </div>
      </div>
    </div>

    <!-- Flagged banner -->
    <div
      v-if="detail.call.needs_human && detail.call.flag_reason"
      class="bg-red-50 border border-red-200 rounded-xl px-5 py-3 flex items-start gap-3"
    >
      <FlagIcon :size="16" class="text-red-500 mt-0.5 flex-shrink-0" />
      <div>
        <p class="text-sm font-semibold text-red-800">Human follow-up required</p>
        <p class="text-sm text-red-600 mt-0.5">{{ detail.call.flag_reason }}</p>
      </div>
    </div>

    <!-- Main layout: transcript + sidebar -->
    <div class="flex gap-5 items-start">
      <!-- Transcript -->
      <div class="flex-1 min-w-0 bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div class="px-5 py-4 border-b border-slate-100 flex items-center justify-between">
          <h3 class="font-semibold text-slate-900 text-sm">Transcript</h3>
          <span class="text-xs text-slate-400">{{ detail.messages.length }} messages</span>
        </div>

        <div v-if="turns.length === 0" class="py-16 text-center">
          <MessageSquareIcon :size="32" class="text-slate-200 mx-auto mb-3" />
          <p class="text-slate-400 text-sm">No messages recorded</p>
        </div>

        <div v-else class="p-5 space-y-6">
          <div v-for="turn in turns" :key="turn.turn_number" class="space-y-2">
            <!-- Turn label -->
            <p class="text-xs text-slate-400 font-medium text-center">Turn {{ turn.turn_number }}</p>

            <!-- User message -->
            <div v-if="turn.user" class="flex items-start gap-3">
              <div class="w-7 h-7 rounded-full bg-slate-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                <UserIcon :size="14" class="text-slate-500" />
              </div>
              <div class="flex-1 min-w-0">
                <p class="text-xs text-slate-400 font-medium mb-1">Caller</p>
                <div class="bg-slate-100 rounded-xl rounded-tl-sm px-4 py-2.5 inline-block max-w-full">
                  <p class="text-sm text-slate-800 whitespace-pre-wrap break-words">{{ turn.user.content }}</p>
                </div>
                <!-- RAG toggle -->
                <div v-if="turn.rag" class="mt-1.5">
                  <button
                    class="flex items-center gap-1.5 text-xs text-slate-400 hover:text-indigo-600 transition-colors"
                    @click="toggleRag(turn.turn_number)"
                  >
                    <DatabaseIcon :size="11" />
                    <span v-if="turn.rag.was_skipped">RAG skipped</span>
                    <span v-else>{{ turn.rag.retrieved_chunks.length }} chunks retrieved</span>
                    <ChevronDownIcon
                      :size="12"
                      class="transition-transform"
                      :class="expandedRag.has(turn.turn_number) ? 'rotate-180' : ''"
                    />
                  </button>

                  <!-- RAG panel -->
                  <div
                    v-if="expandedRag.has(turn.turn_number)"
                    class="mt-2 bg-indigo-50 border border-indigo-100 rounded-xl p-4 space-y-3"
                  >
                    <!-- Queries -->
                    <div class="space-y-1.5">
                      <div class="flex gap-2 text-xs">
                        <span class="text-indigo-400 font-medium w-20 flex-shrink-0">Original</span>
                        <span class="text-indigo-700">{{ turn.rag.original_query }}</span>
                      </div>
                      <div
                        v-if="turn.rag.rewritten_query !== turn.rag.original_query && !turn.rag.was_skipped"
                        class="flex gap-2 text-xs"
                      >
                        <span class="text-indigo-400 font-medium w-20 flex-shrink-0">Rewritten</span>
                        <span class="text-indigo-800 font-medium">{{ turn.rag.rewritten_query }}</span>
                      </div>
                    </div>

                    <!-- Chunks -->
                    <div v-if="!turn.rag.was_skipped && turn.rag.retrieved_chunks.length > 0" class="space-y-2">
                      <p class="text-xs font-semibold text-indigo-500 uppercase tracking-wider">Retrieved chunks</p>
                      <div
                        v-for="(chunk, ci) in turn.rag.retrieved_chunks"
                        :key="ci"
                        class="bg-white border border-indigo-100 rounded-lg p-3"
                      >
                        <div class="flex items-center justify-between mb-1.5">
                          <span class="text-xs font-medium text-indigo-700 bg-indigo-100 px-2 py-0.5 rounded-full">
                            {{ chunk.metadata?.file_name?.replace('.md', '') ?? 'unknown' }}
                          </span>
                          <span class="text-xs text-slate-400">
                            Score {{ (chunk.score * 100).toFixed(0) }}%
                          </span>
                        </div>
                        <p class="text-xs text-slate-600 leading-relaxed">
                          {{ expandedChunks.has(`${turn.turn_number}-${ci}`) ? chunk.content : truncate(chunk.content, 200) }}
                        </p>
                        <button
                          v-if="chunk.content.length > 200"
                          class="text-xs text-indigo-500 hover:underline mt-1"
                          @click="toggleChunk(`${turn.turn_number}-${ci}`)"
                        >
                          {{ expandedChunks.has(`${turn.turn_number}-${ci}`) ? 'Show less' : 'Show more' }}
                        </button>
                      </div>
                    </div>

                    <p v-else-if="turn.rag.was_skipped" class="text-xs text-indigo-400 italic">
                      Conversational message — RAG retrieval skipped
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <!-- Assistant message -->
            <div v-if="turn.assistant" class="flex items-start gap-3 flex-row-reverse">
              <div class="w-7 h-7 rounded-full bg-indigo-600 flex items-center justify-center flex-shrink-0 mt-0.5">
                <BotIcon :size="14" class="text-white" />
              </div>
              <div class="flex-1 min-w-0 flex flex-col items-end">
                <p class="text-xs text-slate-400 font-medium mb-1">Maya</p>
                <div
                  class="bg-indigo-600 rounded-xl rounded-tr-sm px-4 py-2.5 inline-block max-w-prose"
                  :class="turn.assistant.was_interrupted ? 'opacity-80' : ''"
                >
                  <p class="text-sm text-white whitespace-pre-wrap break-words">{{ turn.assistant.content }}</p>
                </div>
                <div v-if="turn.assistant.was_interrupted" class="flex items-center gap-1 mt-1">
                  <ZapIcon :size="10" class="text-amber-500" />
                  <span class="text-xs text-amber-500">interrupted</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Summary card -->
          <div v-if="detail.call.summary" class="pt-4 border-t border-slate-100">
            <p class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Call Summary</p>
            <p class="text-sm text-slate-700 leading-relaxed">{{ detail.call.summary }}</p>
          </div>
        </div>
      </div>

      <!-- Sidebar -->
      <div class="w-72 flex-shrink-0 space-y-4">
        <!-- Booking change card -->
        <div
          v-if="detail.booking_change"
          class="bg-white rounded-xl border border-amber-200 overflow-hidden"
        >
          <div class="bg-amber-50 px-4 py-3 border-b border-amber-200 flex items-center gap-2">
            <CalendarCheckIcon :size="15" class="text-amber-600" />
            <p class="text-sm font-semibold text-amber-800">Booking Change Request</p>
          </div>
          <div class="p-4 space-y-3">
            <div>
              <p class="text-xs text-slate-400 font-medium">Customer Name</p>
              <p class="text-sm text-slate-900 font-medium mt-0.5">{{ detail.booking_change.caller_name }}</p>
            </div>
            <div>
              <p class="text-xs text-slate-400 font-medium">Callback Number</p>
              <p class="text-sm text-slate-900 mt-0.5">{{ detail.booking_change.caller_phone }}</p>
            </div>
            <div>
              <p class="text-xs text-slate-400 font-medium">Change Requested</p>
              <p class="text-sm text-slate-700 leading-relaxed mt-0.5">{{ detail.booking_change.change_details }}</p>
            </div>
            <p class="text-xs text-slate-400">{{ formatDateFull(detail.booking_change.created_at) }}</p>
          </div>
        </div>

        <!-- Call metadata card -->
        <div class="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div class="px-4 py-3 border-b border-slate-100">
            <p class="text-sm font-semibold text-slate-900">Call Info</p>
          </div>
          <div class="p-4 space-y-3">
            <div v-for="row in metaRows" :key="row.label">
              <p class="text-xs text-slate-400 font-medium">{{ row.label }}</p>
              <p class="text-sm text-slate-900 mt-0.5">{{ row.value }}</p>
            </div>
          </div>
        </div>

        <!-- RAG summary card -->
        <div v-if="detail.rag_retrievals.length > 0" class="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div class="px-4 py-3 border-b border-slate-100 flex items-center gap-2">
            <DatabaseIcon :size="14" class="text-indigo-500" />
            <p class="text-sm font-semibold text-slate-900">RAG Summary</p>
          </div>
          <div class="p-4 space-y-2">
            <div class="flex justify-between text-sm">
              <span class="text-slate-500">Total retrievals</span>
              <span class="font-medium text-slate-900">{{ detail.rag_retrievals.length }}</span>
            </div>
            <div class="flex justify-between text-sm">
              <span class="text-slate-500">Skipped turns</span>
              <span class="font-medium text-slate-900">{{ skippedCount }}</span>
            </div>
            <div class="flex justify-between text-sm">
              <span class="text-slate-500">Avg chunks/turn</span>
              <span class="font-medium text-slate-900">{{ avgChunks }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, reactive } from 'vue'
import { useRoute } from 'vue-router'
import { getCall } from '@/api/calls'
import { formatAvgTurn, formatDateFull, formatDuration, truncate } from '@/utils/format'
import StatusBadge from '@/components/StatusBadge.vue'
import CategoryBadge from '@/components/CategoryBadge.vue'
import Spinner from '@/components/Spinner.vue'
import type { CallDetail, Turn } from '@/types'
import {
  ArrowLeft as ArrowLeftIcon,
  Flag as FlagIcon,
  Calendar as CalendarIcon,
  Clock as ClockIcon,
  MessageSquare as MessageSquareIcon,
  User as UserIcon,
  Bot as BotIcon,
  Database as DatabaseIcon,
  ChevronDown as ChevronDownIcon,
  Zap as ZapIcon,
  AlertCircle as AlertCircleIcon,
  CalendarCheck as CalendarCheckIcon,
  Gauge as GaugeIcon,
} from 'lucide-vue-next'

const route = useRoute()
const detail = ref<CallDetail | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

const expandedRag = reactive(new Set<number>())
const expandedChunks = reactive(new Set<string>())

function toggleRag(turnNum: number) {
  expandedRag.has(turnNum) ? expandedRag.delete(turnNum) : expandedRag.add(turnNum)
}
function toggleChunk(key: string) {
  expandedChunks.has(key) ? expandedChunks.delete(key) : expandedChunks.add(key)
}

const turns = computed<Turn[]>(() => {
  if (!detail.value) return []
  const { messages, rag_retrievals } = detail.value
  const ragByTurn = new Map(rag_retrievals.map((r) => [r.turn_number, r]))
  const turnMap = new Map<number, { user?: (typeof messages)[0]; assistant?: (typeof messages)[0] }>()

  for (const msg of messages) {
    if (!turnMap.has(msg.turn_number)) turnMap.set(msg.turn_number, {})
    const t = turnMap.get(msg.turn_number)!
    if (msg.role === 'user') t.user = msg
    else t.assistant = msg
  }

  return Array.from(turnMap.entries())
    .sort(([a], [b]) => a - b)
    .map(([turn_number, msgs]) => ({
      turn_number,
      user: msgs.user ?? null,
      assistant: msgs.assistant ?? null,
      rag: ragByTurn.get(turn_number) ?? null,
    }))
})

const skippedCount = computed(() => detail.value?.rag_retrievals.filter((r) => r.was_skipped).length ?? 0)

const avgChunks = computed(() => {
  if (!detail.value) return '—'
  const nonSkipped = detail.value.rag_retrievals.filter((r) => !r.was_skipped)
  if (nonSkipped.length === 0) return '—'
  const total = nonSkipped.reduce((sum, r) => sum + r.retrieved_chunks.length, 0)
  return (total / nonSkipped.length).toFixed(1)
})

const metaRows = computed(() => {
  if (!detail.value) return []
  const c = detail.value.call
  return [
    { label: 'Call SID', value: c.call_sid },
    { label: 'Phone', value: c.phone_number },
    { label: 'Status', value: c.status },
    { label: 'Started', value: formatDateFull(c.started_at) },
    { label: 'Duration', value: formatDuration(c.started_at, c.ended_at) },
    { label: 'Turns', value: String(c.total_turns) },
    { label: 'Avg / request', value: formatAvgTurn(c.avg_turn_ms) },
  ]
})

onMounted(async () => {
  const id = Number(route.params.id)
  if (!id) { error.value = 'Invalid call ID'; return }
  loading.value = true
  try {
    detail.value = await getCall(id)
  } catch {
    error.value = 'Failed to load call. It may not exist.'
  } finally {
    loading.value = false
  }
})
</script>
