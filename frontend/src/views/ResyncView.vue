<template>
  <div class="max-w-3xl space-y-6">
    <div>
      <h2 class="text-lg font-bold text-slate-900">Resync Knowledge Base</h2>
      <p class="text-sm text-slate-500 mt-0.5">
        Re-pull the source data from Google Sheets and rebuild the RAG vector index.
        Use enriched resync for production-quality answers; use raw resync when you
        need a fast refresh or the enrichment LLM is unavailable.
      </p>
    </div>

    <!-- Enriched resync -->
    <div class="bg-white rounded-xl border border-slate-200 p-5">
      <div class="flex items-start justify-between gap-4">
        <div class="flex items-start gap-3">
          <div class="w-10 h-10 rounded-lg bg-indigo-50 flex items-center justify-center flex-shrink-0">
            <SparklesIcon :size="18" class="text-indigo-600" />
          </div>
          <div>
            <p class="text-sm font-semibold text-slate-900">Enriched Resync</p>
            <p class="text-xs text-slate-500 mt-1 leading-relaxed">
              Feeds the parsed data through the enrichment LLM to produce voicebot-friendly
              prose, then drops the old embeddings and re-indexes the knowledge base.
              Slower (many minutes) but produces the highest-quality retrieval.
            </p>
            <p class="text-xs text-indigo-700 mt-2 font-mono">POST /rag/resync</p>
          </div>
        </div>
        <button
          class="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50 flex-shrink-0"
          :disabled="enrichedRunning || rawRunning"
          @click="runEnriched"
        >
          <RefreshCwIcon :size="14" :class="enrichedRunning ? 'animate-spin' : ''" />
          {{ enrichedRunning ? 'Running…' : 'Run Enriched Resync' }}
        </button>
      </div>

      <div
        v-if="enrichedResult || enrichedError"
        class="mt-4 rounded-lg border px-4 py-3 text-xs"
        :class="enrichedError
          ? 'bg-red-50 border-red-200 text-red-700'
          : 'bg-emerald-50 border-emerald-200 text-emerald-800'"
      >
        <p v-if="enrichedError" class="font-medium">{{ enrichedError }}</p>
        <template v-else-if="enrichedResult">
          <p class="font-medium">{{ enrichedResult.message }}</p>
          <p class="mt-1 text-emerald-700">
            {{ enrichedResult.documents_indexed }} documents indexed •
            {{ enrichedResult.document_count }} vectors total
          </p>
        </template>
      </div>
    </div>

    <!-- Raw resync -->
    <div class="bg-white rounded-xl border border-slate-200 p-5">
      <div class="flex items-start justify-between gap-4">
        <div class="flex items-start gap-3">
          <div class="w-10 h-10 rounded-lg bg-slate-100 flex items-center justify-center flex-shrink-0">
            <DownloadIcon :size="18" class="text-slate-600" />
          </div>
          <div>
            <p class="text-sm font-semibold text-slate-900">Raw Resync (Data Ingest Only)</p>
            <p class="text-xs text-slate-500 mt-1 leading-relaxed">
              Re-pulls the source data and re-indexes the raw parser output, skipping
              the LLM enrichment step. Fast — useful for quick refreshes — but the
              indexed text is the original table-heavy markdown.
            </p>
            <p class="text-xs text-slate-600 mt-2 font-mono">POST /rag/resync-raw</p>
          </div>
        </div>
        <button
          class="flex items-center gap-2 px-4 py-2 bg-slate-900 hover:bg-slate-700 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50 flex-shrink-0"
          :disabled="enrichedRunning || rawRunning"
          @click="runRaw"
        >
          <RefreshCwIcon :size="14" :class="rawRunning ? 'animate-spin' : ''" />
          {{ rawRunning ? 'Running…' : 'Run Raw Resync' }}
        </button>
      </div>

      <div
        v-if="rawResult || rawError"
        class="mt-4 rounded-lg border px-4 py-3 text-xs"
        :class="rawError
          ? 'bg-red-50 border-red-200 text-red-700'
          : 'bg-emerald-50 border-emerald-200 text-emerald-800'"
      >
        <p v-if="rawError" class="font-medium">{{ rawError }}</p>
        <template v-else-if="rawResult">
          <p class="font-medium">{{ rawResult.message }}</p>
          <p class="mt-1 text-emerald-700">
            {{ rawResult.documents_indexed }} documents indexed •
            {{ rawResult.document_count }} vectors total
          </p>
        </template>
      </div>
    </div>

    <p class="text-xs text-slate-400">
      Both endpoints are blocking — keep this page open while a resync is running.
    </p>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { resyncEnriched, resyncRaw, type ResyncResponse } from '@/api/rag'
import { RefreshCw as RefreshCwIcon, Sparkles as SparklesIcon, Download as DownloadIcon } from 'lucide-vue-next'

const enrichedRunning = ref(false)
const rawRunning = ref(false)
const enrichedResult = ref<ResyncResponse | null>(null)
const rawResult = ref<ResyncResponse | null>(null)
const enrichedError = ref('')
const rawError = ref('')

async function runEnriched() {
  enrichedRunning.value = true
  enrichedResult.value = null
  enrichedError.value = ''
  try {
    enrichedResult.value = await resyncEnriched()
  } catch (e: any) {
    enrichedError.value =
      e?.response?.data?.detail ?? e?.message ?? 'Enriched resync failed — is the RAG API running on :8000?'
  } finally {
    enrichedRunning.value = false
  }
}

async function runRaw() {
  rawRunning.value = true
  rawResult.value = null
  rawError.value = ''
  try {
    rawResult.value = await resyncRaw()
  } catch (e: any) {
    rawError.value =
      e?.response?.data?.detail ?? e?.message ?? 'Raw resync failed — is the RAG API running on :8000?'
  } finally {
    rawRunning.value = false
  }
}
</script>
