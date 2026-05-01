<template>
  <div class="max-w-4xl space-y-5">

    <!-- Header -->
    <div class="flex items-start justify-between gap-4">
      <div>
        <h2 class="text-lg font-bold text-slate-900">Knowledge Base</h2>
        <p class="text-xs text-slate-500 mt-0.5">
          {{ chunks.length }} chunk{{ chunks.length !== 1 ? 's' : '' }} ·
          {{ totalWords.toLocaleString() }} words ·
          {{ totalChars.toLocaleString() }} chars
        </p>
      </div>
      <div class="flex items-center gap-2">
        <label
          class="flex items-center gap-1.5 px-3 py-2 text-xs font-medium border border-slate-200 text-slate-600 hover:bg-slate-50 rounded-lg cursor-pointer transition-colors"
          title="Import chunks from JSON"
        >
          <UploadIcon :size="13" />
          Import
          <input type="file" accept=".json" class="hidden" @change="importJson" />
        </label>
        <button
          class="flex items-center gap-1.5 px-3 py-2 text-xs font-medium border border-slate-200 text-slate-600 hover:bg-slate-50 rounded-lg transition-colors"
          title="Export all chunks as JSON"
          @click="exportJson"
        >
          <DownloadIcon :size="13" />
          Export
        </button>
        <button
          class="flex items-center gap-1.5 px-4 py-2 text-sm font-medium rounded-lg transition-colors"
          :class="showAdd
            ? 'bg-slate-100 text-slate-700'
            : 'bg-slate-900 hover:bg-slate-700 text-white'"
          @click="showAdd = !showAdd"
        >
          <PlusIcon :size="14" />
          {{ showAdd ? 'Cancel' : 'Add Chunk' }}
        </button>
      </div>
    </div>

    <!-- Resync card -->
    <div class="bg-indigo-50 border border-indigo-200 rounded-xl p-4">
      <div class="flex items-center justify-between gap-4">
        <div class="flex items-center gap-3 min-w-0">
          <div
            class="w-2.5 h-2.5 rounded-full flex-shrink-0 transition-colors"
            :class="vectorReady ? 'bg-green-500' : 'bg-amber-400'"
          />
          <div class="min-w-0">
            <p class="text-sm font-semibold text-indigo-900">
              Vector Index
              <span class="font-normal ml-1" :class="vectorReady ? 'text-green-700' : 'text-amber-700'">
                {{ vectorReady ? '· Ready' : '· Not ready' }}
              </span>
            </p>
            <p class="text-xs text-indigo-600 mt-0.5">
              Clears existing vectors and re-embeds all chunks. Runs in the background.
            </p>
          </div>
        </div>
        <button
          class="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50 flex-shrink-0"
          :disabled="resyncing"
          @click="doResync"
        >
          <RefreshCwIcon :size="14" :class="resyncing ? 'animate-spin' : ''" />
          {{ resyncing ? 'Queuing…' : 'Resync' }}
        </button>
      </div>
      <p
        v-if="resyncResult"
        class="mt-3 text-xs font-medium px-3 py-2 rounded-lg"
        :class="resyncError ? 'bg-red-100 text-red-700' : 'bg-indigo-100 text-indigo-800'"
      >
        {{ resyncResult }}
      </p>
    </div>

    <!-- Add chunk form (collapsible) -->
    <Transition
      enter-active-class="transition-all duration-200 ease-out"
      enter-from-class="opacity-0 -translate-y-2"
      enter-to-class="opacity-100 translate-y-0"
      leave-active-class="transition-all duration-150 ease-in"
      leave-from-class="opacity-100 translate-y-0"
      leave-to-class="opacity-0 -translate-y-2"
    >
      <div v-if="showAdd" class="bg-white rounded-xl border border-slate-200 p-5">
        <p class="text-sm font-semibold text-slate-900 mb-4">New Chunk</p>
        <form class="space-y-3" @submit.prevent="addChunk">
          <div>
            <label class="block text-xs font-medium text-slate-500 mb-1">Name</label>
            <input
              v-model="newName"
              ref="newNameInput"
              type="text"
              placeholder="e.g. Birthday Party Packages"
              class="w-full text-sm border border-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              :disabled="adding"
            />
          </div>
          <div>
            <div class="flex items-center justify-between mb-1">
              <label class="text-xs font-medium text-slate-500">Content</label>
              <span class="text-xs text-slate-400">
                {{ wordCount(newContent) }} words · {{ newContent.length }} chars
              </span>
            </div>
            <textarea
              v-model="newContent"
              placeholder="Paste the full text of this knowledge chunk…"
              rows="6"
              class="w-full text-sm border border-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-y"
              :disabled="adding"
            />
          </div>
          <p v-if="addError" class="text-xs text-red-600">{{ addError }}</p>
          <div class="flex justify-end pt-1">
            <button
              type="submit"
              class="flex items-center gap-1.5 px-4 py-2 bg-slate-900 hover:bg-slate-700 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50"
              :disabled="adding"
            >
              <PlusIcon :size="14" />
              {{ adding ? 'Adding…' : 'Add Chunk' }}
            </button>
          </div>
        </form>
      </div>
    </Transition>

    <!-- Search + sort bar -->
    <div class="flex items-center gap-3">
      <div class="relative flex-1">
        <SearchIcon :size="14" class="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
        <input
          v-model="search"
          type="text"
          placeholder="Search chunks by name or content…"
          class="w-full text-sm border border-slate-200 rounded-lg pl-8 pr-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 bg-white"
        />
        <button
          v-if="search"
          class="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
          @click="search = ''"
        >
          <XIcon :size="13" />
        </button>
      </div>
      <select
        v-model="sortBy"
        class="text-sm border border-slate-200 rounded-lg px-3 py-2 text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 bg-white"
      >
        <option value="name-asc">Name A→Z</option>
        <option value="name-desc">Name Z→A</option>
        <option value="newest">Newest first</option>
        <option value="oldest">Oldest first</option>
        <option value="largest">Largest first</option>
        <option value="smallest">Smallest first</option>
      </select>
    </div>

    <!-- Chunks list -->
    <div v-if="loading" class="py-16">
      <Spinner full-page />
    </div>

    <div v-else-if="filteredSorted.length === 0" class="py-16 text-center bg-white rounded-xl border border-slate-200">
      <BookOpenIcon :size="36" class="text-slate-200 mx-auto mb-3" />
      <p class="text-slate-400 text-sm font-medium">
        {{ search ? 'No chunks match your search' : 'No chunks yet — add one above' }}
      </p>
      <button v-if="search" class="mt-2 text-xs text-indigo-500 hover:text-indigo-700" @click="search = ''">
        Clear search
      </button>
    </div>

    <div v-else class="space-y-3">
      <!-- Match count when filtering -->
      <p v-if="search" class="text-xs text-slate-500 px-1">
        {{ filteredSorted.length }} of {{ chunks.length }} chunks match
      </p>

      <div
        v-for="chunk in filteredSorted"
        :key="chunk.id"
        class="bg-white rounded-xl border transition-all duration-150"
        :class="editingId === chunk.id ? 'border-indigo-300 ring-2 ring-indigo-100' : 'border-slate-200'"
      >
        <!-- Edit mode -->
        <template v-if="editingId === chunk.id">
          <div class="p-5 space-y-3">
            <div class="flex items-center gap-2 mb-1">
              <PencilIcon :size="13" class="text-indigo-500" />
              <p class="text-xs font-semibold text-indigo-700 uppercase tracking-wide">Editing</p>
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-500 mb-1">Name</label>
              <input
                v-model="editName"
                type="text"
                class="w-full text-sm border border-slate-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>
            <div>
              <div class="flex items-center justify-between mb-1">
                <label class="text-xs font-medium text-slate-500">Content</label>
                <span class="text-xs text-slate-400">
                  {{ wordCount(editContent) }} words · {{ editContent.length }} chars
                </span>
              </div>
              <textarea
                v-model="editContent"
                rows="8"
                class="w-full text-sm border border-slate-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-y"
              />
            </div>
            <p v-if="saveError" class="text-xs text-red-600">{{ saveError }}</p>
            <div class="flex items-center justify-end gap-2 pt-1">
              <button
                class="px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
                @click="cancelEdit"
              >
                Cancel
              </button>
              <button
                class="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition-colors disabled:opacity-50"
                :disabled="saving || !editName.trim() || !editContent.trim()"
                @click="saveEdit(chunk.id)"
              >
                <CheckIcon :size="13" />
                {{ saving ? 'Saving…' : 'Save changes' }}
              </button>
            </div>
          </div>
        </template>

        <!-- View mode -->
        <template v-else>
          <div class="p-5">
            <div class="flex items-start justify-between gap-3">
              <div class="min-w-0 flex-1">
                <div class="flex items-center gap-2 flex-wrap">
                  <span class="text-sm font-semibold text-slate-800 leading-tight">
                    {{ chunk.name }}
                  </span>
                  <span class="text-xs text-slate-400 bg-slate-50 border border-slate-100 rounded px-1.5 py-0.5">
                    {{ wordCount(chunk.content) }}w · {{ chunk.content.length }}c
                  </span>
                </div>
                <p class="text-xs text-slate-400 mt-0.5">
                  Updated {{ formatRelative(chunk.updated_at) }}
                </p>
              </div>

              <!-- Delete confirm state -->
              <div v-if="confirmDeleteId === chunk.id" class="flex items-center gap-1.5 flex-shrink-0">
                <span class="text-xs text-red-600 font-medium">Delete?</span>
                <button
                  class="px-2.5 py-1 text-xs font-semibold bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
                  @click="removeChunk(chunk.id)"
                >
                  Yes
                </button>
                <button
                  class="px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
                  @click="confirmDeleteId = null"
                >
                  No
                </button>
              </div>

              <!-- Normal action buttons -->
              <div v-else class="flex items-center gap-1 flex-shrink-0">
                <button
                  class="p-1.5 rounded-lg text-slate-300 hover:text-indigo-600 hover:bg-indigo-50 transition-colors"
                  title="Edit chunk"
                  @click="startEdit(chunk)"
                >
                  <PencilIcon :size="14" />
                </button>
                <button
                  class="p-1.5 rounded-lg text-slate-300 hover:text-red-500 hover:bg-red-50 transition-colors"
                  title="Delete chunk"
                  @click="confirmDeleteId = chunk.id"
                >
                  <Trash2Icon :size="14" />
                </button>
              </div>
            </div>

            <!-- Content preview -->
            <p
              class="mt-2.5 text-xs text-slate-500 leading-relaxed whitespace-pre-wrap"
              :class="expandedIds.has(chunk.id) ? '' : 'line-clamp-3'"
            >
              {{ chunk.content }}
            </p>
            <button
              v-if="chunk.content.length > 240"
              class="mt-1.5 text-xs text-indigo-500 hover:text-indigo-700 font-medium"
              @click="toggleExpand(chunk.id)"
            >
              {{ expandedIds.has(chunk.id) ? 'Show less ↑' : 'Show more ↓' }}
            </button>
          </div>
        </template>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, nextTick, onMounted } from 'vue'
import { listChunks, createChunk, updateChunk, deleteChunk, resyncKnowledge } from '@/api/knowledge'
import type { KnowledgeChunk } from '@/types'
import Spinner from '@/components/Spinner.vue'
import axios from 'axios'
import {
  RefreshCw as RefreshCwIcon,
  Plus as PlusIcon,
  Pencil as PencilIcon,
  Trash2 as Trash2Icon,
  Check as CheckIcon,
  BookOpen as BookOpenIcon,
  Search as SearchIcon,
  X as XIcon,
  Upload as UploadIcon,
  Download as DownloadIcon,
} from 'lucide-vue-next'

// ── State ──────────────────────────────────────────────────────────────────────

const chunks = ref<KnowledgeChunk[]>([])
const loading = ref(false)

const showAdd = ref(false)
const newName = ref('')
const newContent = ref('')
const adding = ref(false)
const addError = ref('')
const newNameInput = ref<HTMLInputElement | null>(null)

const editingId = ref<number | null>(null)
const editName = ref('')
const editContent = ref('')
const saving = ref(false)
const saveError = ref('')

const confirmDeleteId = ref<number | null>(null)
const expandedIds = reactive(new Set<number>())

const search = ref('')
const sortBy = ref<'name-asc' | 'name-desc' | 'newest' | 'oldest' | 'largest' | 'smallest'>('name-asc')

const resyncing = ref(false)
const resyncResult = ref('')
const resyncError = ref(false)
const vectorReady = ref(false)

// ── Computed ───────────────────────────────────────────────────────────────────

const totalWords = computed(() => chunks.value.reduce((s, c) => s + wordCount(c.content), 0))
const totalChars = computed(() => chunks.value.reduce((s, c) => s + c.content.length, 0))

const filteredSorted = computed(() => {
  const q = search.value.toLowerCase().trim()
  let list = q
    ? chunks.value.filter(
        (c) => c.name.toLowerCase().includes(q) || c.content.toLowerCase().includes(q),
      )
    : [...chunks.value]

  switch (sortBy.value) {
    case 'name-asc':
      list.sort((a, b) => a.name.localeCompare(b.name))
      break
    case 'name-desc':
      list.sort((a, b) => b.name.localeCompare(a.name))
      break
    case 'newest':
      list.sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
      break
    case 'oldest':
      list.sort((a, b) => new Date(a.updated_at).getTime() - new Date(b.updated_at).getTime())
      break
    case 'largest':
      list.sort((a, b) => b.content.length - a.content.length)
      break
    case 'smallest':
      list.sort((a, b) => a.content.length - b.content.length)
      break
  }
  return list
})

// ── Helpers ────────────────────────────────────────────────────────────────────

function wordCount(text: string): number {
  return text.trim() ? text.trim().split(/\s+/).length : 0
}

function formatRelative(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60_000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  const days = Math.floor(hrs / 24)
  if (days < 30) return `${days}d ago`
  return new Date(iso).toLocaleDateString()
}

// ── Actions ────────────────────────────────────────────────────────────────────

async function load() {
  loading.value = true
  try {
    chunks.value = await listChunks()
  } finally {
    loading.value = false
  }
}

async function checkVectorHealth() {
  try {
    const { data } = await axios.get('/api/knowledge/health')
    vectorReady.value = data.ready
  } catch {
    vectorReady.value = false
  }
}

async function addChunk() {
  const name = newName.value.trim()
  const content = newContent.value.trim()
  addError.value = ''
  if (!name && !content) { addError.value = 'Name and content are required'; return }
  if (!name) { addError.value = 'Name is required'; return }
  if (!content) { addError.value = 'Content is required'; return }
  adding.value = true
  try {
    const chunk = await createChunk(name, content)
    chunks.value = [...chunks.value, chunk]
    newName.value = ''
    newContent.value = ''
    showAdd.value = false
  } catch (e: any) {
    addError.value = e?.response?.data?.detail ?? 'Failed to create chunk'
  } finally {
    adding.value = false
  }
}

function startEdit(chunk: KnowledgeChunk) {
  editingId.value = chunk.id
  editName.value = chunk.name
  editContent.value = chunk.content
  saveError.value = ''
  confirmDeleteId.value = null
}

function cancelEdit() {
  editingId.value = null
  saveError.value = ''
}

async function saveEdit(id: number) {
  const name = editName.value.trim()
  const content = editContent.value.trim()
  if (!name || !content) return
  saving.value = true
  saveError.value = ''
  try {
    const updated = await updateChunk(id, name, content)
    const idx = chunks.value.findIndex((c) => c.id === id)
    if (idx !== -1) chunks.value[idx] = updated
    editingId.value = null
  } catch (e: any) {
    saveError.value = e?.response?.data?.detail ?? 'Failed to save changes'
  } finally {
    saving.value = false
  }
}

async function removeChunk(id: number) {
  try {
    await deleteChunk(id)
    chunks.value = chunks.value.filter((c) => c.id !== id)
    expandedIds.delete(id)
    if (editingId.value === id) editingId.value = null
  } catch {
    // ignore
  } finally {
    confirmDeleteId.value = null
  }
}

function toggleExpand(id: number) {
  if (expandedIds.has(id)) expandedIds.delete(id)
  else expandedIds.add(id)
}

async function doResync() {
  resyncing.value = true
  resyncResult.value = ''
  resyncError.value = false
  try {
    const result = await resyncKnowledge()
    resyncResult.value = result.message
    resyncError.value = false
    setTimeout(checkVectorHealth, 3_000)
  } catch {
    resyncResult.value = 'Resync failed — check that the server is running.'
    resyncError.value = true
  } finally {
    resyncing.value = false
  }
}

function exportJson() {
  const data = chunks.value.map(({ name, content }) => ({ name, content }))
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `knowledge-base-${new Date().toISOString().slice(0, 10)}.json`
  a.click()
  URL.revokeObjectURL(url)
}

async function importJson(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (!file) return
  try {
    const text = await file.text()
    const parsed = JSON.parse(text)
    if (!Array.isArray(parsed)) { alert('Invalid format: expected a JSON array'); return }
    const items = parsed.filter(
      (i): i is { name: string; content: string } =>
        typeof i?.name === 'string' && typeof i?.content === 'string',
    )
    if (items.length === 0) { alert('No valid chunks found in file'); return }
    if (!confirm(`Import ${items.length} chunk(s)? Duplicates will be skipped.`)) return
    let added = 0
    for (const item of items) {
      try {
        const chunk = await createChunk(item.name, item.content)
        chunks.value.push(chunk)
        added++
      } catch {
        // skip duplicates
      }
    }
    alert(`Imported ${added} chunk(s).`)
  } catch {
    alert('Failed to parse JSON file')
  } finally {
    ;(event.target as HTMLInputElement).value = ''
  }
}

// Focus name input when add form opens
import { watch } from 'vue'
watch(showAdd, async (val) => {
  if (val) {
    await nextTick()
    newNameInput.value?.focus()
  }
})

onMounted(() => {
  load()
  checkVectorHealth()
})
</script>
