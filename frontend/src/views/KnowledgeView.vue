<template>
  <div class="max-w-3xl space-y-6">
    <!-- Header -->
    <div>
      <h2 class="text-lg font-bold text-slate-900">Knowledge Base</h2>
      <p class="text-sm text-slate-500 mt-0.5">
        Each chunk is embedded as a whole for vector search and retrieved during voice calls. Keep
        chunks focused on a single topic.
      </p>
    </div>

    <!-- Resync card -->
    <div class="bg-indigo-50 border border-indigo-200 rounded-xl p-5">
      <div class="flex items-center justify-between gap-4">
        <div class="min-w-0">
          <p class="text-sm font-semibold text-indigo-900">Resync Vector Index</p>
          <p class="text-xs text-indigo-600 mt-0.5">
            Clears existing vectors and re-embeds all chunks. Runs in the background.
          </p>
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

    <!-- Add chunk -->
    <div class="bg-white rounded-xl border border-slate-200 p-5">
      <p class="text-sm font-semibold text-slate-900 mb-4">Add Chunk</p>
      <form class="space-y-3" @submit.prevent="addChunk">
        <div>
          <label class="block text-xs font-medium text-slate-500 mb-1">Name</label>
          <input
            v-model="newName"
            type="text"
            placeholder="e.g. Birthday Party Packages"
            class="w-full text-sm border border-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            :disabled="adding"
          />
        </div>
        <div>
          <label class="block text-xs font-medium text-slate-500 mb-1">Content</label>
          <textarea
            v-model="newContent"
            placeholder="Paste the full text of this knowledge chunk…"
            rows="5"
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

    <!-- Chunks list -->
    <div class="bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div class="px-5 py-4 border-b border-slate-100 flex items-center justify-between">
        <p class="text-sm font-semibold text-slate-900">Chunks</p>
        <span class="text-xs text-slate-400">{{ chunks.length }} total</span>
      </div>

      <div v-if="loading" class="py-10">
        <Spinner full-page />
      </div>

      <div v-else-if="chunks.length === 0" class="py-12 text-center">
        <BookOpenIcon :size="32" class="text-slate-200 mx-auto mb-3" />
        <p class="text-slate-400 text-sm">No chunks yet — add one above</p>
      </div>

      <ul v-else class="divide-y divide-slate-100">
        <li v-for="chunk in chunks" :key="chunk.id" class="px-5 py-4">
          <!-- Edit mode -->
          <template v-if="editingId === chunk.id">
            <div class="space-y-3">
              <div>
                <label class="block text-xs font-medium text-slate-500 mb-1">Name</label>
                <input
                  v-model="editName"
                  type="text"
                  class="w-full text-sm border border-slate-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-500 mb-1">Content</label>
                <textarea
                  v-model="editContent"
                  rows="7"
                  class="w-full text-sm border border-slate-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-y"
                />
              </div>
              <p v-if="saveError" class="text-xs text-red-600">{{ saveError }}</p>
              <div class="flex items-center justify-end gap-2">
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
                  {{ saving ? 'Saving…' : 'Save' }}
                </button>
              </div>
            </div>
          </template>

          <!-- View mode -->
          <template v-else>
            <div class="flex items-start justify-between gap-3">
              <span class="text-sm font-semibold text-slate-800 leading-tight">
                {{ chunk.name }}
              </span>
              <div class="flex items-center gap-1 flex-shrink-0">
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
                  @click="removeChunk(chunk.id)"
                >
                  <Trash2Icon :size="14" />
                </button>
              </div>
            </div>
            <p
              class="mt-1.5 text-xs text-slate-500 leading-relaxed whitespace-pre-wrap"
              :class="expandedIds.has(chunk.id) ? '' : 'line-clamp-3'"
            >
              {{ chunk.content }}
            </p>
            <button
              v-if="chunk.content.length > 280"
              class="mt-1 text-xs text-indigo-500 hover:text-indigo-700 font-medium"
              @click="toggleExpand(chunk.id)"
            >
              {{ expandedIds.has(chunk.id) ? 'Show less' : 'Show more' }}
            </button>
          </template>
        </li>
      </ul>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { listChunks, createChunk, updateChunk, deleteChunk, resyncKnowledge } from '@/api/knowledge'
import type { KnowledgeChunk } from '@/types'
import Spinner from '@/components/Spinner.vue'
import {
  RefreshCw as RefreshCwIcon,
  Plus as PlusIcon,
  Pencil as PencilIcon,
  Trash2 as Trash2Icon,
  Check as CheckIcon,
  BookOpen as BookOpenIcon,
} from 'lucide-vue-next'

const chunks = ref<KnowledgeChunk[]>([])
const loading = ref(false)

const newName = ref('')
const newContent = ref('')
const adding = ref(false)
const addError = ref('')

const editingId = ref<number | null>(null)
const editName = ref('')
const editContent = ref('')
const saving = ref(false)
const saveError = ref('')

const expandedIds = reactive(new Set<number>())

const resyncing = ref(false)
const resyncResult = ref('')
const resyncError = ref(false)

async function load() {
  loading.value = true
  try {
    chunks.value = await listChunks()
  } finally {
    loading.value = false
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
    chunks.value = [...chunks.value, chunk].sort((a, b) => a.name.localeCompare(b.name))
    newName.value = ''
    newContent.value = ''
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
  } catch {
    resyncResult.value = 'Resync failed — check that the server is running.'
    resyncError.value = true
  } finally {
    resyncing.value = false
  }
}

onMounted(load)
</script>
