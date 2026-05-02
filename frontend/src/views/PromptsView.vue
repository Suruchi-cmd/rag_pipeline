<template>
  <div class="max-w-5xl space-y-5">
    <!-- Header -->
    <div class="flex items-start justify-between gap-4">
      <div>
        <h2 class="text-lg font-bold text-slate-900">Prompts</h2>
        <p class="text-xs text-slate-500 mt-0.5">
          {{ prompts.length }} prompt{{ prompts.length === 1 ? '' : 's' }} ·
          versioned, swap the active one anytime
        </p>
      </div>
      <button
        class="flex items-center gap-1.5 px-4 py-2 text-sm font-medium rounded-lg transition-colors"
        :class="showAdd
          ? 'bg-slate-100 text-slate-700'
          : 'bg-slate-900 hover:bg-slate-700 text-white'"
        @click="showAdd = !showAdd"
      >
        <PlusIcon :size="14" />
        {{ showAdd ? 'Cancel' : 'New Prompt' }}
      </button>
    </div>

    <!-- New prompt form -->
    <Transition
      enter-active-class="transition-all duration-200 ease-out"
      enter-from-class="opacity-0 -translate-y-2"
      enter-to-class="opacity-100 translate-y-0"
      leave-active-class="transition-all duration-150 ease-in"
      leave-from-class="opacity-100 translate-y-0"
      leave-to-class="opacity-0 -translate-y-2"
    >
      <div v-if="showAdd" class="bg-white rounded-xl border border-slate-200 p-5">
        <p class="text-sm font-semibold text-slate-900 mb-4">New Prompt</p>
        <form class="space-y-3" @submit.prevent="addPrompt">
          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block text-xs font-medium text-slate-500 mb-1">Slug</label>
              <input
                v-model="newSlug"
                type="text"
                placeholder="e.g. greeting"
                class="w-full text-sm border border-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-mono"
                :disabled="adding"
              />
              <p class="text-xs text-slate-400 mt-1">
                lowercase letters, numbers, _ or -
              </p>
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-500 mb-1">Name</label>
              <input
                v-model="newName"
                type="text"
                placeholder="Display name"
                class="w-full text-sm border border-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                :disabled="adding"
              />
            </div>
          </div>
          <div>
            <label class="block text-xs font-medium text-slate-500 mb-1">Description (optional)</label>
            <input
              v-model="newDescription"
              type="text"
              placeholder="What this prompt is for"
              class="w-full text-sm border border-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              :disabled="adding"
            />
          </div>
          <div>
            <label class="block text-xs font-medium text-slate-500 mb-1">Initial content</label>
            <textarea
              v-model="newContent"
              rows="8"
              placeholder="Prompt body…"
              class="w-full text-sm font-mono border border-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-y"
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
              {{ adding ? 'Creating…' : 'Create Prompt' }}
            </button>
          </div>
        </form>
      </div>
    </Transition>

    <!-- Loading -->
    <div v-if="loading" class="py-16">
      <Spinner full-page />
    </div>

    <div v-else-if="prompts.length === 0" class="py-16 text-center bg-white rounded-xl border border-slate-200">
      <FileTextIcon :size="36" class="text-slate-200 mx-auto mb-3" />
      <p class="text-slate-400 text-sm font-medium">No prompts yet — create one above</p>
    </div>

    <template v-else>
      <!-- Tabs -->
      <div class="flex items-center gap-1 border-b border-slate-200 overflow-x-auto">
        <button
          v-for="prompt in prompts"
          :key="prompt.id"
          class="relative px-4 py-2.5 text-sm font-medium whitespace-nowrap transition-colors -mb-px border-b-2"
          :class="activeId === prompt.id
            ? 'text-indigo-600 border-indigo-600'
            : 'text-slate-500 border-transparent hover:text-slate-800'"
          @click="selectPrompt(prompt.id)"
        >
          {{ prompt.name }}
          <span class="ml-1.5 text-xs text-slate-400 font-mono">
            {{ activeLabelFor(prompt) }}
          </span>
        </button>
      </div>

      <!-- Active prompt body -->
      <div v-if="loadingDetail" class="py-12">
        <Spinner full-page />
      </div>

      <div v-else-if="detail" class="space-y-4">
        <!-- Meta + actions -->
        <div class="bg-white rounded-xl border border-slate-200 p-5">
          <div class="flex items-start justify-between gap-3">
            <div class="min-w-0 flex-1">
              <div v-if="!editingMeta">
                <div class="flex items-center gap-2 flex-wrap">
                  <h3 class="text-base font-semibold text-slate-900">{{ detail.name }}</h3>
                  <code class="text-xs px-1.5 py-0.5 bg-slate-100 text-slate-600 rounded">
                    {{ detail.slug }}
                  </code>
                </div>
                <p v-if="detail.description" class="text-xs text-slate-500 mt-1">
                  {{ detail.description }}
                </p>
                <p class="text-xs text-slate-400 mt-1">
                  {{ detail.version_count }} version{{ detail.version_count === 1 ? '' : 's' }} ·
                  updated {{ formatRelative(detail.updated_at) }}
                </p>
              </div>
              <div v-else class="space-y-2">
                <input
                  v-model="metaName"
                  type="text"
                  class="w-full text-sm border border-slate-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
                <input
                  v-model="metaDescription"
                  type="text"
                  placeholder="Description"
                  class="w-full text-sm border border-slate-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
            </div>
            <div class="flex items-center gap-1 flex-shrink-0">
              <template v-if="!editingMeta">
                <button
                  class="p-1.5 rounded-lg text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 transition-colors"
                  title="Edit name & description"
                  @click="startMetaEdit"
                >
                  <PencilIcon :size="14" />
                </button>
                <div v-if="confirmDeletePrompt" class="flex items-center gap-1.5">
                  <span class="text-xs text-red-600 font-medium">Delete prompt?</span>
                  <button
                    class="px-2.5 py-1 text-xs font-semibold bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
                    @click="removePrompt"
                  >
                    Yes
                  </button>
                  <button
                    class="px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
                    @click="confirmDeletePrompt = false"
                  >
                    No
                  </button>
                </div>
                <button
                  v-else
                  class="p-1.5 rounded-lg text-slate-400 hover:text-red-500 hover:bg-red-50 transition-colors"
                  title="Delete prompt"
                  @click="confirmDeletePrompt = true"
                >
                  <Trash2Icon :size="14" />
                </button>
              </template>
              <template v-else>
                <button
                  class="px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
                  @click="editingMeta = false"
                >
                  Cancel
                </button>
                <button
                  class="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition-colors disabled:opacity-50"
                  :disabled="savingMeta || !metaName.trim()"
                  @click="saveMetaEdit"
                >
                  <CheckIcon :size="13" />
                  {{ savingMeta ? 'Saving…' : 'Save' }}
                </button>
              </template>
            </div>
          </div>
        </div>

        <!-- Versions list -->
        <div class="space-y-3">
          <div class="flex items-center justify-between">
            <p class="text-xs font-semibold text-slate-500 uppercase tracking-wide">Versions</p>
            <button
              class="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium border border-slate-200 text-slate-700 hover:bg-slate-50 rounded-lg transition-colors"
              @click="startNewVersion"
            >
              <PlusIcon :size="13" />
              New version
            </button>
          </div>

          <!-- New version form -->
          <div
            v-if="showNewVersion"
            class="bg-white rounded-xl border border-indigo-300 ring-2 ring-indigo-100 p-5 space-y-3"
          >
            <div class="flex items-center gap-2">
              <PlusIcon :size="13" class="text-indigo-500" />
              <p class="text-xs font-semibold text-indigo-700 uppercase tracking-wide">
                New version (will be v{{ nextVersionNo }})
              </p>
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-500 mb-1">
                Label (optional)
              </label>
              <input
                v-model="newVersionLabel"
                type="text"
                placeholder="e.g. v1_rajan, baseline, experiment-a"
                class="w-full text-sm border border-slate-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>
            <textarea
              v-model="newVersionContent"
              rows="14"
              class="w-full text-sm font-mono border border-slate-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-y"
              placeholder="Prompt body for this version…"
            />
            <label class="flex items-center gap-2 text-xs text-slate-600 select-none">
              <input
                v-model="newVersionActivate"
                type="checkbox"
                class="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
              />
              Make this the active version
            </label>
            <p v-if="newVersionError" class="text-xs text-red-600">{{ newVersionError }}</p>
            <div class="flex items-center justify-end gap-2 pt-1">
              <button
                class="px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
                @click="showNewVersion = false"
              >
                Cancel
              </button>
              <button
                class="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition-colors disabled:opacity-50"
                :disabled="savingNewVersion || !newVersionContent.trim()"
                @click="saveNewVersion"
              >
                <CheckIcon :size="13" />
                {{ savingNewVersion ? 'Saving…' : 'Save version' }}
              </button>
            </div>
          </div>

          <div
            v-for="version in detail.versions"
            :key="version.id"
            class="bg-white rounded-xl border transition-colors"
            :class="version.is_active
              ? 'border-emerald-300 ring-2 ring-emerald-100'
              : (editingVersionId === version.id ? 'border-indigo-300 ring-2 ring-indigo-100' : 'border-slate-200')"
          >
            <div class="p-5 space-y-3">
              <div class="flex items-start justify-between gap-3">
                <div class="flex items-center gap-2 flex-wrap min-w-0">
                  <span class="text-sm font-semibold text-slate-800">
                    Version {{ version.version_no }}
                  </span>
                  <span
                    v-if="version.label"
                    class="inline-flex items-center text-xs font-mono px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 border border-slate-200"
                  >
                    {{ version.label }}
                  </span>
                  <span
                    v-if="version.is_active"
                    class="inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-700"
                  >
                    <CheckCircleIcon :size="11" />
                    Active
                  </span>
                  <span class="text-xs text-slate-400">
                    Updated {{ formatRelative(version.updated_at) }}
                  </span>
                </div>
                <div class="flex items-center gap-1 flex-shrink-0">
                  <template v-if="editingVersionId === version.id">
                    <button
                      class="px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
                      @click="cancelVersionEdit"
                    >
                      Cancel
                    </button>
                    <button
                      class="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition-colors disabled:opacity-50"
                      :disabled="savingVersion || !versionEditContent.trim()"
                      @click="saveVersionEdit(version.id)"
                    >
                      <CheckIcon :size="13" />
                      {{ savingVersion ? 'Saving…' : 'Save' }}
                    </button>
                  </template>
                  <template v-else>
                    <button
                      v-if="!version.is_active"
                      class="flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-emerald-700 hover:bg-emerald-50 rounded-lg transition-colors"
                      title="Make active"
                      @click="activate(version.id)"
                    >
                      <CheckCircleIcon :size="13" />
                      Activate
                    </button>
                    <button
                      class="p-1.5 rounded-lg text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 transition-colors"
                      title="Edit content"
                      @click="startVersionEdit(version)"
                    >
                      <PencilIcon :size="14" />
                    </button>
                    <div v-if="confirmDeleteVersionId === version.id" class="flex items-center gap-1.5">
                      <span class="text-xs text-red-600 font-medium">Delete?</span>
                      <button
                        class="px-2.5 py-1 text-xs font-semibold bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
                        @click="removeVersion(version.id)"
                      >
                        Yes
                      </button>
                      <button
                        class="px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
                        @click="confirmDeleteVersionId = null"
                      >
                        No
                      </button>
                    </div>
                    <button
                      v-else-if="!version.is_active && detail.versions.length > 1"
                      class="p-1.5 rounded-lg text-slate-400 hover:text-red-500 hover:bg-red-50 transition-colors"
                      title="Delete version"
                      @click="confirmDeleteVersionId = version.id"
                    >
                      <Trash2Icon :size="14" />
                    </button>
                  </template>
                </div>
              </div>

              <template v-if="editingVersionId === version.id">
                <div>
                  <label class="block text-xs font-medium text-slate-500 mb-1">
                    Label (optional)
                  </label>
                  <input
                    v-model="versionEditLabel"
                    type="text"
                    placeholder="e.g. v1_rajan"
                    class="w-full text-sm border border-slate-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                </div>
                <textarea
                  v-model="versionEditContent"
                  rows="14"
                  class="w-full text-sm font-mono border border-slate-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-y"
                />
              </template>
              <pre
                v-else
                class="text-xs font-mono text-slate-700 whitespace-pre-wrap leading-relaxed bg-slate-50 border border-slate-100 rounded-lg p-3 max-h-72 overflow-auto"
              >{{ version.content }}</pre>

              <p
                v-if="editingVersionId === version.id && versionEditError"
                class="text-xs text-red-600"
              >
                {{ versionEditError }}
              </p>
            </div>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import {
  Plus as PlusIcon,
  Pencil as PencilIcon,
  Trash2 as Trash2Icon,
  Check as CheckIcon,
  CheckCircle2 as CheckCircleIcon,
  FileText as FileTextIcon,
} from 'lucide-vue-next'
import Spinner from '@/components/Spinner.vue'
import {
  listPrompts,
  getPrompt,
  createPrompt,
  updatePromptMeta,
  deletePrompt,
  createVersion,
  updateVersion,
  activateVersion,
  deleteVersion,
} from '@/api/prompts'
import type { PromptDetail, PromptSummary } from '@/types'

const prompts = ref<PromptSummary[]>([])
const loading = ref(false)
const activeId = ref<number | null>(null)
const detail = ref<PromptDetail | null>(null)
const loadingDetail = ref(false)

const showAdd = ref(false)
const newSlug = ref('')
const newName = ref('')
const newDescription = ref('')
const newContent = ref('')
const adding = ref(false)
const addError = ref('')

const editingMeta = ref(false)
const metaName = ref('')
const metaDescription = ref('')
const savingMeta = ref(false)
const confirmDeletePrompt = ref(false)

const editingVersionId = ref<number | null>(null)
const versionEditContent = ref('')
const versionEditLabel = ref('')
const versionEditError = ref('')
const savingVersion = ref(false)
const confirmDeleteVersionId = ref<number | null>(null)

const showNewVersion = ref(false)
const newVersionContent = ref('')
const newVersionLabel = ref('')
const newVersionActivate = ref(false)
const newVersionError = ref('')
const savingNewVersion = ref(false)

const nextVersionNo = computed(() => {
  if (!detail.value || detail.value.versions.length === 0) return 1
  return Math.max(...detail.value.versions.map((v) => v.version_no)) + 1
})

function activeLabelFor(prompt: PromptSummary): string {
  if (prompt.active_version_label) return prompt.active_version_label
  if (prompt.active_version_no != null) return `v${prompt.active_version_no}`
  return '–'
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

async function load() {
  loading.value = true
  try {
    prompts.value = await listPrompts()
    const first = prompts.value[0]
    if (first && activeId.value == null) {
      await selectPrompt(first.id)
    }
  } finally {
    loading.value = false
  }
}

async function selectPrompt(id: number) {
  activeId.value = id
  resetTransientState()
  loadingDetail.value = true
  try {
    detail.value = await getPrompt(id)
  } finally {
    loadingDetail.value = false
  }
}

function resetTransientState() {
  editingMeta.value = false
  confirmDeletePrompt.value = false
  editingVersionId.value = null
  versionEditError.value = ''
  confirmDeleteVersionId.value = null
  showNewVersion.value = false
  newVersionContent.value = ''
  newVersionActivate.value = false
  newVersionError.value = ''
}

async function refreshDetail() {
  if (activeId.value == null) return
  detail.value = await getPrompt(activeId.value)
  const idx = prompts.value.findIndex((p) => p.id === activeId.value)
  if (idx !== -1 && detail.value) {
    prompts.value[idx] = {
      id: detail.value.id,
      slug: detail.value.slug,
      name: detail.value.name,
      description: detail.value.description,
      created_at: detail.value.created_at,
      updated_at: detail.value.updated_at,
      active_version_id: detail.value.active_version_id,
      active_version_no: detail.value.active_version_no,
      active_version_label: detail.value.active_version_label,
      version_count: detail.value.version_count,
    }
  }
}

async function addPrompt() {
  addError.value = ''
  if (!newSlug.value.trim()) { addError.value = 'Slug is required'; return }
  if (!newName.value.trim()) { addError.value = 'Name is required'; return }
  if (!newContent.value.trim()) { addError.value = 'Initial content is required'; return }
  adding.value = true
  try {
    const created = await createPrompt({
      slug: newSlug.value.trim().toLowerCase(),
      name: newName.value.trim(),
      description: newDescription.value.trim() || null,
      content: newContent.value,
    })
    prompts.value = [...prompts.value, created]
    newSlug.value = ''
    newName.value = ''
    newDescription.value = ''
    newContent.value = ''
    showAdd.value = false
    await selectPrompt(created.id)
  } catch (e: any) {
    addError.value = e?.response?.data?.detail ?? 'Failed to create prompt'
  } finally {
    adding.value = false
  }
}

function startMetaEdit() {
  if (!detail.value) return
  metaName.value = detail.value.name
  metaDescription.value = detail.value.description ?? ''
  editingMeta.value = true
}

async function saveMetaEdit() {
  if (!detail.value) return
  savingMeta.value = true
  try {
    await updatePromptMeta(detail.value.id, {
      name: metaName.value.trim(),
      description: metaDescription.value.trim() || null,
    })
    editingMeta.value = false
    await refreshDetail()
  } finally {
    savingMeta.value = false
  }
}

async function removePrompt() {
  if (!detail.value) return
  const id = detail.value.id
  try {
    await deletePrompt(id)
    prompts.value = prompts.value.filter((p) => p.id !== id)
    detail.value = null
    activeId.value = null
    confirmDeletePrompt.value = false
    const next = prompts.value[0]
    if (next) await selectPrompt(next.id)
  } catch {
    // ignore
  }
}

function startVersionEdit(version: { id: number; content: string; label: string | null }) {
  editingVersionId.value = version.id
  versionEditContent.value = version.content
  versionEditLabel.value = version.label ?? ''
  versionEditError.value = ''
  confirmDeleteVersionId.value = null
}

function cancelVersionEdit() {
  editingVersionId.value = null
  versionEditError.value = ''
}

async function saveVersionEdit(versionId: number) {
  if (!detail.value) return
  if (!versionEditContent.value.trim()) {
    versionEditError.value = 'Content is required'
    return
  }
  savingVersion.value = true
  versionEditError.value = ''
  try {
    await updateVersion(detail.value.id, versionId, {
      content: versionEditContent.value,
      label: versionEditLabel.value.trim() || null,
    })
    editingVersionId.value = null
    await refreshDetail()
  } catch (e: any) {
    versionEditError.value = e?.response?.data?.detail ?? 'Failed to save'
  } finally {
    savingVersion.value = false
  }
}

async function activate(versionId: number) {
  if (!detail.value) return
  try {
    await activateVersion(detail.value.id, versionId)
    await refreshDetail()
  } catch {
    // ignore
  }
}

async function removeVersion(versionId: number) {
  if (!detail.value) return
  try {
    await deleteVersion(detail.value.id, versionId)
    confirmDeleteVersionId.value = null
    await refreshDetail()
  } catch (e: any) {
    alert(e?.response?.data?.detail ?? 'Failed to delete version')
    confirmDeleteVersionId.value = null
  }
}

function startNewVersion() {
  if (!detail.value) return
  const active = detail.value.versions.find((v) => v.is_active)
  newVersionContent.value = active?.content ?? ''
  newVersionLabel.value = ''
  newVersionActivate.value = false
  newVersionError.value = ''
  showNewVersion.value = true
}

async function saveNewVersion() {
  if (!detail.value) return
  if (!newVersionContent.value.trim()) {
    newVersionError.value = 'Content is required'
    return
  }
  savingNewVersion.value = true
  newVersionError.value = ''
  try {
    await createVersion(detail.value.id, {
      content: newVersionContent.value,
      activate: newVersionActivate.value,
      label: newVersionLabel.value.trim() || null,
    })
    showNewVersion.value = false
    newVersionContent.value = ''
    newVersionLabel.value = ''
    newVersionActivate.value = false
    await refreshDetail()
  } catch (e: any) {
    newVersionError.value = e?.response?.data?.detail ?? 'Failed to create version'
  } finally {
    savingNewVersion.value = false
  }
}

onMounted(load)
</script>
