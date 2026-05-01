<template>
  <div class="max-w-2xl space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h2 class="text-lg font-bold text-slate-900">Call Categories</h2>
        <p class="text-sm text-slate-500 mt-0.5">
          Categories are used to classify calls automatically when they end.
          After adding or removing categories, press Resync to reclassify all existing calls.
        </p>
      </div>
    </div>

    <!-- Resync card -->
    <div class="bg-indigo-50 border border-indigo-200 rounded-xl p-5 flex items-center justify-between gap-4">
      <div>
        <p class="text-sm font-semibold text-indigo-900">Resync Classifications</p>
        <p class="text-xs text-indigo-600 mt-0.5">
          Reclassifies all completed and abandoned calls using the current category list.
          {{ resyncResult ? resyncResult : 'Runs in the background — may take a few minutes.' }}
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

    <!-- Add category -->
    <div class="bg-white rounded-xl border border-slate-200 p-5">
      <p class="text-sm font-semibold text-slate-900 mb-3">Add Category</p>
      <form class="flex gap-3" @submit.prevent="addCategory">
        <input
          v-model="newName"
          type="text"
          placeholder="e.g. Laser Tag"
          class="flex-1 text-sm border border-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          :disabled="adding"
        />
        <button
          type="submit"
          class="flex items-center gap-1.5 px-4 py-2 bg-slate-900 hover:bg-slate-700 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50"
          :disabled="adding || !newName.trim()"
        >
          <PlusIcon :size="14" />
          Add
        </button>
      </form>
      <p v-if="addError" class="text-xs text-red-600 mt-2">{{ addError }}</p>
    </div>

    <!-- Categories list -->
    <div class="bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div class="px-5 py-4 border-b border-slate-100 flex items-center justify-between">
        <p class="text-sm font-semibold text-slate-900">Current Categories</p>
        <span class="text-xs text-slate-400">{{ categories.length }} total</span>
      </div>

      <div v-if="loading" class="py-10">
        <Spinner full-page />
      </div>

      <div v-else-if="categories.length === 0" class="py-12 text-center">
        <TagIcon :size="32" class="text-slate-200 mx-auto mb-3" />
        <p class="text-slate-400 text-sm">No categories yet</p>
      </div>

      <ul v-else class="divide-y divide-slate-50">
        <li
          v-for="cat in categories"
          :key="cat.id"
          class="flex items-center justify-between px-5 py-3.5 hover:bg-slate-50 transition-colors"
        >
          <CategoryBadge :name="cat.name" />
          <button
            class="p-1.5 rounded-lg text-slate-300 hover:text-red-500 hover:bg-red-50 transition-colors"
            title="Delete category"
            @click="removeCategory(cat.id)"
          >
            <Trash2Icon :size="15" />
          </button>
        </li>
      </ul>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import {
  listCategories,
  createCategory,
  deleteCategory,
  resyncCategories,
} from '@/api/categories'
import type { Category } from '@/types'
import CategoryBadge from '@/components/CategoryBadge.vue'
import Spinner from '@/components/Spinner.vue'
import { RefreshCw as RefreshCwIcon, Plus as PlusIcon, Tag as TagIcon, Trash2 as Trash2Icon } from 'lucide-vue-next'

const categories = ref<Category[]>([])
const loading = ref(false)
const newName = ref('')
const adding = ref(false)
const addError = ref('')
const resyncing = ref(false)
const resyncResult = ref('')

async function load() {
  loading.value = true
  try {
    categories.value = await listCategories()
  } finally {
    loading.value = false
  }
}

async function addCategory() {
  const name = newName.value.trim()
  if (!name) return
  adding.value = true
  addError.value = ''
  try {
    const cat = await createCategory(name)
    categories.value = [...categories.value, cat].sort((a, b) => a.name.localeCompare(b.name))
    newName.value = ''
  } catch (e: any) {
    addError.value = e?.response?.data?.detail ?? 'Failed to create category'
  } finally {
    adding.value = false
  }
}

async function removeCategory(id: number) {
  try {
    await deleteCategory(id)
    categories.value = categories.value.filter((c) => c.id !== id)
  } catch {
    // ignore
  }
}

async function doResync() {
  resyncing.value = true
  resyncResult.value = ''
  try {
    const result = await resyncCategories()
    resyncResult.value = result.message
  } catch {
    resyncResult.value = 'Resync failed — is the API running?'
  } finally {
    resyncing.value = false
  }
}

onMounted(load)
</script>
