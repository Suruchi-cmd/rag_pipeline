<template>
  <div class="flex items-center justify-between px-1">
    <p class="text-sm text-slate-500">
      Showing {{ from }}–{{ to }}
    </p>
    <div class="flex gap-2">
      <button
        class="px-3 py-1.5 text-sm rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        :disabled="offset === 0"
        @click="$emit('prev')"
      >
        ← Prev
      </button>
      <button
        class="px-3 py-1.5 text-sm rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        :disabled="count < limit"
        @click="$emit('next')"
      >
        Next →
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ offset: number; limit: number; count: number }>()
defineEmits<{ prev: []; next: [] }>()

const from = computed(() => props.count === 0 ? 0 : props.offset + 1)
const to = computed(() => props.offset + props.count)
</script>
