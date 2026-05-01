<template>
  <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium" :class="cfg.cls">
    <span class="w-1.5 h-1.5 rounded-full" :class="cfg.dot" />
    {{ cfg.label }}
  </span>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ status: string }>()

const MAP: Record<string, { label: string; cls: string; dot: string }> = {
  active:    { label: 'Active',    cls: 'bg-green-100 text-green-800',  dot: 'bg-green-500' },
  completed: { label: 'Completed', cls: 'bg-slate-100 text-slate-600',  dot: 'bg-slate-400' },
  abandoned: { label: 'Abandoned', cls: 'bg-amber-100 text-amber-800',  dot: 'bg-amber-500' },
}

const cfg = computed(() => MAP[props.status] ?? { label: props.status, cls: 'bg-slate-100 text-slate-600', dot: 'bg-slate-400' })
</script>
