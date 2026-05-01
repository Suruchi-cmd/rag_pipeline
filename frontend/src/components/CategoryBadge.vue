<template>
  <span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium" :class="color">
    {{ name }}
  </span>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ name: string }>()

const PALETTES = [
  'bg-pink-100 text-pink-800',
  'bg-amber-100 text-amber-800',
  'bg-blue-100 text-blue-800',
  'bg-green-100 text-green-800',
  'bg-purple-100 text-purple-800',
  'bg-red-100 text-red-800',
  'bg-teal-100 text-teal-800',
  'bg-orange-100 text-orange-800',
  'bg-cyan-100 text-cyan-800',
  'bg-lime-100 text-lime-800',
]

// Deterministic color from name so the same category always gets the same color
const color = computed(() => {
  let hash = 0
  for (const ch of props.name) hash = (hash * 31 + ch.charCodeAt(0)) & 0xffffffff
  return PALETTES[Math.abs(hash) % PALETTES.length]
})
</script>
