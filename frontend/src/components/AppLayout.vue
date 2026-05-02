<template>
  <div class="flex h-screen bg-slate-50 overflow-hidden">
    <!-- Sidebar -->
    <aside class="w-60 flex-shrink-0 bg-slate-900 flex flex-col">
      <!-- Brand -->
      <div class="flex items-center justify-center px-5 py-4 border-b border-slate-800">
        <img
          src="https://storage.googleapis.com/aerosports/logo/logo_png_websites.png"
          alt="AeroSports"
          class="h-10 w-auto object-contain"
        />
      </div>

      <!-- Nav -->
      <nav class="flex-1 px-3 py-4 space-y-0.5">
        <p class="text-slate-600 text-xs font-semibold uppercase tracking-wider px-3 mb-2">Menu</p>
        <RouterLink
          v-for="item in navItems"
          :key="item.label"
          :to="item.to"
          class="flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors duration-150"
          :class="isActive(item) ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:bg-slate-800 hover:text-slate-100'"
        >
          <component :is="item.icon" :size="16" />
          {{ item.label }}
          <span
            v-if="item.badge != null && item.badge > 0"
            class="ml-auto text-xs font-semibold px-1.5 py-0.5 rounded-full"
            :class="isActive(item) ? 'bg-indigo-500 text-white' : 'bg-slate-700 text-slate-300'"
          >
            {{ item.badge }}
          </span>
        </RouterLink>
      </nav>

      <!-- Health -->
      <div class="px-5 py-4 border-t border-slate-800">
        <div class="flex items-center gap-2">
          <span
            class="w-2 h-2 rounded-full flex-shrink-0"
            :class="apiOnline ? 'bg-green-400' : 'bg-red-500'"
          />
          <span class="text-xs text-slate-500">
            API {{ apiOnline ? 'online' : 'offline' }}
          </span>
          <span v-if="store.stats?.active_calls" class="ml-auto">
            <span class="relative flex h-2 w-2">
              <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
              <span class="relative inline-flex rounded-full h-2 w-2 bg-green-400" />
            </span>
          </span>
        </div>
      </div>
    </aside>

    <!-- Content -->
    <div class="flex-1 flex flex-col overflow-hidden">
      <!-- Header -->
      <header class="h-14 bg-white border-b border-slate-200 flex items-center justify-between px-6 flex-shrink-0">
        <h1 class="text-slate-900 font-semibold text-sm">{{ pageTitle }}</h1>
        <div class="flex items-center gap-2 text-xs text-slate-400">
          <CalendarIcon :size="14" />
          {{ today }}
        </div>
      </header>

      <!-- Main scrollable area -->
      <main class="flex-1 overflow-y-auto p-6">
        <slot />
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import {
  LayoutDashboard,
  Phone,
  Flag,
  Activity,
  Tag,
  BookOpen,
  FileText,
  Calendar as CalendarIcon,
} from 'lucide-vue-next'
import axios from 'axios'
import { useCallsStore } from '@/stores/calls'

const route = useRoute()
const store = useCallsStore()

const navItems = computed(() => [
  { label: 'Dashboard', to: '/', icon: LayoutDashboard, exact: true, badge: null },
  { label: 'All Calls', to: '/calls', icon: Phone, exact: false, badge: null },
  { label: 'Categories', to: '/categories', icon: Tag, exact: true, badge: null },
  { label: 'Knowledge Base', to: '/knowledge', icon: BookOpen, exact: true, badge: null },
  { label: 'Prompts', to: '/prompts', icon: FileText, exact: true, badge: null },
  {
    label: 'Needs Follow-up',
    to: { path: '/calls', query: { needs_human: 'true' } },
    icon: Flag,
    exact: false,
    badge: store.stats?.needs_human ?? null,
  },
  {
    label: 'Active Calls',
    to: { path: '/calls', query: { status: 'active' } },
    icon: Activity,
    exact: false,
    badge: store.stats?.active_calls ?? null,
  },
])

function isActive(item: (typeof navItems.value)[number]): boolean {
  if (typeof item.to === 'string') {
    return item.exact ? route.path === item.to : route.path.startsWith(item.to)
  }
  const toPath = item.to.path
  if (route.path !== toPath) return false
  const query = item.to.query ?? {}
  return Object.entries(query).every(([k, v]) => route.query[k] === v)
}

const pageTitle = computed(() => (route.meta?.title as string) || 'AeroBot')

const today = computed(() =>
  new Date().toLocaleDateString('en-CA', { weekday: 'short', month: 'short', day: 'numeric' }),
)

const apiOnline = ref(true)

async function checkHealth() {
  try {
    await axios.get('/api/health')
    apiOnline.value = true
  } catch {
    apiOnline.value = false
  }
}

let healthTimer: ReturnType<typeof setInterval>
onMounted(async () => {
  await checkHealth()
  await store.fetchStats()
  healthTimer = setInterval(checkHealth, 30_000)
})
onUnmounted(() => clearInterval(healthTimer))
</script>
