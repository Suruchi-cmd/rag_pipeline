<template>
  <Teleport to="body">
    <Transition name="panel">
      <div
        v-if="activeCalls.length > 0"
        class="fixed bottom-6 right-6 z-40 w-80 rounded-2xl overflow-hidden shadow-2xl ring-1 ring-white/10"
      >
        <!-- Header -->
        <div class="bg-slate-900 px-4 py-3 flex items-center justify-between">
          <div class="flex items-center gap-2.5">
            <!-- Pulsing dot -->
            <span class="relative flex h-2.5 w-2.5">
              <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
              <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-400" />
            </span>
            <span class="text-sm font-semibold text-white">Live Calls</span>
            <span class="text-xs font-bold bg-green-500 text-white px-1.5 py-0.5 rounded-full">
              {{ activeCalls.length }}
            </span>
          </div>
          <PhoneCallIcon :size="16" class="text-slate-400" />
        </div>

        <!-- Call rows -->
        <ul class="bg-slate-800 divide-y divide-slate-700/50">
          <li
            v-for="call in activeCalls"
            :key="call.call_sid"
            class="flex items-center gap-3 px-4 py-3 hover:bg-slate-700/50 transition-colors cursor-pointer"
            @click="call.call_id && $router.push(`/calls/${call.call_id}`)"
          >
            <!-- Avatar -->
            <div class="w-9 h-9 rounded-full bg-indigo-600/30 border border-indigo-500/30 flex items-center justify-center flex-shrink-0">
              <UserIcon :size="16" class="text-indigo-400" />
            </div>

            <div class="flex-1 min-w-0">
              <p class="text-sm font-semibold text-white truncate">{{ call.phone_number }}</p>
              <p class="text-xs text-slate-400 mt-0.5">In progress</p>
            </div>

            <!-- Elapsed timer -->
            <div class="text-right flex-shrink-0">
              <p class="text-sm font-mono font-bold text-green-400">{{ elapsed(call.started_at) }}</p>
            </div>
          </li>
        </ul>

        <!-- Footer -->
        <div class="bg-slate-900/80 px-4 py-2 flex items-center justify-between">
          <span class="text-xs text-slate-500">Updates automatically</span>
          <RouterLink to="/calls?status=active" class="text-xs text-indigo-400 hover:text-indigo-300 transition-colors">
            View all →
          </RouterLink>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { PhoneCall as PhoneCallIcon, User as UserIcon } from 'lucide-vue-next'
import { useLiveCalls } from '@/composables/useLiveCalls'

const { activeCalls } = useLiveCalls()

// Tick every second to update elapsed timers
const tick = ref(0)
let timer: ReturnType<typeof setInterval>
onMounted(() => { timer = setInterval(() => tick.value++, 1_000) })
onUnmounted(() => clearInterval(timer))

function elapsed(since: Date): string {
  // tick.value is accessed to make this reactive to the interval
  void tick.value
  const secs = Math.floor((Date.now() - since.getTime()) / 1_000)
  const m = Math.floor(secs / 60)
  const s = secs % 60
  return `${m}:${String(s).padStart(2, '0')}`
}
</script>

<style scoped>
.panel-enter-active { transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1); }
.panel-leave-active { transition: all 0.25s ease-in; }
.panel-enter-from  { opacity: 0; transform: translateY(24px) scale(0.95); }
.panel-leave-to    { opacity: 0; transform: translateY(24px) scale(0.95); }
</style>
