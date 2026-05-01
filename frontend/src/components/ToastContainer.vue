<template>
  <Teleport to="body">
    <div class="fixed top-5 right-5 z-50 flex flex-col gap-3 pointer-events-none" style="max-width: 340px">
      <TransitionGroup
        name="toast"
        tag="div"
        class="flex flex-col gap-3"
      >
        <div
          v-for="toast in toasts"
          :key="toast.id"
          class="pointer-events-auto flex items-start gap-3 rounded-xl px-4 py-3.5 shadow-xl border backdrop-blur-sm cursor-default select-none"
          :class="kindStyle(toast.kind).card"
        >
          <!-- Icon -->
          <div
            class="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5"
            :class="kindStyle(toast.kind).icon"
          >
            <component :is="kindStyle(toast.kind).component" :size="16" />
          </div>

          <!-- Content -->
          <div class="flex-1 min-w-0">
            <p class="text-sm font-semibold leading-tight" :class="kindStyle(toast.kind).title">
              {{ toast.title }}
            </p>
            <p class="text-xs mt-0.5 leading-snug" :class="kindStyle(toast.kind).body">
              {{ toast.body }}
            </p>
          </div>

          <!-- Dismiss -->
          <button
            class="flex-shrink-0 opacity-40 hover:opacity-80 transition-opacity mt-0.5"
            @click="dismiss(toast.id)"
          >
            <XIcon :size="14" :class="kindStyle(toast.kind).title" />
          </button>
        </div>
      </TransitionGroup>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { PhoneIncoming, PhoneOff, Info, AlertCircle, X as XIcon } from 'lucide-vue-next'
import { useToasts } from '@/composables/useToasts'
import type { ToastKind } from '@/composables/useToasts'

const { toasts, dismiss } = useToasts()

const STYLES: Record<ToastKind, { card: string; icon: string; title: string; body: string; component: any }> = {
  incoming: {
    card:  'bg-white border-green-200',
    icon:  'bg-green-100',
    title: 'text-green-900',
    body:  'text-green-700',
    component: PhoneIncoming,
  },
  ended: {
    card:  'bg-white border-slate-200',
    icon:  'bg-slate-100',
    title: 'text-slate-800',
    body:  'text-slate-500',
    component: PhoneOff,
  },
  info: {
    card:  'bg-white border-indigo-200',
    icon:  'bg-indigo-100',
    title: 'text-indigo-900',
    body:  'text-indigo-700',
    component: Info,
  },
  error: {
    card:  'bg-white border-red-200',
    icon:  'bg-red-100',
    title: 'text-red-900',
    body:  'text-red-700',
    component: AlertCircle,
  },
}

function kindStyle(kind: ToastKind) {
  return STYLES[kind]
}
</script>

<style scoped>
.toast-enter-active { transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1); }
.toast-leave-active { transition: all 0.2s ease-in; }
.toast-enter-from  { opacity: 0; transform: translateX(40px) scale(0.95); }
.toast-leave-to    { opacity: 0; transform: translateX(40px) scale(0.95); }
</style>
