import { ref } from 'vue'

export type ToastKind = 'incoming' | 'ended' | 'info' | 'error'

export interface Toast {
  id: number
  kind: ToastKind
  title: string
  body: string
  ttl: number // ms
}

const toasts = ref<Toast[]>([])
let _id = 0

export function useToasts() {
  function push(toast: Omit<Toast, 'id'>) {
    const id = ++_id
    toasts.value.push({ ...toast, id })
    setTimeout(() => dismiss(id), toast.ttl)
    return id
  }

  function dismiss(id: number) {
    const i = toasts.value.findIndex((t) => t.id === id)
    if (i !== -1) toasts.value.splice(i, 1)
  }

  return { toasts, push, dismiss }
}
