<template>
  <AppLayout>
    <RouterView />
  </AppLayout>
  <LiveCallsPanel />
  <ToastContainer />
</template>

<script setup lang="ts">
import AppLayout from '@/components/AppLayout.vue'
import LiveCallsPanel from '@/components/LiveCallsPanel.vue'
import ToastContainer from '@/components/ToastContainer.vue'
import { useEvents } from '@/composables/useEvents'
import { useLiveCalls } from '@/composables/useLiveCalls'
import { useToasts } from '@/composables/useToasts'

const { add, remove } = useLiveCalls()
const { push } = useToasts()

useEvents({
  call_started(data) {
    add(data)
    push({
      kind: 'incoming',
      title: 'Incoming Call',
      body: data.phone_number,
      ttl: 8_000,
    })
  },

  call_ended(data) {
    remove(data.call_sid)
    push({
      kind: 'ended',
      title: data.status === 'abandoned' ? 'Call Dropped' : 'Call Ended',
      body: data.needs_human ? `${data.phone_number ?? ''} · needs follow-up` : (data.summary ?? 'Call finished'),
      ttl: 6_000,
    })
  },
})
</script>
