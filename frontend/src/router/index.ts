import { createRouter, createWebHistory } from 'vue-router'
import DashboardView from '@/views/DashboardView.vue'
import CallsView from '@/views/CallsView.vue'
import CallDetailView from '@/views/CallDetailView.vue'
import CategoriesView from '@/views/CategoriesView.vue'

export default createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', component: DashboardView, meta: { title: 'Dashboard' } },
    { path: '/calls', component: CallsView, meta: { title: 'All Calls' } },
    { path: '/calls/:id', component: CallDetailView, meta: { title: 'Call Detail' } },
    { path: '/categories', component: CategoriesView, meta: { title: 'Categories' } },
  ],
})
