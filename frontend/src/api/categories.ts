import client from './client'
import type { Category } from '@/types'

export async function listCategories(): Promise<Category[]> {
  const { data } = await client.get<Category[]>('/api/categories/')
  return data
}

export async function createCategory(name: string): Promise<Category> {
  const { data } = await client.post<Category>('/api/categories/', { name })
  return data
}

export async function deleteCategory(id: number): Promise<void> {
  await client.delete(`/api/categories/${id}`)
}

export async function resyncCategories(): Promise<{ queued: number; message: string }> {
  const { data } = await client.post('/api/categories/resync')
  return data
}
