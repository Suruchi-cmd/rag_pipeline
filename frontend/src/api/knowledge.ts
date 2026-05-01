import client from './client'
import type { KnowledgeChunk } from '@/types'

export async function listChunks(): Promise<KnowledgeChunk[]> {
  const { data } = await client.get<KnowledgeChunk[]>('/api/knowledge/chunks')
  return data
}

export async function createChunk(name: string, content: string): Promise<KnowledgeChunk> {
  const { data } = await client.post<KnowledgeChunk>('/api/knowledge/chunks', { name, content })
  return data
}

export async function updateChunk(
  id: number,
  name: string,
  content: string,
): Promise<KnowledgeChunk> {
  const { data } = await client.put<KnowledgeChunk>(`/api/knowledge/chunks/${id}`, {
    name,
    content,
  })
  return data
}

export async function deleteChunk(id: number): Promise<void> {
  await client.delete(`/api/knowledge/chunks/${id}`)
}

export async function resyncKnowledge(): Promise<{ queued: number; message: string }> {
  const { data } = await client.post<{ queued: number; message: string }>(
    '/api/knowledge/resync',
  )
  return data
}
