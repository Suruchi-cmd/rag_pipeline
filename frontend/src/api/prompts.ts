import client from './client'
import type { PromptDetail, PromptSummary, PromptVersion } from '@/types'

export async function listPrompts(): Promise<PromptSummary[]> {
  const { data } = await client.get<PromptSummary[]>('/api/prompts')
  return data
}

export async function getPrompt(id: number): Promise<PromptDetail> {
  const { data } = await client.get<PromptDetail>(`/api/prompts/${id}`)
  return data
}

export async function createPrompt(payload: {
  slug: string
  name: string
  description?: string | null
  content: string
}): Promise<PromptSummary> {
  const { data } = await client.post<PromptSummary>('/api/prompts', payload)
  return data
}

export async function updatePromptMeta(
  id: number,
  payload: { name?: string; description?: string | null },
): Promise<PromptSummary> {
  const { data } = await client.put<PromptSummary>(`/api/prompts/${id}`, payload)
  return data
}

export async function deletePrompt(id: number): Promise<void> {
  await client.delete(`/api/prompts/${id}`)
}

export async function createVersion(
  promptId: number,
  payload: { content: string; activate?: boolean; label?: string | null },
): Promise<PromptVersion> {
  const { data } = await client.post<PromptVersion>(
    `/api/prompts/${promptId}/versions`,
    payload,
  )
  return data
}

export async function updateVersion(
  promptId: number,
  versionId: number,
  payload: { content?: string; label?: string | null },
): Promise<PromptVersion> {
  const { data } = await client.put<PromptVersion>(
    `/api/prompts/${promptId}/versions/${versionId}`,
    payload,
  )
  return data
}

export async function activateVersion(
  promptId: number,
  versionId: number,
): Promise<PromptVersion> {
  const { data } = await client.post<PromptVersion>(
    `/api/prompts/${promptId}/versions/${versionId}/activate`,
  )
  return data
}

export async function deleteVersion(
  promptId: number,
  versionId: number,
): Promise<void> {
  await client.delete(`/api/prompts/${promptId}/versions/${versionId}`)
}
