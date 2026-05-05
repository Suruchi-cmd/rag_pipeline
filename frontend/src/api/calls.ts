import client from './client'
import type { Call, CallDetail, Stats } from '@/types'

export interface ListCallsParams {
  offset?: number
  limit?: number
  needs_human?: boolean | null
  status?: string
}

export async function listCalls(params: ListCallsParams = {}): Promise<Call[]> {
  const query: Record<string, unknown> = {
    offset: params.offset ?? 0,
    limit: params.limit ?? 50,
  }
  if (params.needs_human != null) query.needs_human = params.needs_human
  if (params.status) query.status = params.status
  const { data } = await client.get<Call[]>('/api/calls/', { params: query })
  return data
}

export async function getCall(id: number): Promise<CallDetail> {
  const { data } = await client.get<CallDetail>(`/api/calls/${id}`)
  return data
}

export async function getStats(): Promise<Stats> {
  const { data } = await client.get<Stats>('/api/calls/stats')
  return data
}

export async function deleteCall(id: number): Promise<void> {
  await client.delete(`/api/calls/${id}`)
}

export async function bulkDeleteCalls(ids: number[]): Promise<{ deleted: number }> {
  const { data } = await client.post<{ deleted: number }>('/api/calls/bulk-delete', { ids })
  return data
}
