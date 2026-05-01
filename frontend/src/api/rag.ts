import client from './client'

export interface ResyncResponse {
  parser_ok: boolean
  documents_indexed: number
  document_count: number
  message: string
}

export async function resyncEnriched(forceDownload = true): Promise<ResyncResponse> {
  const { data } = await client.post<ResyncResponse>('/rag/resync', {
    force_download: forceDownload,
  })
  return data
}

export async function resyncRaw(forceDownload = true): Promise<ResyncResponse> {
  const { data } = await client.post<ResyncResponse>('/rag/resync-raw', {
    force_download: forceDownload,
  })
  return data
}
