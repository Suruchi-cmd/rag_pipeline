export interface Call {
  id: number
  call_sid: string
  phone_number: string
  started_at: string
  ended_at: string | null
  status: 'active' | 'completed' | 'abandoned'
  summary: string | null
  needs_human: boolean
  flag_reason: string | null
  total_turns: number
}

export interface Message {
  id: number
  call_id: number
  role: 'user' | 'assistant'
  content: string
  turn_number: number
  created_at: string
  was_interrupted: boolean
}

export interface RagChunk {
  content: string
  score: number
  metadata: {
    file_name: string | null
    page: number | null
    source: string | null
  }
}

export interface RAGRetrieval {
  id: number
  call_id: number
  turn_number: number
  original_query: string
  rewritten_query: string
  retrieved_chunks: RagChunk[]
  was_skipped: boolean
  created_at: string
}

export interface BookingChange {
  id: number
  call_id: number
  caller_name: string
  caller_phone: string
  change_details: string
  created_at: string
}

export interface CallDetail {
  call: Call
  messages: Message[]
  rag_retrievals: RAGRetrieval[]
  booking_change: BookingChange | null
}

export interface Stats {
  total_calls: number
  needs_human: number
  active_calls: number
}

export interface Category {
  id: number
  name: string
  created_at: string
}

export interface KnowledgeChunk {
  id: number
  name: string
  content: string
  created_at: string
  updated_at: string
}

export interface Turn {
  turn_number: number
  user: Message | null
  assistant: Message | null
  rag: RAGRetrieval | null
}
