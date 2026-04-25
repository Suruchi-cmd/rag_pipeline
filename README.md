
Minimum stock body:


{
  "query": "your question",
  "response_mode": "compact",
  "use_custom_prompt": false,
  "similarity_cutoff": 0.0
}
Why each:

response_mode: "compact" — llama_index default. Your server default is tree_summarize, not stock.
use_custom_prompt: false — skips the customer-service template, uses built-in text_qa_template/refine_template.
similarity_cutoff: 0.0 — disables the 0.6 filter (stock has no cutoff).
use_hyde, multi_step — already default false. Don't need to send.
curl:


curl -X POST :8825/rag/query -H 'content-type: application/json' \
  -d '{"query":"hours?","response_mode":"compact","use_custom_prompt":false,"similarity_cutoff":0.0}'
Caveat — one non-stock bit isn't flagged: repositories.py:268 does optimized_top_k = min(top_k * 2, 15) — retrieves 2× what you asked before trimming to top_k in results. Not exposed as flag. Want me to gate that too (e.g., raw_top_k: bool)?