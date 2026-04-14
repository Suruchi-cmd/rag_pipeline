import json
import logging
import re

import httpx

logger = logging.getLogger(__name__)

RAG_URL = "http://localhost:8000/rag/retrieve"

SYSTEM_PROMPT = (
    "You are a helpful customer support assistant for AeroSports Scarborough trampoline park. "
    "Answer questions about pricing, bookings, hours, and activities. "
    "When the customer says goodbye, confirms they have everything they need, or when you cannot "
    "help them and they need a human agent, call the end_call function. "
    "Do not say goodbye in text — call the function instead. "
    "If for any reason you cannot call the end_call function, instead emit this exact format on "
    'its own line: [END_CALL]{"summary": "...", "needs_human": true_or_false, "flag_reason": "..."}[/END_CALL]'
)


def _fetch_rag_context(query: str) -> str:
    """Query the RAG service at port 8000 and return formatted context."""
    try:
        response = httpx.post(
            RAG_URL,
            json={"query": query, "top_k": 3},
            timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()
        docs = data.get("source_documents", [])
        if not docs:
            return ""
        snippets = [doc["content"] for doc in docs if doc.get("content")]
        return "\n\n---\n\n".join(snippets)
    except Exception as e:
        logger.warning("RAG context fetch failed: %s", e)
        return ""

_END_CALL_TOOL = {
    "type": "function",
    "function": {
        "name": "end_call",
        "description": (
            "Call this when the customer has indicated the conversation is complete "
            "(saying goodbye, confirming they have what they need, etc.) OR when the customer "
            "needs a human agent and you cannot help further."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A 1-2 sentence summary of what happened in the call",
                },
                "needs_human": {
                    "type": "boolean",
                    "description": "True if a human agent should follow up with this customer",
                },
                "flag_reason": {
                    "type": "string",
                    "description": "If needs_human is true, briefly why. Empty string otherwise.",
                },
            },
            "required": ["summary", "needs_human", "flag_reason"],
        },
    },
}

_SENTINEL_RE = re.compile(r"\[END_CALL\](.+?)\[/END_CALL\]", re.DOTALL)


def process_turn(call_id: int, user_message: str, conversation_history: list[dict]) -> dict:
    conversation_history.append({"role": "user", "content": user_message})
    db.log_message(call_id, "user", user_message)

    rag_context = _fetch_rag_context(user_message)
    if rag_context:
        system_content = (
            SYSTEM_PROMPT
            + "\n\nUse the following knowledge base excerpts to answer accurately:\n\n"
            + rag_context
        )
    else:
        system_content = SYSTEM_PROMPT

    messages = [{"role": "system", "content": system_content}] + conversation_history

    response = httpx.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "phi4:latest",
            "messages": messages,
            "tools": [_END_CALL_TOOL],
            "stream": False,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    message = data["message"]

    tool_calls = message.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            try:
                if tc.get("function", {}).get("name") == "end_call":
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    db.end_call(
                        call_id,
                        summary=args["summary"],
                        needs_human=args["needs_human"],
                        flag_reason=args.get("flag_reason") or None,
                    )
                    return {
                        "assistant_message": "",
                        "call_ended": True,
                        "end_reason": args["summary"],
                    }
            except Exception:
                pass

    content = message.get("content", "")

    match = _SENTINEL_RE.search(content)
    if match:
        try:
            args = json.loads(match.group(1))
            db.end_call(
                call_id,
                summary=args["summary"],
                needs_human=args["needs_human"],
                flag_reason=args.get("flag_reason") or None,
            )
            return {
                "assistant_message": "",
                "call_ended": True,
                "end_reason": args["summary"],
            }
        except Exception:
            pass

    db.log_message(call_id, "assistant", content)
    return {
        "assistant_message": content,
        "call_ended": False,
        "end_reason": None,
    }
