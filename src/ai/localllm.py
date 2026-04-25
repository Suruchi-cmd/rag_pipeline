from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec


class OllamaChat:
    """
    Async-capable chat class wrapping OllamaLLM with optional persistent history.

    Features:
    - chat_with_history()        : sync chat, persists messages per user/session
    - chat()                     : sync stateless chat, no memory
    - achat_with_history()       : async chat with persistent history
    - achat()                    : async stateless chat
    - stream / astream variants  : token-by-token generators
    - get_history()              : retrieve raw history for a session
    - clear_history()            : wipe a session's message history
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful assistant. Answer all questions to the best of your ability."
    )

    def __init__(
        self,
        model: str = "phi4:latest",
        db_url: str = "sqlite:///memory.db",
        system_prompt: str | None = None,
        temperature: float = 0.7,
        num_predict: int = -1,
        base_url: str = "https://aerollm.share.zrok.io",
    ):
        self.base_url = base_url
        self.model_name = model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # Sync engine  (sqlite:///memory.db)
        self._sync_engine = create_engine(db_url)

        # Async engine (sqlite+aiosqlite:///memory.db)
        async_db_url = db_url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
        self._async_engine = create_async_engine(async_db_url)

        self._model = OllamaLLM(
            model=model,
            temperature=temperature,
            num_predict=num_predict,
            base_url=self.base_url,
        )
        self._parser = StrOutputParser()

        # ── Shared prompt templates ──────────────────────────────────────────
        self._history_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.system_prompt),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )
        self._plain_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )

        # ── Chains ───────────────────────────────────────────────────────────
        _history_chain = self._history_prompt | self._model | self._parser
        self._plain_chain = self._plain_prompt | self._model | self._parser

        history_factory_config = [
            ConfigurableFieldSpec(
                id="session_id",
                annotation=str,
                name="Session ID",
                description="Unique identifier for the conversation session.",
                default="",
                is_shared=True,
            ),
            ConfigurableFieldSpec(
                id="user_id",
                annotation=str,
                name="User ID",
                description="Unique identifier for the user.",
                default="",
                is_shared=True,
            ),
        ]

        # Sync runnable — uses sync engine
        self._sync_runnable = RunnableWithMessageHistory(
            runnable=_history_chain,
            input_messages_key="question",
            get_session_history=self._get_sync_history,
            history_messages_key="history",
            history_factory_config=history_factory_config,
        )

        # Async runnable — uses async engine
        self._async_runnable = RunnableWithMessageHistory(
            runnable=_history_chain,
            input_messages_key="question",
            get_session_history=self._get_async_history,
            history_messages_key="history",
            history_factory_config=history_factory_config,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _session_key(self, session_id: str, user_id: str) -> str:
        return f"{user_id}---{session_id}"

    def _get_sync_history(self, session_id: str, user_id: str) -> SQLChatMessageHistory:
        return SQLChatMessageHistory(
            session_id=self._session_key(session_id, user_id),
            connection=self._sync_engine,
        )

    def _get_async_history(
        self, session_id: str, user_id: str
    ) -> SQLChatMessageHistory:
        return SQLChatMessageHistory(
            session_id=self._session_key(session_id, user_id),
            connection=self._async_engine,
        )

    def _make_config(self, session_id: str, user_id: str) -> dict:
        return {"configurable": {"session_id": session_id, "user_id": user_id}}

    # ── Sync API ─────────────────────────────────────────────────────────────

    def chat_with_history(self, question: str, session_id: str, user_id: str) -> str:
        """Synchronous chat that persists message history."""
        return self._sync_runnable.invoke(
            {"question": question},
            config=self._make_config(session_id, user_id),
        )

    def chat(self, question: str) -> str:
        """Synchronous stateless chat — no history read or written."""
        return self._plain_chain.invoke({"question": question})

    def stream_with_history(self, question: str, session_id: str, user_id: str):
        """Sync generator — streams tokens while persisting history."""
        yield from self._sync_runnable.stream(
            {"question": question},
            config=self._make_config(session_id, user_id),
        )

    def stream(self, question: str):
        """Sync generator — streams tokens, no history."""
        yield from self._plain_chain.stream({"question": question})

    # ── Async API ────────────────────────────────────────────────────────────

    async def achat_with_history(
        self, question: str, session_id: str, user_id: str
    ) -> str:
        """Async chat that persists message history."""
        return await self._async_runnable.ainvoke(
            {"question": question},
            config=self._make_config(session_id, user_id),
        )

    async def achat(self, question: str) -> str:
        """Async stateless chat — no history read or written."""
        return await self._plain_chain.ainvoke({"question": question})

    async def astream_with_history(self, question: str, session_id: str, user_id: str):
        """Async generator — streams tokens while persisting history."""
        async for chunk in self._async_runnable.astream(
            {"question": question},
            config=self._make_config(session_id, user_id),
        ):
            yield chunk

    async def astream(self, question: str):
        """Async generator — streams tokens, no history."""
        async for chunk in self._plain_chain.astream({"question": question}):
            yield chunk

    # ── History management ───────────────────────────────────────────────────

    def get_history(self, session_id: str, user_id: str) -> list:
        """Return a list of BaseMessage objects for this session (sync engine)."""
        return self._get_sync_history(session_id, user_id).messages

    def clear_history(self, session_id: str, user_id: str) -> None:
        """Wipe all messages for this session (sync engine)."""
        self._get_sync_history(session_id, user_id).clear()

    def __repr__(self) -> str:
        return f"OllamaChat(model={self.model_name!r})"


# ── Demo ─────────────────────────────────────────────────────────────────────


async def _async_demo():
    llm = OllamaChat(model="qwen3:4b")

    print("=== achat_with_history ===")
    r1 = await llm.achat_with_history(
        "My name is Rajan.", session_id="s1", user_id="rajan"
    )
    print(r1)

    r2 = await llm.achat_with_history(
        "What is my name?", session_id="s1", user_id="rajan"
    )
    print(r2)

    print("\n=== achat (no history) ===")
    r3 = await llm.achat("What is 2 + 2?")
    print(r3)

    print("\n=== astream_with_history ===")
    async for token in llm.astream_with_history(
        "Tell me a fun fact.", session_id="s1", user_id="rajan"
    ):
        print(token, end="", flush=True)
    print()

    print("\n=== History ===")
    for msg in llm.get_history(session_id="s1", user_id="rajan"):
        print(f"[{msg.type}] {msg.content[:80]}")

    llm.clear_history(session_id="s1", user_id="rajan")
    print("\nHistory cleared.")
