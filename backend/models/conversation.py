"""Persisted conversations.

Why two tables rather than one blob
    The prompt window and the transcript have opposite requirements and used to be the same
    object. `ChatHistory` is capped at CHAT_HISTORY_LENGTH turns because every turn in it costs
    context on the next request; a transcript must not be capped, because the point of keeping
    one is that the user can scroll back. Storing the transcript as rows lets the window be a
    query -- the last N turns -- instead of a second, conflicting policy.

    Rows also make each turn addressable, which is what lets an answer keep the sources it was
    built from. Serialised into one blob per conversation, that association is gone.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    """Timezone-aware UTC. `datetime.utcnow` returns a naive value, which compares wrongly
    against anything that carries an offset and is deprecated from Python 3.12."""
    return datetime.now(timezone.utc)


class Conversation(SQLModel, table=True):
    """One chat thread."""

    __tablename__ = "conversations"

    conversation_id: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    title: str = Field(default="")
    """First user message, truncated. Populated on the first turn so a conversation list has
    something to show without reading every message row."""


class ChatMessage(SQLModel, table=True):
    """One message in a thread."""

    __tablename__ = "chat_messages"

    id: int | None = Field(default=None, primary_key=True)
    conversation_id: str = Field(foreign_key="conversations.conversation_id", index=True)
    role: str = Field(default="user")
    """"user" or "assistant"."""

    content: str = Field(default="")
    created_at: datetime = Field(default_factory=_utcnow)

    grounded: bool | None = Field(default=None)
    """Whether an assistant message was backed by retrieved documents. None for user messages
    and for answers that never went through retrieval. Stored because it changes how the message
    should be rendered when the thread is reloaded -- an ungrounded answer that comes back
    looking grounded is the same failure this project already fixed once in the live stream."""

    sources_json: str | None = Field(default=None)
    """The retrieved chunks, as stored on the wire. Kept as JSON rather than normalised into
    another table: nothing queries inside it, and a restored thread only needs to show what the
    live one showed."""
