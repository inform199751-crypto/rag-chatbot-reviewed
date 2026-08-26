from datetime import datetime

from pydantic import BaseModel, Field


class StoredMessage(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime
    grounded: bool | None = None
    """Kept on restore so a reloaded thread renders the same as the live one did. An ungrounded
    answer that comes back looking grounded is the failure this project already fixed once in
    the stream; dropping the flag here would reintroduce it on refresh."""

    sources: list[dict] = Field(default_factory=list)


class ConversationMessages(BaseModel):
    conversation_id: str
    messages: list[StoredMessage] = Field(default_factory=list)


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationList(BaseModel):
    conversations: list[ConversationSummary] = Field(default_factory=list)
