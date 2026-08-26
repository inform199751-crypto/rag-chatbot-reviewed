"""Reading and writing persisted conversations.

The prompt window is derived here rather than stored. `ChatHistory` caps itself at
CHAT_HISTORY_LENGTH turns because each one costs context on the next request; the transcript has
no such limit. Keeping one store and deriving the other means the two cannot drift -- which they
did when history was a single in-memory list serving both purposes.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from chat_history import ChatHistory
from sqlmodel import Session, select

from models.conversation import ChatMessage, Conversation

logger = logging.getLogger(__name__)

TITLE_MAX_LENGTH = 80


class ConversationStore:
    """Persistence for chat threads."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def ensure(self, conversation_id: str, first_message: str = "") -> Conversation:
        """Fetch the thread, creating it if this is its first message."""
        conversation = self._session.get(Conversation, conversation_id)
        if conversation is None:
            conversation = Conversation(
                conversation_id=conversation_id,
                title=first_message[:TITLE_MAX_LENGTH],
            )
            self._session.add(conversation)
            self._session.commit()
        return conversation

    def append_turn(
        self,
        conversation_id: str,
        question: str,
        answer: str,
        grounded: bool | None = None,
        sources: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Record one exchange.

        Written as a pair after the answer completes rather than as two independent writes. A
        question stored without its answer would come back as a thread that appears to have been
        ignored, and would also poison the prompt window on the next request.
        """
        conversation = self.ensure(conversation_id, first_message=question)

        self._session.add(ChatMessage(conversation_id=conversation_id, role="user", content=question))
        self._session.add(
            ChatMessage(
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
                grounded=grounded,
                sources_json=json.dumps(sources, ensure_ascii=False) if sources else None,
            )
        )
        conversation.updated_at = datetime.now(timezone.utc)
        self._session.add(conversation)
        self._session.commit()

    def messages(self, conversation_id: str, limit: int | None = None) -> list[ChatMessage]:
        """Every message in the thread, oldest first."""
        statement = select(ChatMessage).where(ChatMessage.conversation_id == conversation_id).order_by(ChatMessage.id)
        if limit is not None:
            # Take the newest `limit` and put them back in order. Slicing the front would return
            # the beginning of the thread, which is the opposite of what a prompt window wants.
            rows = list(self._session.exec(statement).all())
            return rows[-limit:]
        return list(self._session.exec(statement).all())

    def load_prompt_window(self, conversation_id: str, total_length: int) -> ChatHistory:
        """
        Rebuild the bounded history the prompt uses, from the tail of the transcript.

        Reconnecting used to start the model from nothing: the transcript was gone with the
        process, so a follow-up question like "and the second one?" had no antecedent and
        `refine_question` rewrote it into something unanswerable.
        """
        history = ChatHistory(total_length=total_length)
        if total_length <= 0:
            return history

        # Two rows per turn, so ask for twice as many.
        rows = self.messages(conversation_id, limit=total_length * 2)

        pending_question: str | None = None
        for row in rows:
            if row.role == "user":
                pending_question = row.content
            elif pending_question is not None:
                history.append(f"question: {pending_question}, answer: {row.content}")
                pending_question = None

        return history

    def clear(self, conversation_id: str) -> None:
        """Delete a thread and everything in it."""
        for message in self.messages(conversation_id):
            self._session.delete(message)
        conversation = self._session.get(Conversation, conversation_id)
        if conversation is not None:
            self._session.delete(conversation)
        self._session.commit()

    def list_conversations(self, limit: int = 50) -> list[Conversation]:
        """Most recently updated first."""
        statement = select(Conversation).order_by(Conversation.updated_at.desc()).limit(limit)
        return list(self._session.exec(statement).all())
