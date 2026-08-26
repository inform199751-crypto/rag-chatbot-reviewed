"""Reading and deleting stored chat threads.

The stream itself stays on the WebSocket; this is the REST side that lets a client come back to
a conversation it started earlier. Before this, a refresh lost everything -- the history lived in
the process, so the transcript and the model's context both went with the connection.
"""

import json

from fastapi import APIRouter, Response
from schemas.conversation import (
    ConversationList,
    ConversationMessages,
    ConversationSummary,
    StoredMessage,
)
from services.conversation_store import ConversationStore

from api.deps import SessionDep

router = APIRouter()


@router.get("/conversations", response_model=ConversationList)
async def list_conversations(session: SessionDep, limit: int = 50):
    """Recently updated threads first."""
    store = ConversationStore(session)
    return ConversationList(
        conversations=[
            ConversationSummary(
                conversation_id=c.conversation_id,
                title=c.title,
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
            for c in store.list_conversations(limit=limit)
        ]
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationMessages)
async def get_conversation(conversation_id: str, session: SessionDep):
    """
    Every message in a thread, oldest first.

    An unknown id returns an empty list rather than a 404. The client generates the id locally
    before the first message, so "no rows yet" is the normal state of a new conversation, not an
    error -- and making the client special-case a 404 on first load would be noise.
    """
    store = ConversationStore(session)
    messages = [
        StoredMessage(
            id=m.id or 0,
            role=m.role,
            content=m.content,
            created_at=m.created_at,
            grounded=m.grounded,
            sources=json.loads(m.sources_json) if m.sources_json else [],
        )
        for m in store.messages(conversation_id)
    ]
    return ConversationMessages(conversation_id=conversation_id, messages=messages)


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, session: SessionDep):
    """Delete a thread and its messages."""
    ConversationStore(session).clear(conversation_id)
    return Response(status_code=204)
