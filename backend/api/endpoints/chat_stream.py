import state
from chat_history import init_chat_history
from config import settings
from fastapi import APIRouter, Response, WebSocket, WebSocketDisconnect
from helpers.log import get_logger
from schemas.chat import ChatRequest
from services.conversation_store import ConversationStore
from sqlmodel import Session

from api.deps import LlamaCppClientDep, RerankerDep, VectorDatabaseDep
from api.services.chat_stream import TurnResult, stream_chat_response, stream_rag_response

logger = get_logger(__name__)

router = APIRouter()


@router.delete(
    path="/chat/history",
    status_code=204,
)
async def clear_chat_history():
    """
    Clear the chat history.

    Chat history is owned by each individual WebSocket connection, so there is no shared server-side
    state left for this endpoint to reset: dropping the connection is what clears it, and the
    frontend already reconnects before calling this. Kept as a no-op so the client contract holds.

    Persisted transcripts are deleted through `DELETE /conversations/{id}` instead -- deliberately
    a separate call, because forgetting the in-memory context and destroying a stored thread are
    different intentions and should not share one button.
    """
    return Response(status_code=204)


def _persist(conversation_id: str, question: str, result: TurnResult) -> None:
    """
    Store one completed turn.

    A fresh session per turn rather than one held for the life of the connection: a WebSocket can
    stay open for hours, and a connection-scoped session would hold a SQLite transaction open for
    that whole time, blocking writers.

    Failures are logged and swallowed. The answer has already been streamed and read by the time
    this runs, so raising here would turn a bookkeeping problem into a visible error about a
    request that actually succeeded.
    """
    if not result.answer:
        return
    try:
        with Session(state.db_engine) as session:
            ConversationStore(session).append_turn(
                conversation_id=conversation_id,
                question=question,
                answer=result.answer,
                grounded=result.grounded,
                sources=list(result.sources),
            )
    except Exception as exc:
        logger.exception("Failed to persist turn for conversation %s: %s", conversation_id, exc)


@router.websocket(
    path="/chat/stream",
)
async def chat_stream(
    websocket: WebSocket,
    llm_client: LlamaCppClientDep,
    index: VectorDatabaseDep,
    reranker: RerankerDep,
    conversation_id: str | None = None,
):
    """
    WebSocket endpoint for streaming chat responses.

    `conversation_id` is a query parameter the client generates and keeps. Omit it and the
    conversation is not persisted, which keeps the endpoint usable from a plain socket client
    with no storage of its own.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted (conversation_id=%s)", conversation_id)

    # Restore the prompt window from the transcript so a reconnect does not amnesia the model.
    # Previously each connection started empty: after a refresh, a follow-up like "and the second
    # one?" had no antecedent and `refine_question` rewrote it into something unanswerable.
    chat_history = init_chat_history(settings.CHAT_HISTORY_LENGTH)
    if conversation_id:
        try:
            with Session(state.db_engine) as session:
                chat_history = ConversationStore(session).load_prompt_window(
                    conversation_id, settings.CHAT_HISTORY_LENGTH
                )
            logger.info("Restored %d prior turns for conversation %s", len(chat_history), conversation_id)
        except Exception as exc:
            # An unreadable transcript should cost the user their context, not their session.
            logger.exception("Could not restore history for %s: %s", conversation_id, exc)

    try:
        while True:
            data = await websocket.receive_json()
            logger.info(f"Received data: {data}")
            query = ChatRequest(**data)
            if query.rag:
                result = await stream_rag_response(websocket, llm_client, query, chat_history, index, reranker)
            else:
                result = await stream_chat_response(websocket, llm_client, query, chat_history)

            if conversation_id:
                _persist(conversation_id, query.text, result)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception(f"Unexpected error in WebSocket handler: {e}")
        raise
