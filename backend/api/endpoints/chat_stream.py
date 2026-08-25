from chat_history import init_chat_history
from config import settings
from fastapi import APIRouter, Response, WebSocket, WebSocketDisconnect
from helpers.log import get_logger
from schemas.chat import ChatRequest

from api.deps import LlamaCppClientDep, VectorDatabaseDep
from api.services.chat_stream import stream_chat_response, stream_rag_response

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
    """
    return Response(status_code=204)


@router.websocket(
    path="/chat/stream",
)
async def chat_stream(websocket: WebSocket, llm_client: LlamaCppClientDep, index: VectorDatabaseDep):
    """WebSocket endpoint for streaming chat responses token by token."""
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    # Each connection owns its chat history. A single module-level instance would leak one user's
    # questions and answers into another user's prompt context, and let either one clear the other.
    chat_history = init_chat_history(settings.CHAT_HISTORY_LENGTH)
    try:
        while True:
            data = await websocket.receive_json()
            logger.info(f"Received data: {data}")
            query = ChatRequest(**data)
            if query.rag:
                await stream_rag_response(websocket, llm_client, query, chat_history, index)
            else:
                await stream_chat_response(websocket, llm_client, query, chat_history)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception(f"Unexpected error in WebSocket handler: {e}")
        raise
