import time
from typing import Any, NamedTuple

from chat_history import ChatHistory
from config import settings
from fastapi import WebSocket
from helpers.log import get_logger
from schemas.chat import ChatRequest
from schemas.stream_events import (
    answer_start_event,
    done_event,
    error_event,
    sources_event,
    token_event,
)
from services.chat_service.conversation_handler import (
    answer,
    answer_with_context,
    extract_content_after_reasoning,
    refine_question,
)
from services.chat_service.ctx_strategy import get_ctx_synthesis_strategy
from services.retrieval_service import explain_no_context, retrieve

from api.deps import LlamaCppClientDep, RerankerDep, VectorDatabaseDep

logger = get_logger(__name__)


class TurnResult(NamedTuple):
    """What one exchange produced, so the caller can persist it.

    Returned rather than written here on purpose: these functions own the socket, not the
    database, and a failed write should not be able to interrupt a stream that already
    succeeded.
    """

    answer: str = ""
    grounded: bool | None = None
    # A tuple, not a list: NamedTuple evaluates defaults once and shares them across every
    # instance, so a mutable default would be one object handed out to all of them.
    sources: tuple[dict[str, Any], ...] = ()
    declined: bool = False


async def _stream_tokens(websocket: WebSocket, llm_client, streamer) -> str:
    """Forward generated tokens as TOKEN events and return the assembled text."""
    full_response = ""
    async for output in streamer:
        token = llm_client.parse_token(output)
        if token:
            full_response += token
            await websocket.send_text(token_event(token))
    return full_response


def _final_answer(llm_client, full_response: str, fallback: str) -> str:
    if llm_client.model_settings.reasoning:
        answer_text = extract_content_after_reasoning(full_response, llm_client.model_settings.reasoning_stop_tag)
        return answer_text or fallback
    return full_response


# TODO: https://github.com/umbertogriffo/rag-chatbot/pull/10#discussion_r2936567672
async def stream_chat_response(
    websocket: WebSocket, llm_client: LlamaCppClientDep, query: ChatRequest, chat_history: ChatHistory
):
    """
    Stream a plain chat response as typed events.

     Args:
        websocket (WebSocket): The WebSocket connection to send responses through.
        llm_client (LamaCppClientDep): The LLM client dependency for generating responses.
        query (ChatRequest): The chat request containing the user's query.
        chat_history (ChatHistory): The chat history for this connection.
    """
    start_time = time.time()
    result = TurnResult()
    try:
        # No retrieval on this path, so nothing is grounded in documents. Saying so explicitly
        # keeps the client's rendering rule the same for both modes.
        await websocket.send_text(answer_start_event(grounded=False))

        stream = await answer(
            llm=llm_client,
            question=query.text,
            chat_history=chat_history,
            max_new_tokens=settings.MAX_NEW_TOKENS,
        )
        full_response = await _stream_tokens(websocket, llm_client, stream)

        final_answer = _final_answer(llm_client, full_response, "I didn't provide the answer; perhaps I can try again.")
        chat_history.append(f"question: {query.text}, answer: {final_answer}")
        logger.debug(f"Updated chat history: {chat_history}")
        result = TurnResult(answer=final_answer)
    except Exception as exc:
        logger.exception("Error during streaming: %s", exc)
        await websocket.send_text(error_event("Error during streaming."))
    finally:
        # DONE on every path, including the failure above. The client unlocks its input on this
        # event, so a path that skips it leaves the UI stuck with no way to recover.
        await websocket.send_text(done_event(took_seconds=round(time.time() - start_time, 2)))
        logger.info(f"\n--- Took {time.time() - start_time:.2f} seconds ---")
    return result


# TODO: https://github.com/umbertogriffo/rag-chatbot/pull/10#discussion_r2936567672
async def stream_rag_response(
    websocket: WebSocket,
    llm_client: LlamaCppClientDep,
    query: ChatRequest,
    chat_history: ChatHistory,
    index: VectorDatabaseDep,
    reranker: RerankerDep = None,
):
    """
    Stream a retrieval-augmented response as typed events.

     Args:
        websocket (WebSocket): The WebSocket connection to send responses through.
        llm_client (LamaCppClientDep): The LLM client dependency for generating responses.
        query (ChatRequest): The chat request containing the user's query.
        chat_history (ChatHistory): The chat history for this connection.
        index (VectorDatabaseDep): The vector database dependency for retrieval.
        reranker (RerankerDep): The cross-encoder for the second stage, or None to retrieve in
            one stage.
    """
    start_time = time.time()
    declined = False
    result = TurnResult()
    try:
        ctx_synthesis_strategy = get_ctx_synthesis_strategy(settings.SYNTHESIS_STRATEGY, llm=llm_client)

        refined_user_input = await refine_question(
            llm_client, query.text, chat_history=chat_history, max_new_tokens=settings.MAX_NEW_TOKENS
        )
        outcome = retrieve(
            index,
            query=refined_user_input,
            num_retrievals=settings.NUM_RETRIEVALS,
            threshold=settings.RETRIEVAL_THRESHOLD,
            reranker=reranker,
            candidates=settings.RERANK_CANDIDATES,
            rerank_threshold=settings.RERANK_THRESHOLD,
        )
        grounded = bool(outcome.documents)

        await websocket.send_text(
            sources_event(
                documents=outcome.sources,
                grounded=grounded,
                reason=outcome.reason.value if outcome.reason else None,
                message="" if grounded else explain_no_context(outcome).strip(),
            )
        )

        if not grounded and not settings.ANSWER_WITHOUT_CONTEXT:
            # Stop rather than answering from the model's own knowledge. `answer_with_context`
            # falls back to plain chat on an empty context, and that reply would be
            # indistinguishable from a document-backed one.
            declined = True
            chat_history.append(f"question: {query.text}, answer: (declined -- no supporting documents)")
            logger.info("Declined to answer without context (reason=%s)", outcome.reason)
            result = TurnResult(
                answer=explain_no_context(outcome).strip(),
                grounded=False,
                sources=tuple(outcome.sources),
                declined=True,
            )
            return result

        await websocket.send_text(answer_start_event(grounded=grounded))

        streamer, _ = await answer_with_context(
            llm_client,
            ctx_synthesis_strategy,
            query.text,
            chat_history,
            outcome.documents,
            settings.MAX_NEW_TOKENS,
        )
        full_response = await _stream_tokens(websocket, llm_client, streamer)

        final_answer = _final_answer(
            llm_client, full_response, "I wasn't able to provide the answer; Do you want me to try again?"
        )
        chat_history.append(f"question: {query.text}, answer: {final_answer}")
        result = TurnResult(answer=final_answer, grounded=grounded, sources=tuple(outcome.sources))
    except Exception as exc:
        logger.exception("Error during RAG streaming: %s", exc)
        await websocket.send_text(error_event("Error during RAG streaming."))
    finally:
        await websocket.send_text(done_event(took_seconds=round(time.time() - start_time, 2), declined=declined))
        logger.info(f"\n--- Took {time.time() - start_time:.2f} seconds ---")
