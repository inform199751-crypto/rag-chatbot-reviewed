import time

from chat_history import ChatHistory
from config import settings
from fastapi import WebSocket
from helpers.log import get_logger
from helpers.prettier import prettify_source
from schemas.chat import ChatRequest
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


# TODO: https://github.com/umbertogriffo/rag-chatbot/pull/10#discussion_r2936567672
async def stream_chat_response(
    websocket: WebSocket, llm_client: LlamaCppClientDep, query: ChatRequest, chat_history: ChatHistory
):
    """
    Helper function to stream chat responses token by token.
     Args:
        websocket (WebSocket): The WebSocket connection to send responses through.
        llm_client (LamaCppClientDep): The LLM client dependency for generating responses.
        query (ChatRequest): The chat request containing the user's query.
        chat_history (ChatHistory): The chat history for this connection.
    """
    try:
        start_time = time.time()

        full_response = ""
        stream = await answer(
            llm=llm_client,
            question=query.text,
            chat_history=chat_history,
            max_new_tokens=settings.MAX_NEW_TOKENS,
        )
        async for output in stream:
            token = llm_client.parse_token(output)
            if token:
                full_response += token
                await websocket.send_text(token)

        if llm_client.model_settings.reasoning:
            final_answer = extract_content_after_reasoning(full_response, llm_client.model_settings.reasoning_stop_tag)
            if final_answer == "":
                final_answer = "I didn't provide the answer; perhaps I can try again."
        else:
            final_answer = full_response

        chat_history.append(f"question: {query.text}, answer: {final_answer}")
        logger.debug(f"Updated chat history: {chat_history}")

        took = time.time() - start_time
        logger.info(f"\n--- Took {took:.2f} seconds ---")
    except Exception as exc:
        logger.exception("Error during streaming: %s", exc)
        await websocket.send_text("Error during streaming.")


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
    Helper function to stream RAG responses token by token.
     Args:
        websocket (WebSocket): The WebSocket connection to send responses through.
        llm_client (LamaCppClientDep): The LLM client dependency for generating responses.
        query (ChatRequest): The chat request containing the user's query.
        chat_history (ChatHistory): The chat history for this connection.
        index (VectorDatabaseDep): The vector database dependency for retrieval.
        reranker (RerankerDep): The cross-encoder for the second stage, or None to retrieve in
            one stage. Defaults to None so existing callers -- including the tests -- keep
            working without being rewritten.
    """
    try:
        start_time = time.time()
        ctx_synthesis_strategy = get_ctx_synthesis_strategy(settings.SYNTHESIS_STRATEGY, llm=llm_client)

        retrieval_response = ""
        full_response = ""

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
        retrieved_contents = outcome.documents

        if retrieved_contents:
            retrieval_response += "Here are the retrieved text chunks with a content preview: \n\n"

            for source in outcome.sources:
                retrieval_response += prettify_source(source)
                retrieval_response += "\n\n"
        else:
            retrieval_response += explain_no_context(outcome)

        await websocket.send_text(retrieval_response)
        await websocket.send_text("-" * 20 + "\n\n")

        if not retrieved_contents and not settings.ANSWER_WITHOUT_CONTEXT:
            # Stop here rather than answering from the model's own knowledge. `answer_with_context`
            # would silently fall back to plain chat, and the reply would be indistinguishable
            # from a document-backed one.
            await websocket.send_text(
                "**No answer:** \n\nNothing in the indexed documents covers this, and this "
                "deployment is configured not to answer without a source.\n"
            )
            chat_history.append(f"question: {query.text}, answer: (declined -- no supporting documents)")
            logger.info("Declined to answer without context (reason=%s)", outcome.reason)
            return

        # Label the answer whenever it is not grounded. The header is the only place the user
        # can tell the two cases apart -- the prose that follows reads identically either way.
        await websocket.send_text(
            "**Answer:** \n\n"
            if retrieved_contents
            else "**Answer (from the model's own knowledge, not your documents):** \n\n"
        )

        streamer, _ = await answer_with_context(
            llm_client,
            ctx_synthesis_strategy,
            query.text,
            chat_history,
            retrieved_contents,
            settings.MAX_NEW_TOKENS,
        )

        async for output in streamer:
            token = llm_client.parse_token(output)
            if token:
                full_response += token
                await websocket.send_text(token)

        if llm_client.model_settings.reasoning:
            final_answer = extract_content_after_reasoning(full_response, llm_client.model_settings.reasoning_stop_tag)
            if final_answer == "":
                final_answer = "I wasn't able to provide the answer; Do you want me to try again?"
        else:
            final_answer = full_response

        chat_history.append(f"question: {query.text}, answer: {final_answer}")

        took = time.time() - start_time
        logger.info(f"\n--- Took {took:.2f} seconds ---")

    except Exception as exc:
        logger.exception("Error during RAG streaming: %s", exc)
        await websocket.send_text("Error during RAG streaming.")
