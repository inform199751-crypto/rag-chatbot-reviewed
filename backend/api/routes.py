from fastapi import APIRouter

from api.endpoints import capabilities, chat, chat_stream, conversations, documents, health

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(capabilities.router, tags=["capabilities"])
api_router.include_router(chat.router, prefix="", tags=["chat"])
api_router.include_router(documents.router, prefix="", tags=["documents"])
api_router.include_router(chat_stream.router, prefix="", tags=["chat-stream"])
api_router.include_router(conversations.router, prefix="", tags=["conversations"])
