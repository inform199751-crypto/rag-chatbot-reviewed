"""The typed events a chat stream sends over the WebSocket.

What this replaces
    Every frame used to be a bare string, and the client appended all of them to the current
    message. A retrieved-source block, the "**Answer:**" heading, a generated token and
    "Error during streaming." were indistinguishable once they arrived -- so an error rendered
    as if the model had said it, and sources could not be styled apart from prose.

    Worse, nothing marked the end. The client inferred it from silence: 500 ms without a frame
    meant done. That is a race against generation speed, not a protocol. At the ~7 tokens/second
    a quantised model manages on CPU, an ordinary pause between tokens crosses the threshold, so
    the UI declares the answer finished, unlocks the input, and then keeps appending tokens to a
    message it has already closed.

Design notes
    `type` is the discriminator and every payload lives under `data`, so a client can switch on
    one field and unknown event types can be skipped rather than mis-rendered -- which is what
    lets a newer server talk to an older client.

    DONE is sent on every terminal path, including errors and declined answers. A client that
    only unlocks its input on DONE would otherwise hang forever on the paths that forgot it, so
    there is exactly one rule: the stream is over when DONE arrives.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StreamEventType(str, Enum):
    SOURCES = "sources"
    """The retrieved context, sent before generation starts. Also sent when nothing was
    retrieved, carrying the reason -- the absence of context is information the user needs."""

    ANSWER_START = "answer_start"
    """Generation is about to begin. `grounded` says whether the answer that follows is backed
    by the retrieved documents or comes from the model alone."""

    TOKEN = "token"
    """One piece of generated text. The only event that should be appended to the answer."""

    DONE = "done"
    """Terminal. Sent exactly once per request, on every path including failure."""

    ERROR = "error"
    """Something went wrong. Always followed by DONE."""


class SourceItem(BaseModel):
    score: float
    document: str | None = None
    content_preview: str = ""


class SourcesData(BaseModel):
    documents: list[SourceItem] = Field(default_factory=list)
    grounded: bool = False
    """False when nothing was retrieved."""

    reason: str | None = None
    """Machine-readable cause when `grounded` is false: empty_index, below_threshold,
    below_rerank_threshold. Lets a client offer the right next step -- upload a file, rephrase,
    or tell an operator -- instead of one generic apology."""

    message: str = ""
    """Human-readable version of the same thing."""


class AnswerStartData(BaseModel):
    grounded: bool


class TokenData(BaseModel):
    text: str


class DoneData(BaseModel):
    took_seconds: float | None = None
    declined: bool = False
    """True when the server chose not to answer because it had no supporting documents. The
    client should not present that as a model failure -- it is the configured behaviour."""


class ErrorData(BaseModel):
    message: str


class StreamEvent(BaseModel):
    """One frame on the wire."""

    type: StreamEventType
    data: dict[str, Any] = Field(default_factory=dict)


def _event(event_type: StreamEventType, payload: BaseModel | None = None) -> str:
    return StreamEvent(
        type=event_type,
        data=payload.model_dump() if payload else {},
    ).model_dump_json()


def sources_event(
    documents: list[dict[str, Any]],
    grounded: bool,
    reason: str | None = None,
    message: str = "",
) -> str:
    return _event(
        StreamEventType.SOURCES,
        SourcesData(
            documents=[SourceItem(**doc) for doc in documents],
            grounded=grounded,
            reason=reason,
            message=message,
        ),
    )


def answer_start_event(grounded: bool) -> str:
    return _event(StreamEventType.ANSWER_START, AnswerStartData(grounded=grounded))


def token_event(text: str) -> str:
    return _event(StreamEventType.TOKEN, TokenData(text=text))


def done_event(took_seconds: float | None = None, declined: bool = False) -> str:
    return _event(StreamEventType.DONE, DoneData(took_seconds=took_seconds, declined=declined))


def error_event(message: str) -> str:
    return _event(StreamEventType.ERROR, ErrorData(message=message))


__all__ = [
    "AnswerStartData",
    "DoneData",
    "ErrorData",
    "SourceItem",
    "SourcesData",
    "StreamEvent",
    "StreamEventType",
    "TokenData",
    "answer_start_event",
    "done_event",
    "error_event",
    "sources_event",
    "token_event",
]
