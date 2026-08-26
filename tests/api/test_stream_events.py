"""The wire format of the chat stream.

Unlike test_chat_stream.py these need no server: they check the shape of what goes on the wire,
which is the half of the protocol a client depends on and the half that used not to exist. Every
frame was a bare string, so there was nothing here to test -- and nothing stopping a change to
the format from reaching a client unnoticed.
"""

import json

from schemas.stream_events import (
    StreamEventType,
    answer_start_event,
    done_event,
    error_event,
    sources_event,
    token_event,
)


def parse(raw: str) -> dict:
    event = json.loads(raw)
    assert set(event) == {"type", "data"}, "every frame is exactly type + data"
    return event


def test_every_event_carries_a_known_type():
    frames = [
        sources_event([], grounded=False),
        answer_start_event(grounded=True),
        token_event("hi"),
        done_event(took_seconds=1.0),
        error_event("boom"),
    ]
    types = {parse(f)["type"] for f in frames}
    assert types == {e.value for e in StreamEventType}, "the helpers cover every declared type"


def test_token_event_carries_the_text_verbatim():
    # Whitespace matters: the client concatenates these, so a helper that trimmed would silently
    # run words together.
    event = parse(token_event(" world\n"))
    assert event["type"] == "token"
    assert event["data"]["text"] == " world\n"


def test_sources_event_reports_documents_and_scores():
    event = parse(
        sources_event(
            [{"score": 0.42, "document": "/docs/a.md", "content_preview": "..."}],
            grounded=True,
        )
    )
    assert event["data"]["grounded"] is True
    assert event["data"]["reason"] is None
    assert event["data"]["documents"][0]["score"] == 0.42
    assert event["data"]["documents"][0]["document"] == "/docs/a.md"


def test_sources_event_explains_an_empty_retrieval():
    """The three ways retrieval comes back empty stay distinguishable on the wire."""
    for reason in ("empty_index", "below_threshold", "below_rerank_threshold"):
        event = parse(sources_event([], grounded=False, reason=reason, message="why"))
        assert event["data"]["grounded"] is False
        assert event["data"]["reason"] == reason
        assert event["data"]["message"] == "why"
        assert event["data"]["documents"] == []


def test_answer_start_marks_whether_the_answer_is_grounded():
    assert parse(answer_start_event(grounded=True))["data"]["grounded"] is True
    assert parse(answer_start_event(grounded=False))["data"]["grounded"] is False


def test_done_distinguishes_a_declined_answer_from_a_failed_one():
    # A declined answer is the configured behaviour, not a fault, and the client should not
    # present it as one.
    assert parse(done_event(declined=True))["data"]["declined"] is True
    assert parse(done_event())["data"]["declined"] is False


def test_done_reports_elapsed_time_when_it_has_it():
    assert parse(done_event(took_seconds=2.5))["data"]["took_seconds"] == 2.5
    assert parse(done_event())["data"]["took_seconds"] is None


def test_error_event_carries_a_message():
    assert parse(error_event("nope"))["data"]["message"] == "nope"


def test_frames_are_single_line_json():
    """One frame is one JSON object. Embedded newlines in the payload must not split it."""
    raw = token_event("line one\nline two")
    assert "\n" not in raw, "the newline is escaped inside the JSON string, not emitted raw"
    assert json.loads(raw)["data"]["text"] == "line one\nline two"
