"""Tests for the WebSocket chat streaming endpoint.

These need a live llama.cpp server serving the model named in `conftest.model_settings`; the
fixture builds a real client rather than a stub. What they assert is the wire protocol, not the
content of the reply -- the text depends on the model, the event sequence does not.

The frames used to be bare strings, so a test could only count them. Now every frame is a typed
event, which lets these read as a protocol contract: an answer is bracketed by ANSWER_START and
DONE, DONE arrives exactly once, and nothing follows it.
"""

import json

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

# Bounds the read so a protocol bug fails the test instead of hanging it. The previous version
# looped a fixed 10 receives, which blocks forever whenever a response is shorter than that --
# including every response that errors early.
MAX_FRAMES = 500


def drain(websocket) -> list[dict]:
    """Read frames until DONE, which the server sends on every terminal path."""
    events = []
    for _ in range(MAX_FRAMES):
        event = json.loads(websocket.receive_text())
        events.append(event)
        if event["type"] == "done":
            return events
    raise AssertionError(f"No 'done' event within {MAX_FRAMES} frames")


def assert_valid_stream(events: list[dict]) -> None:
    assert events, "expected at least one event"
    assert all("type" in e and "data" in e for e in events), "every frame carries type and data"
    assert events[-1]["type"] == "done", "the stream ends with done"
    assert [e["type"] for e in events].count("done") == 1, "done is sent exactly once"


def test_chat_stream_successful_response(client_with_overridden_deps: TestClient):
    """A plain chat request produces a well-formed event stream."""
    with client_with_overridden_deps.websocket_connect("/chat/stream") as websocket:
        websocket.send_json({"text": "Hello, how are you?"})
        events = drain(websocket)

    assert_valid_stream(events)
    assert events[0]["type"] == "answer_start"
    assert events[0]["data"]["grounded"] is False, "no retrieval ran, so nothing is grounded"


def test_chat_stream_emits_tokens(client_with_overridden_deps: TestClient):
    """The answer arrives as TOKEN events rather than as untyped frames."""
    with client_with_overridden_deps.websocket_connect("/chat/stream") as websocket:
        websocket.send_json({"text": "Say something."})
        events = drain(websocket)

    tokens = [e for e in events if e["type"] == "token"]
    assert tokens, "expected at least one token"
    assert all(isinstance(e["data"]["text"], str) for e in tokens)


def test_chat_stream_empty_message(client_with_overridden_deps: TestClient):
    """An empty prompt still terminates cleanly instead of leaving the client waiting."""
    with client_with_overridden_deps.websocket_connect("/chat/stream") as websocket:
        websocket.send_json({"text": ""})
        events = drain(websocket)

    assert_valid_stream(events)


def test_chat_stream_multiple_requests(client_with_overridden_deps: TestClient):
    """Each request on a shared connection gets its own complete stream."""
    with client_with_overridden_deps.websocket_connect("/chat/stream") as websocket:
        for i in range(3):
            websocket.send_json({"text": f"Question {i + 1}"})
            assert_valid_stream(drain(websocket))


def test_chat_stream_rag_reports_sources(client_with_overridden_deps: TestClient):
    """A RAG request reports what it retrieved -- including when it retrieved nothing."""
    with client_with_overridden_deps.websocket_connect("/chat/stream") as websocket:
        websocket.send_json({"text": "What is in the documents?", "rag": True})
        events = drain(websocket)

    assert_valid_stream(events)
    sources = [e for e in events if e["type"] == "sources"]
    assert len(sources) == 1, "sources is sent once, before the answer"
    data = sources[0]["data"]
    assert isinstance(data["grounded"], bool)
    if not data["grounded"]:
        # The reason is the point: an empty index, an off-topic question and a threshold set too
        # high are three different problems and used to share one message.
        assert data["reason"] in {"empty_index", "below_threshold", "below_rerank_threshold"}
        assert data["message"], "an ungrounded answer explains why"


def test_chat_stream_invalid_payload(client_with_overridden_deps: TestClient):
    """Test WebSocket with invalid payload."""
    with pytest.raises(ValidationError, match="validation error for ChatRequest"):
        with client_with_overridden_deps.websocket_connect("/chat/stream") as websocket:
            # Send invalid data (missing 'text' field)
            websocket.send_json({"invalid": "data"})
            websocket.receive_text()


def test_chat_stream_connection_and_disconnection(client_with_overridden_deps: TestClient):
    """Test WebSocket connection can be established and closed."""
    with client_with_overridden_deps.websocket_connect("/chat/stream") as websocket:
        websocket.send_json({"text": "Test message"})

        response = json.loads(websocket.receive_text())
        assert response["type"] in {"answer_start", "sources", "token", "error", "done"}
