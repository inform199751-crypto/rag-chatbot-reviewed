"""Persisted conversations.

The point of separating the transcript from the prompt window is that the two have opposite
limits: the window is capped because every turn in it costs context, the transcript is not
because the user wants to scroll back. These tests hold that separation in place.
"""

import json

import pytest
from services.conversation_store import ConversationStore
from sqlmodel import Session, SQLModel, create_engine, select

from models.conversation import ChatMessage, Conversation


@pytest.fixture(name="store")
def store_fixture():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield ConversationStore(session)


def test_append_turn_stores_both_sides(store):
    store.append_turn("c1", "what is X?", "X is Y")

    messages = store.messages("c1")
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == "what is X?"
    assert messages[1].content == "X is Y"


def test_first_question_becomes_the_title(store):
    store.append_turn("c1", "how do I reindex?", "like this")
    store.append_turn("c1", "and the second one?", "like that")

    conversation = store._session.get(Conversation, "c1")
    # Set once, from the first message -- a title that follows the latest question would rename
    # the thread every turn.
    assert conversation.title == "how do I reindex?"


def test_transcript_is_not_capped(store):
    for i in range(20):
        store.append_turn("c1", f"q{i}", f"a{i}")

    assert len(store.messages("c1")) == 40, "the transcript keeps everything"


def test_prompt_window_is_capped(store):
    for i in range(20):
        store.append_turn("c1", f"q{i}", f"a{i}")

    window = store.load_prompt_window("c1", total_length=2)
    assert len(window) == 2, "the window keeps only what the prompt can afford"
    # The tail, not the head: a window of the oldest turns is the opposite of useful.
    assert "q19" in window[-1]
    assert "q18" in window[0]


def test_prompt_window_of_an_unknown_conversation_is_empty(store):
    assert list(store.load_prompt_window("never-seen", total_length=2)) == []


def test_prompt_window_pairs_questions_with_their_answers(store):
    store.append_turn("c1", "first", "one")
    store.append_turn("c1", "second", "two")

    window = store.load_prompt_window("c1", total_length=5)
    assert window[0] == "question: first, answer: one"
    assert window[1] == "question: second, answer: two"


def test_grounding_and_sources_survive_a_round_trip(store):
    sources = [{"score": 0.9, "document": "/docs/a.md", "content_preview": "..."}]
    store.append_turn("c1", "q", "a", grounded=False, sources=sources)

    assistant = store.messages("c1")[1]
    # Kept because a reloaded thread has to render the same as the live one did. An ungrounded
    # answer that comes back looking grounded is the failure this project fixed in the stream.
    assert assistant.grounded is False
    assert json.loads(assistant.sources_json) == sources


def test_conversations_are_isolated(store):
    store.append_turn("a", "qa", "aa")
    store.append_turn("b", "qb", "ab")

    assert len(store.messages("a")) == 2
    assert all(m.conversation_id == "a" for m in store.messages("a"))


def test_clear_removes_the_thread_and_its_messages(store):
    store.append_turn("c1", "q", "a")
    store.clear("c1")

    assert store.messages("c1") == []
    assert store._session.get(Conversation, "c1") is None
    assert store._session.exec(select(ChatMessage)).all() == []


def test_list_conversations_is_most_recent_first(store):
    store.append_turn("old", "q", "a")
    store.append_turn("new", "q", "a")
    store.append_turn("old", "q2", "a2")  # touching it moves it back to the front

    assert [c.conversation_id for c in store.list_conversations()][0] == "old"
