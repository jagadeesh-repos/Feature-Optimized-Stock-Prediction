"""Offline tests for persistent LangGraph fallback and thread isolation."""

import pytest

from app import create_app
from app.models.database_models import db


@pytest.fixture
def client(tmp_path):
    """Use separate temporary files for reports and LangGraph checkpoints."""

    class TemporaryConfig:
        TESTING = True
        SECRET_KEY = "test-only"
        SQLALCHEMY_DATABASE_URI = (
            f"sqlite:///{(tmp_path / 'reports.sqlite3').as_posix()}"
        )
        LANGGRAPH_CHECKPOINT_DATABASE = str(tmp_path / "checkpoints.sqlite3")
        GROQ_API_KEY = None
        GROQ_MODEL = "llama-3.3-70b-versatile"

    app = create_app(TemporaryConfig)
    with app.app_context():
        db.drop_all()
        db.create_all()
    yield app.test_client()
    with app.app_context():
        db.session.remove()


def test_missing_groq_key_returns_labeled_fallback_and_persists_history(client):
    """No-key chat uses actual report-only fallback and checkpoints both turns."""
    created = client.post("/api/conversations").get_json()["conversation"]
    thread_id = created["thread_id"]

    response = client.post(
        "/api/chat",
        json={"thread_id": thread_id, "message": "What is the crisis probability?"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["mode"] == "fallback"
    assert "Fallback mode" in payload["reply"]
    assert "cannot infer or invent" in payload["reply"]
    assert [turn["role"] for turn in payload["messages"]] == ["user", "assistant"]

    conversation = client.get(f"/api/conversations/{thread_id}").get_json()
    assert conversation["messages"] == payload["messages"]


def test_threads_do_not_share_checkpointed_messages(client):
    """Each thread retrieves only its own persisted LangGraph state."""
    first_id = client.post("/api/conversations").get_json()["conversation"]["thread_id"]
    second_id = client.post("/api/conversations").get_json()["conversation"][
        "thread_id"
    ]
    client.post(
        "/api/chat", json={"thread_id": first_id, "message": "First thread question"}
    )
    client.post(
        "/api/chat", json={"thread_id": second_id, "message": "Second thread question"}
    )

    first_messages = client.get(f"/api/conversations/{first_id}").get_json()["messages"]
    second_messages = client.get(f"/api/conversations/{second_id}").get_json()[
        "messages"
    ]

    assert "First thread question" in first_messages[0]["content"]
    assert all("Second thread" not in turn["content"] for turn in first_messages)
    assert "Second thread question" in second_messages[0]["content"]
    assert all("First thread" not in turn["content"] for turn in second_messages)


@pytest.mark.parametrize(
    "payload", [None, {}, {"message": "  "}, {"message": "x" * 4001}]
)
def test_chat_rejects_empty_or_oversized_requests(client, payload):
    """Chat validates input length and JSON shape before invoking the graph."""
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 400
