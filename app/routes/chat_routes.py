"""REST endpoint for persistent, thread-isolated LangGraph conversations."""

from flask import Blueprint, current_app, jsonify, request
from sqlalchemy import select

from app.agent.graph import conversation_messages, fallback_response
from app.models.database_models import Conversation, db, utc_now
from langchain_core.messages import HumanMessage

chat_routes = Blueprint("chat_api", __name__, url_prefix="/api")


@chat_routes.post("/chat")
def chat():
    """Append a user message, run the graph, and return its actual response."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "message is required"}), 400
    if len(message) > 4000:
        return jsonify({"error": "message must be 4000 characters or fewer"}), 400

    thread_id = payload.get("thread_id")
    if thread_id is None:
        conversation = Conversation()
        db.session.add(conversation)
        db.session.commit()
        thread_id = conversation.thread_id
    elif not isinstance(thread_id, str):
        return jsonify({"error": "thread_id must be a string"}), 400
    else:
        conversation = db.session.scalar(
            select(Conversation).where(Conversation.thread_id == thread_id)
        )
        if conversation is None:
            return jsonify({"error": "Conversation not found"}), 404

    conversation.updated_at = utc_now()
    db.session.commit()

    graph = current_app.extensions["agent_graph"]
    try:
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=message.strip())],
                "thread_id": thread_id,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        history = conversation_messages(graph, thread_id)
        reply = next(
            (
                turn["content"]
                for turn in reversed(history)
                if turn["role"] == "assistant"
            ),
            "The assistant did not return a response.",
        )
        return jsonify(
            {
                "thread_id": thread_id,
                "reply": reply,
                "mode": result.get("response_mode", "groq"),
                "messages": history,
            }
        )
    except Exception:
        current_app.logger.exception("Agent graph failed")
        reply = fallback_response(thread_id, message)
        return jsonify(
            {
                "thread_id": thread_id,
                "reply": reply,
                "mode": "fallback",
                "messages": conversation_messages(graph, thread_id),
            }
        )
