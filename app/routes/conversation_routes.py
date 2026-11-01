"""REST endpoints for stable conversation IDs and linked report retrieval."""

from flask import Blueprint, current_app, jsonify

from app.models.database_models import Conversation, ConversationReport, db
from app.agent.graph import conversation_messages
from app.services.report_service import get_report, report_to_dict

conversation_routes = Blueprint("conversations_api", __name__, url_prefix="/api")


@conversation_routes.post("/conversations")
def create_conversation():
    """Create a conversation identity used by the later chat/checkpoint layer."""
    conversation = Conversation()
    db.session.add(conversation)
    db.session.commit()
    return (
        jsonify(
            {
                "conversation": {
                    "thread_id": conversation.thread_id,
                    "created_at": conversation.created_at.isoformat(),
                    "updated_at": conversation.updated_at.isoformat(),
                }
            }
        ),
        201,
    )


@conversation_routes.get("/conversations")
def list_conversations():
    """List conversation IDs without exposing checkpoint database paths."""
    conversations = db.session.execute(
        db.select(Conversation).order_by(Conversation.updated_at.desc()).limit(100)
    ).scalars()
    return jsonify(
        {
            "conversations": [
                {
                    "thread_id": item.thread_id,
                    "created_at": item.created_at.isoformat(),
                    "updated_at": item.updated_at.isoformat(),
                }
                for item in conversations
            ]
        }
    )


@conversation_routes.get("/conversations/<thread_id>")
def conversation_detail(thread_id: str):
    """Return one conversation and reports explicitly linked to its thread."""
    conversation = db.session.execute(
        db.select(Conversation).where(Conversation.thread_id == thread_id)
    ).scalar_one_or_none()
    if conversation is None:
        return jsonify({"error": "Conversation not found"}), 404

    links = db.session.execute(
        db.select(ConversationReport).where(ConversationReport.thread_id == thread_id)
    ).scalars()
    reports = [
        report_to_dict(report)
        for link in links
        if (report := get_report(link.report_id)) is not None
    ]
    return jsonify(
        {
            "conversation": {
                "thread_id": conversation.thread_id,
                "created_at": conversation.created_at.isoformat(),
                "updated_at": conversation.updated_at.isoformat(),
            },
            "reports": reports,
            "messages": conversation_messages(
                current_app.extensions["agent_graph"], thread_id
            ),
        }
    )
