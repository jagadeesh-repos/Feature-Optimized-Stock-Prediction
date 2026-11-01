"""SQLAlchemy models for durable analysis reports and conversation links.

Report data is stored separately from LangGraph checkpoints so reports remain
available even if conversation state is reset or the agent is unavailable.
"""

from datetime import datetime, timezone
from uuid import uuid4

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import UniqueConstraint

db = SQLAlchemy()


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for persisted records."""
    return datetime.now(timezone.utc)


class AnalysisReport(db.Model):
    """Persisted inputs and outputs for one completed stock analysis."""

    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(
        db.String(36), unique=True, nullable=False, default=lambda: str(uuid4())
    )
    ticker = db.Column(db.String(32), nullable=False, index=True)
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date, nullable=False)
    crisis_threshold = db.Column(db.Float, nullable=False)
    prediction_horizon = db.Column(db.Integer, nullable=False)
    selected_model = db.Column(db.String(16), nullable=False)
    xgboost_results = db.Column(db.JSON, nullable=True)
    dnn_results = db.Column(db.JSON, nullable=True)
    evaluation_metrics = db.Column(db.JSON, nullable=True)
    selected_features = db.Column(db.JSON, nullable=True)
    predictions = db.Column(db.JSON, nullable=True)
    analysis_summary = db.Column(db.Text, nullable=True)
    model_metadata = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)


class Conversation(db.Model):
    """Stable application-level identity for one agent conversation."""

    id = db.Column(db.Integer, primary_key=True)
    thread_id = db.Column(
        db.String(36), unique=True, nullable=False, default=lambda: str(uuid4())
    )
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)


class ConversationReport(db.Model):
    """Associate saved reports with the conversation that produced them."""

    id = db.Column(db.Integer, primary_key=True)
    thread_id = db.Column(
        db.String(36),
        db.ForeignKey("conversation.thread_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    report_id = db.Column(
        db.String(36),
        db.ForeignKey("analysis_report.report_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)

    __table_args__ = (
        UniqueConstraint("thread_id", "report_id", name="uq_conversation_report"),
    )
