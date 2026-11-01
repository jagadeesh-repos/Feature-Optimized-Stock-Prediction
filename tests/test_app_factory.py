"""Focused smoke tests for app startup and initial database schema creation."""

from pathlib import Path

from sqlalchemy import UniqueConstraint

from app import create_app
from app.models.database_models import (
    AnalysisReport,
    Conversation,
    ConversationReport,
    db,
)


def test_app_factory_creates_database_and_health_endpoint(tmp_path: Path):
    """The app must boot against an isolated SQLite file and register its tables."""
    database_path = tmp_path / "phase1.sqlite3"

    class TemporaryConfig:
        TESTING = True
        SECRET_KEY = "test-only"
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{database_path.as_posix()}"
        SQLALCHEMY_TRACK_MODIFICATIONS = False

    app = create_app(TemporaryConfig)
    client = app.test_client()

    assert client.get("/health").get_json() == {"status": "ok"}
    assert b"Stock Crisis Prediction" in client.get("/").data
    assert client.get("/reports").status_code == 200
    assert client.get("/about").status_code == 200
    assert client.get("/assistant").status_code == 200
    with app.app_context():
        assert {"analysis_report", "conversation", "conversation_report"}.issubset(
            set(db.metadata.tables)
        )
        assert AnalysisReport.__tablename__ == "analysis_report"
        assert Conversation.__tablename__ == "conversation"
        assert ConversationReport.__tablename__ == "conversation_report"
    assert database_path.exists()

    # create_all() must tolerate app recreation against an existing database.
    second_app = create_app(TemporaryConfig)
    with second_app.app_context():
        assert db.session.execute(db.select(Conversation)).all() == []


def test_models_have_required_fields_and_relationship_constraints():
    """The report schema must preserve all requested data and stable identifiers."""
    report_fields = {
        "report_id",
        "ticker",
        "start_date",
        "end_date",
        "crisis_threshold",
        "prediction_horizon",
        "selected_model",
        "xgboost_results",
        "dnn_results",
        "evaluation_metrics",
        "selected_features",
        "predictions",
        "analysis_summary",
        "model_metadata",
        "created_at",
    }
    assert report_fields.issubset(AnalysisReport.__table__.columns.keys())
    assert AnalysisReport.__table__.c.report_id.unique
    assert not AnalysisReport.__table__.c.report_id.nullable

    conversation_fields = {"thread_id", "created_at", "updated_at"}
    assert conversation_fields.issubset(Conversation.__table__.columns.keys())
    assert Conversation.__table__.c.thread_id.unique
    assert not Conversation.__table__.c.thread_id.nullable

    conversation_report_fields = {"id", "thread_id", "report_id", "created_at"}
    assert conversation_report_fields.issubset(
        ConversationReport.__table__.columns.keys()
    )
    foreign_keys = {
        (foreign_key.parent.name, foreign_key.target_fullname)
        for foreign_key in ConversationReport.__table__.foreign_keys
    }
    assert foreign_keys == {
        ("thread_id", "conversation.thread_id"),
        ("report_id", "analysis_report.report_id"),
    }
    unique_pairs = {
        tuple(column.name for column in constraint.columns)
        for constraint in ConversationReport.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    assert ("thread_id", "report_id") in unique_pairs


def test_relative_sqlite_url_uses_flask_instance_directory(tmp_path: Path):
    """The documented sqlite:/// URL creates its DB under the instance folder."""

    class RelativeDatabaseConfig:
        TESTING = True
        SECRET_KEY = "test-only"
        SQLALCHEMY_DATABASE_URI = "sqlite:///relative.sqlite3"
        SQLALCHEMY_TRACK_MODIFICATIONS = False
        LANGGRAPH_CHECKPOINT_DATABASE = str(tmp_path / "checkpoints.sqlite3")

    app = create_app(RelativeDatabaseConfig)
    expected_database = Path(app.instance_path) / "relative.sqlite3"

    assert expected_database.exists()
