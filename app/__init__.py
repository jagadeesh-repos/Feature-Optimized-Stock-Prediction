"""Flask application factory and shared extension initialization."""

from pathlib import Path
import sqlite3

from flask import Flask, jsonify

from config import Config
from app.models.database_models import db


def create_app(config_object: type = Config) -> Flask:
    """Create and configure the Flask application.

    The factory keeps tests isolated and lets deployment code choose its own
    configuration without mutating process-wide application state.
    """
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_object)

    # SQLite's parent directory is not created by the SQLite driver.
    database_uri = app.config.get("SQLALCHEMY_DATABASE_URI", "")
    if database_uri.startswith("sqlite:///") and ":memory:" not in database_uri:
        database_path = Path(database_uri.removeprefix("sqlite:///"))
        if not database_path.is_absolute():
            database_path = Path(app.instance_path) / database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)

    db.init_app(app)

    from app.routes.analysis_routes import analysis_routes
    from app.routes.chat_routes import chat_routes
    from app.routes.conversation_routes import conversation_routes
    from app.routes.main_routes import main
    from app.routes.report_routes import report_routes

    app.register_blueprint(main)
    app.register_blueprint(analysis_routes)
    app.register_blueprint(chat_routes)
    app.register_blueprint(report_routes)
    app.register_blueprint(conversation_routes)

    @app.get("/health")
    def health_check():
        """Expose a small readiness endpoint for local checks and deployment."""
        return jsonify({"status": "ok"})

    with app.app_context():
        # Importing models before create_all registers every table on the shared
        # metadata object. Migrations can replace this bootstrap in a later phase.
        from app.models import database_models  # noqa: F401

        db.create_all()

    from app.agent.graph import build_agent_graph
    from langgraph.checkpoint.sqlite import SqliteSaver

    configured_checkpoint = app.config.get("LANGGRAPH_CHECKPOINT_DATABASE")
    if configured_checkpoint:
        checkpoint_path = Path(configured_checkpoint)
    elif database_uri.startswith("sqlite:///") and ":memory:" not in database_uri:
        report_database_path = Path(database_uri.removeprefix("sqlite:///"))
        if not report_database_path.is_absolute():
            report_database_path = Path(app.instance_path) / report_database_path
        checkpoint_path = report_database_path.parent / "langgraph_checkpoints.sqlite3"
    else:
        checkpoint_path = Path(app.instance_path) / "langgraph_checkpoints.sqlite3"
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(app.root_path).parent / checkpoint_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_connection = sqlite3.connect(
        checkpoint_path, check_same_thread=False, timeout=30
    )
    checkpoint_connection.execute("PRAGMA journal_mode=WAL")
    checkpointer = SqliteSaver(checkpoint_connection)
    checkpointer.setup()
    app.extensions["agent_checkpoint_connection"] = checkpoint_connection
    app.extensions["agent_graph"] = build_agent_graph(
        checkpointer,
        groq_api_key=app.config.get("GROQ_API_KEY"),
        model_name=app.config.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
    )

    return app
