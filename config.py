"""Environment-backed configuration for the Flask application.

Paths and secrets are resolved at runtime so local development does not need
system-wide configuration or checked-in credentials.
"""

import os
import secrets
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent


def load_environment(env_file: Path = PROJECT_ROOT / ".env") -> None:
    """Load local settings while preserving values already set by the process."""
    load_dotenv(env_file, override=False)


load_environment()


class Config:
    """Base settings shared by local development and tests."""

    SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_urlsafe(32)
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{(PROJECT_ROOT / 'instance' / 'stock_crisis.sqlite3').as_posix()}",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    LANGGRAPH_CHECKPOINT_DATABASE = os.getenv(
        "LANGGRAPH_CHECKPOINT_DATABASE",
        str(PROJECT_ROOT / "instance" / "langgraph_checkpoints.sqlite3"),
    )


class TestConfig(Config):
    """Isolated settings for tests; callers should supply a temporary database."""

    TESTING = True
