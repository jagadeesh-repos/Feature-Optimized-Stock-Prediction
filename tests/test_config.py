"""Configuration tests that do not depend on a real local secrets file."""

import os

from config import load_environment


def test_load_environment_reads_dotenv_and_preserves_process_values(
    tmp_path, monkeypatch
):
    """File settings load, while explicit process environment remains authoritative."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "STOCK_CRISIS_DOTENV_TEST=loaded-from-file\nSECRET_KEY=file-value\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("STOCK_CRISIS_DOTENV_TEST", raising=False)
    monkeypatch.setenv("SECRET_KEY", "process-value")

    load_environment(env_file)

    assert os.environ["STOCK_CRISIS_DOTENV_TEST"] == "loaded-from-file"
    assert os.environ["SECRET_KEY"] == "process-value"
