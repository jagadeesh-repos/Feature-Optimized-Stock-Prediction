"""Local development entry point: run with ``python run.py``."""

from app import create_app

app = create_app()


if __name__ == "__main__":
    # Bind to loopback only; do not expose the development server to a network.
    app.run(host="127.0.0.1", port=5000, debug=app.config.get("DEBUG", False))
