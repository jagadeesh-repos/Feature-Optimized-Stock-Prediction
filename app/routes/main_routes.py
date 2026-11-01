"""Minimal browser-facing routes for the Phase 1 application shell."""

from flask import Blueprint, jsonify, render_template

main = Blueprint("main", __name__)


@main.get("/")
def index():
    """Render the interactive stock analysis dashboard."""
    return render_template("dashboard.html")


@main.get("/reports")
def reports_page():
    """Render saved analysis reports."""
    return render_template("reports.html")


@main.get("/about")
def about_page():
    """Explain the implemented target and the limits of research verification."""
    return render_template("about.html")


@main.get("/assistant")
def assistant_page():
    """Render the persistent assistant chat interface."""
    return render_template("chat.html")
