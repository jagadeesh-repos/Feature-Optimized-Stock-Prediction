"""REST endpoints for saved reports, predictions, metrics, and comparisons."""

from flask import Blueprint, jsonify, request

from app.services.report_service import (
    compare_reports,
    get_report,
    get_reports,
    report_to_dict,
)

report_routes = Blueprint("reports_api", __name__, url_prefix="/api/reports")


@report_routes.get("")
def list_reports():
    """List recent saved reports with a bounded optional page size."""
    raw_limit = request.args.get("limit", "50")
    try:
        limit = int(raw_limit)
    except ValueError:
        return jsonify({"error": "limit must be an integer from 1 to 100"}), 400
    if not 1 <= limit <= 100:
        return jsonify({"error": "limit must be an integer from 1 to 100"}), 400
    return jsonify(
        {"reports": [report_to_dict(report) for report in get_reports(limit)]}
    )


@report_routes.get("/<report_id>")
def report_detail(report_id: str):
    """Return the complete persisted report by report_id."""
    report = get_report(report_id)
    if report is None:
        return jsonify({"error": "Report not found"}), 404
    return jsonify({"report": report_to_dict(report)})


@report_routes.get("/<report_id>/metrics")
def report_metrics(report_id: str):
    """Return stored metrics for a report."""
    report = get_report(report_id)
    if report is None:
        return jsonify({"error": "Report not found"}), 404
    return jsonify(
        {"report_id": report.report_id, "metrics": report.evaluation_metrics}
    )


@report_routes.get("/<report_id>/predictions")
def report_predictions(report_id: str):
    """Return stored historical-test and latest forecast output."""
    report = get_report(report_id)
    if report is None:
        return jsonify({"error": "Report not found"}), 404
    return jsonify({"report_id": report.report_id, "predictions": report.predictions})


@report_routes.post("/compare")
def compare_saved_reports():
    """Compare two existing reports using their stored metrics and inputs."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400
    first_id = payload.get("report_id_1")
    second_id = payload.get("report_id_2")
    if not isinstance(first_id, str) or not isinstance(second_id, str):
        return jsonify({"error": "report_id_1 and report_id_2 are required"}), 400
    first = get_report(first_id)
    second = get_report(second_id)
    if first is None or second is None:
        return jsonify({"error": "Both reports must exist"}), 404
    return jsonify({"comparison": compare_reports(first, second)})
