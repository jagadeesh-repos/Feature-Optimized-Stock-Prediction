"""REST endpoint for running and saving a deterministic stock analysis."""

from datetime import date

from flask import Blueprint, current_app, jsonify, request

from app.services.analysis_service import run_analysis
from app.services.data_collection import MarketDataError
from app.services.report_service import persist_analysis_report, report_to_dict

analysis_routes = Blueprint("analysis_api", __name__, url_prefix="/api")


def _parse_iso_date(value: object, field: str) -> date:
    """Parse a required JSON ISO date with a field-specific error."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a YYYY-MM-DD string")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} must be a YYYY-MM-DD string") from error


@analysis_routes.post("/analyze")
def analyze():
    """Run models, persist their actual output, and return a report reference."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400
    try:
        ticker = payload.get("ticker")
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("ticker is required")
        start_date = _parse_iso_date(payload.get("start_date"), "start_date")
        end_date = _parse_iso_date(payload.get("end_date"), "end_date")
        if start_date > end_date:
            raise ValueError("start_date must be on or before end_date")
        horizon = payload.get("prediction_horizon")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("prediction_horizon must be a positive integer")
        threshold = payload.get("crisis_threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, (float, int)):
            raise ValueError("crisis_threshold must be a number between 0 and 1")
        if not 0 < threshold < 1:
            raise ValueError("crisis_threshold must be greater than 0 and less than 1")
        model = payload.get("selected_model", "both")
        if not isinstance(model, str) or model not in {"xgboost", "dnn", "both"}:
            raise ValueError("selected_model must be xgboost, dnn, or both")
        thread_id = payload.get("thread_id")
        if thread_id is not None and not isinstance(thread_id, str):
            raise ValueError("thread_id must be a string")

        result = run_analysis(
            ticker.strip(),
            start_date,
            end_date,
            prediction_horizon=horizon,
            crisis_threshold=float(threshold),
            selected_model=model,
        )
        report = persist_analysis_report(
            result,
            start_date=start_date,
            end_date=end_date,
            conversation_thread_id=thread_id,
        )
        return jsonify({"report": report_to_dict(report)}), 201
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except MarketDataError as error:
        return jsonify({"error": str(error)}), 502
    except Exception:
        current_app.logger.exception("Stock analysis failed")
        return (
            jsonify({"error": "Analysis failed; check the inputs and server logs"}),
            500,
        )
