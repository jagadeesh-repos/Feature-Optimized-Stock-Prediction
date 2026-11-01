"""LangChain tools backed by actual analysis services and thread-linked reports."""

import json
from datetime import date
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from sqlalchemy import select

from app.models.database_models import AnalysisReport, ConversationReport, db
from app.services.analysis_service import run_analysis
from app.services.report_service import (
    compare_reports as compare_stored_reports,
    get_report,
    persist_analysis_report,
    report_to_dict,
)


def _linked_report(report_id: str, thread_id: str) -> AnalysisReport | None:
    """Return a report only when it belongs to the active conversation thread."""
    linked = db.session.scalar(
        select(ConversationReport.id).where(
            ConversationReport.report_id == report_id,
            ConversationReport.thread_id == thread_id,
        )
    )
    return get_report(report_id) if linked is not None else None


@tool
def analyze_stock(
    ticker: str,
    start_date: str,
    end_date: str,
    prediction_horizon: int,
    crisis_threshold: float,
    selected_model: str = "both",
    thread_id: Annotated[str, InjectedState("thread_id")] = "",
) -> str:
    """Run actual stock analysis with explicit dates, horizon, and threshold."""
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        result = run_analysis(
            ticker,
            start,
            end,
            prediction_horizon=prediction_horizon,
            crisis_threshold=crisis_threshold,
            selected_model=selected_model,
        )
        report = persist_analysis_report(
            result,
            start_date=start,
            end_date=end,
            conversation_thread_id=thread_id,
        )
        return json.dumps(report_to_dict(report), allow_nan=False)
    except Exception as error:
        return json.dumps({"error": str(error)})


@tool
def get_analysis_report(
    report_id: str,
    thread_id: Annotated[str, InjectedState("thread_id")] = "",
) -> str:
    """Retrieve a saved report linked to this conversation."""
    report = _linked_report(report_id, thread_id)
    if report is None:
        return json.dumps(
            {"error": "No report with that ID is linked to this conversation"}
        )
    return json.dumps(report_to_dict(report), allow_nan=False)


@tool
def compare_reports(
    report_id_1: str,
    report_id_2: str,
    thread_id: Annotated[str, InjectedState("thread_id")] = "",
) -> str:
    """Compare metrics from two reports linked to this conversation."""
    first = _linked_report(report_id_1, thread_id)
    second = _linked_report(report_id_2, thread_id)
    if first is None or second is None:
        return json.dumps({"error": "Both reports must be linked to this conversation"})
    return json.dumps(compare_stored_reports(first, second), allow_nan=False)


@tool
def get_feature_importance(
    report_id: str,
    thread_id: Annotated[str, InjectedState("thread_id")] = "",
) -> str:
    """Return XGBoost feature importance stored in a linked report."""
    report = _linked_report(report_id, thread_id)
    if report is None:
        return json.dumps(
            {"error": "No report with that ID is linked to this conversation"}
        )
    xgboost_results = report.xgboost_results or {}
    importance = xgboost_results.get("feature_importance")
    if importance is None:
        return json.dumps(
            {"message": "This report has no XGBoost feature-importance values."}
        )
    return json.dumps(
        {"report_id": report_id, "feature_importance": importance},
        allow_nan=False,
    )


@tool
def get_historical_predictions(
    report_id: str,
    thread_id: Annotated[str, InjectedState("thread_id")] = "",
) -> str:
    """Return actual held-out outcomes and predictions from a linked report."""
    report = _linked_report(report_id, thread_id)
    if report is None:
        return json.dumps(
            {"error": "No report with that ID is linked to this conversation"}
        )
    return json.dumps(
        {"report_id": report_id, "predictions": report.predictions},
        allow_nan=False,
    )


def available_tools() -> list[Any]:
    """Expose the supported, deterministic application tools to LangGraph."""
    return [
        analyze_stock,
        get_analysis_report,
        compare_reports,
        get_feature_importance,
        get_historical_predictions,
    ]
