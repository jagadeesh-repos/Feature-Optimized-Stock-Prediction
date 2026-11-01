"""Persist and retrieve completed analysis results independently of chat state."""

from datetime import date
from typing import Any

from sqlalchemy import select

from app.models.database_models import (
    AnalysisReport,
    Conversation,
    ConversationReport,
    db,
    utc_now,
)


def persist_analysis_report(
    result: dict[str, Any],
    *,
    start_date: date,
    end_date: date,
    conversation_thread_id: str | None = None,
) -> AnalysisReport:
    """Store the deterministic analysis output and optionally link a conversation."""
    conversation = None
    if conversation_thread_id is not None:
        conversation = db.session.scalar(
            select(Conversation).where(Conversation.thread_id == conversation_thread_id)
        )
        if conversation is None:
            raise ValueError("conversation thread_id does not exist")

    models = result.get("models", {})
    evaluations = {
        name: model_result.get("metrics", {}) for name, model_result in models.items()
    }
    predictions = {
        "test_dates": result.get("test_dates", []),
        "actual_test_drawdowns": result.get("actual_test_drawdowns", []),
        "actual_test_crisis_events": result.get("actual_test_crisis_events", []),
        "market_history": result.get("market_history", {}),
        "models": {
            name: {
                "test_predictions": model_result.get("test_predictions", []),
                "test_predicted_crisis_events": model_result.get(
                    "test_predicted_crisis_events", []
                ),
                "latest_prediction_date": result.get("latest_prediction_date"),
                "latest_predicted_drawdown": model_result.get(
                    "latest_predicted_drawdown"
                ),
                "latest_predicted_crisis": model_result.get("latest_predicted_crisis"),
            }
            for name, model_result in models.items()
        },
    }
    metadata = {
        "target_definition": result.get("target_definition"),
        "split": result.get("split", {}),
        "feature_selection": result.get("feature_selection", {}),
        "model_parameters": {
            name: model_result.get("metadata", {})
            for name, model_result in models.items()
        },
    }
    report = AnalysisReport(
        ticker=result["ticker"],
        start_date=start_date,
        end_date=end_date,
        crisis_threshold=result["crisis_threshold"],
        prediction_horizon=result["prediction_horizon"],
        selected_model=result["selected_model"],
        xgboost_results=models.get("xgboost"),
        dnn_results=models.get("dnn"),
        evaluation_metrics=evaluations,
        selected_features=result.get("selected_features", []),
        predictions=predictions,
        analysis_summary=(
            "Regression analysis completed. Forecast values estimate future drawdown; "
            "they are not calibrated crisis probabilities."
        ),
        model_metadata=metadata,
    )
    db.session.add(report)
    db.session.flush()
    if conversation is not None:
        db.session.add(
            ConversationReport(
                thread_id=conversation.thread_id,
                report_id=report.report_id,
            )
        )
        conversation.updated_at = utc_now()
    db.session.commit()
    return report


def report_to_dict(report: AnalysisReport) -> dict[str, Any]:
    """Serialize report fields without exposing local paths or credentials."""
    stored_predictions = report.predictions or {}
    prediction_models = stored_predictions.get("models", {})
    model_results: dict[str, Any] = {}
    for model_name, model_result in (
        ("xgboost", report.xgboost_results),
        ("dnn", report.dnn_results),
    ):
        if model_result is not None:
            model_results[model_name] = {
                **model_result,
                **prediction_models.get(model_name, {}),
            }

    return {
        "report_id": report.report_id,
        "ticker": report.ticker,
        "start_date": report.start_date.isoformat(),
        "end_date": report.end_date.isoformat(),
        "crisis_threshold": report.crisis_threshold,
        "prediction_horizon": report.prediction_horizon,
        "selected_model": report.selected_model,
        "xgboost_results": report.xgboost_results,
        "dnn_results": report.dnn_results,
        "evaluation_metrics": report.evaluation_metrics,
        "selected_features": report.selected_features,
        "predictions": report.predictions,
        "analysis_summary": report.analysis_summary,
        "model_metadata": report.model_metadata,
        "models": model_results,
        "latest_prediction_date": next(
            (
                values.get("latest_prediction_date")
                for values in prediction_models.values()
                if values.get("latest_prediction_date")
            ),
            None,
        ),
        "created_at": report.created_at.isoformat() if report.created_at else None,
    }


def get_report(report_id: str) -> AnalysisReport | None:
    """Return one report by its public opaque identifier."""
    return db.session.scalar(
        select(AnalysisReport).where(AnalysisReport.report_id == report_id)
    )


def get_reports(limit: int = 50) -> list[AnalysisReport]:
    """Return recent reports, bounded to prevent unbounded response sizes."""
    return list(
        db.session.scalars(
            select(AnalysisReport)
            .order_by(AnalysisReport.created_at.desc())
            .limit(limit)
        )
    )


def compare_reports(first: AnalysisReport, second: AnalysisReport) -> dict[str, Any]:
    """Compare persisted metrics and inputs without recomputing model outputs."""
    first_metrics = first.evaluation_metrics or {}
    second_metrics = second.evaluation_metrics or {}
    comparison: dict[str, Any] = {}
    for model_name in sorted(set(first_metrics).intersection(second_metrics)):
        shared_metrics = set(first_metrics[model_name]).intersection(
            second_metrics[model_name]
        )
        comparison[model_name] = {
            metric: {
                "first": first_metrics[model_name][metric],
                "second": second_metrics[model_name][metric],
                "difference_second_minus_first": (
                    second_metrics[model_name][metric]
                    - first_metrics[model_name][metric]
                ),
            }
            for metric in sorted(shared_metrics)
        }
    return {
        "first_report_id": first.report_id,
        "second_report_id": second.report_id,
        "same_ticker": first.ticker == second.ticker,
        "tickers": [first.ticker, second.ticker],
        "metrics": comparison,
    }
