"""Offline API tests using temporary SQLite databases and stubbed analysis."""

from datetime import date

import pytest

from app import create_app
from app.models.database_models import db


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Provide an isolated app and replace network/model work with fixed outputs."""
    database_path = tmp_path / "api.sqlite3"

    class TemporaryConfig:
        TESTING = True
        SECRET_KEY = "test-only"
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{database_path.as_posix()}"
        LANGGRAPH_CHECKPOINT_DATABASE = str(tmp_path / "checkpoints.sqlite3")
        SQLALCHEMY_TRACK_MODIFICATIONS = False

    import app.routes.analysis_routes as analysis_routes

    def fake_analysis(ticker, start_date, end_date, **kwargs):
        return {
            "ticker": ticker,
            "start_date": "2024-01-02",
            "end_date": "2024-03-01",
            "crisis_threshold": kwargs["crisis_threshold"],
            "prediction_horizon": kwargs["prediction_horizon"],
            "selected_model": kwargs["selected_model"],
            "selected_features": ["rsi_14"],
            "feature_selection": {"selection_rule": "test"},
            "split": {"test_rows": 2},
            "target_definition": "test drawdown",
            "test_dates": ["2024-02-28", "2024-03-01"],
            "actual_test_drawdowns": [0.02, 0.05],
            "actual_test_crisis_events": [0, 1],
            "market_history": {
                "dates": ["2024-01-02", "2024-03-01"],
                "close": [100.0, 104.0],
            },
            "latest_prediction_date": "2024-03-01",
            "models": {
                "xgboost": {
                    "metrics": {"mse": 0.01, "mae": 0.08, "rmse": 0.1},
                    "test_predictions": [0.03, 0.04],
                    "test_predicted_crisis_events": [0, 0],
                    "latest_predicted_drawdown": 0.04,
                    "latest_predicted_crisis": False,
                    "metadata": {"n_estimators": 5},
                }
            },
        }

    monkeypatch.setattr(analysis_routes, "run_analysis", fake_analysis)
    app = create_app(TemporaryConfig)
    with app.app_context():
        db.drop_all()
        db.create_all()
    yield app.test_client()
    with app.app_context():
        db.session.remove()


def _request_payload(**overrides):
    payload = {
        "ticker": "RELIANCE.NS",
        "start_date": "2024-01-01",
        "end_date": "2024-03-01",
        "prediction_horizon": 5,
        "crisis_threshold": 0.1,
        "selected_model": "xgboost",
    }
    payload.update(overrides)
    return payload


def test_create_conversation_analyze_and_retrieve_linked_report(client):
    """A report is persisted and linked to the requested stable thread ID."""
    conversation_response = client.post("/api/conversations")
    assert conversation_response.status_code == 201
    thread_id = conversation_response.get_json()["conversation"]["thread_id"]

    analysis_response = client.post(
        "/api/analyze", json=_request_payload(thread_id=thread_id)
    )
    assert analysis_response.status_code == 201
    report = analysis_response.get_json()["report"]
    assert report["ticker"] == "RELIANCE.NS"
    assert report["evaluation_metrics"]["xgboost"]["mse"] == 0.01
    assert report["models"]["xgboost"]["latest_predicted_drawdown"] == 0.04
    assert report["latest_prediction_date"] == "2024-03-01"
    assert report["predictions"]["market_history"]["close"] == [100.0, 104.0]
    assert report["predictions"]["actual_test_crisis_events"] == [0, 1]
    assert (
        report["predictions"]["models"]["xgboost"]["latest_predicted_drawdown"] == 0.04
    )

    conversation_detail = client.get(f"/api/conversations/{thread_id}")
    assert conversation_detail.status_code == 200
    assert (
        conversation_detail.get_json()["reports"][0]["report_id"] == report["report_id"]
    )

    report_detail = client.get(f"/api/reports/{report['report_id']}")
    assert report_detail.status_code == 200
    assert report_detail.get_json()["report"]["report_id"] == report["report_id"]
    assert client.get(f"/api/reports/{report['report_id']}/metrics").status_code == 200
    assert (
        client.get(f"/api/reports/{report['report_id']}/predictions").status_code == 200
    )


def test_report_listing_and_comparison_use_persisted_metrics(client):
    """Comparison reads saved report values rather than rerunning model logic."""
    first = client.post("/api/analyze", json=_request_payload()).get_json()["report"]
    second = client.post(
        "/api/analyze",
        json=_request_payload(ticker="TCS.NS", crisis_threshold=0.15),
    ).get_json()["report"]

    listing = client.get("/api/reports?limit=10")
    assert listing.status_code == 200
    assert len(listing.get_json()["reports"]) == 2

    compared = client.post(
        "/api/reports/compare",
        json={"report_id_1": first["report_id"], "report_id_2": second["report_id"]},
    )
    assert compared.status_code == 200
    comparison = compared.get_json()["comparison"]
    assert comparison["same_ticker"] is False
    assert comparison["metrics"]["xgboost"]["mse"]["difference_second_minus_first"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {"ticker": "RELIANCE.NS"},
        _request_payload(start_date="2024-03-02"),
        _request_payload(prediction_horizon=True),
        _request_payload(crisis_threshold=1.0),
        _request_payload(selected_model="unknown"),
    ],
)
def test_analysis_api_rejects_invalid_requests(client, payload):
    """Invalid JSON and analysis parameters are rejected before model execution."""
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_unknown_report_and_conversation_return_not_found(client):
    """Public identifiers do not reveal internal database details."""
    assert client.get("/api/reports/no-such-report").status_code == 404
    assert client.get("/api/conversations/no-such-thread").status_code == 404


def test_analysis_rejects_unknown_conversation(client):
    """Reports cannot be attached to a nonexistent thread."""
    response = client.post(
        "/api/analyze", json=_request_payload(thread_id="unknown-thread")
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "conversation thread_id does not exist"
