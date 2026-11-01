# Stock Crisis Prediction

A local Flask application for historical Indian-market OHLCV analysis, report
storage, and report-grounded assistant conversations. The project is based on
the notebook supplied by the user and references Naik and Mohan, IEEE Access
(2021), DOI [10.1109/ACCESS.2021.3088999](https://doi.org/10.1109/ACCESS.2021.3088999).

## Run locally

Run these commands from the project root in PowerShell. All packages are
installed inside the project virtual environment, not system Python.

```powershell
py -m venv .venv                 # Only if .venv does not already exist
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
python run.py
```

The current workspace can also be started without activation:

```powershell
& .\.venv\Scripts\python.exe run.py
```

Open `http://127.0.0.1:5000`. The app creates its report database and separate
LangGraph checkpoint database under `instance/` on startup. Stop the local
development server with Ctrl+C.

## Pages and APIs

- `/`: Dashboard for ticker/date analysis, required drawdown threshold/horizon,
	XGBoost, PyTorch DNN, or both.
- `/reports`: persisted report history.
- `/assistant`: persistent LangGraph conversation interface.
- `/about`: target, evaluation, and research limitations.
- `/health`: readiness check.
- `POST /api/analyze`: fetch data, train/evaluate, and persist a report.
- `GET /api/reports` and `GET /api/reports/<report_id>`: report retrieval.
- `GET /api/reports/<report_id>/metrics` and `/predictions`: stored outputs.
- `POST /api/reports/compare`: compare two saved reports.
- `GET/POST /api/conversations` and `GET /api/conversations/<thread_id>`:
	conversation identities, checkpointed messages, and linked reports.
- `POST /api/chat`: submit a message to the report-grounded assistant.

Yahoo Finance access is needed for a real analysis. Unit tests use synthetic
fixtures and do not call Yahoo Finance or Groq.

## Environment settings

`.env` is optional and ignored by Git. `DATABASE_URL` selects report storage;
the default is SQLite under `instance/`. `LANGGRAPH_CHECKPOINT_DATABASE` selects
the separate checkpoint file. `SECRET_KEY` can be set for a stable local key.
`GROQ_API_KEY` and `GROQ_MODEL` enable live Groq tool calling. Without a key, chat
uses a clearly labeled fallback that only reports values from reports linked to
the active thread. Never commit real keys.

## Tests

Run the complete deterministic suite with the project interpreter:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

## Analysis behavior

The user-supplied notebook defines a regression baseline. For each session the
target is the maximum non-negative decline from that session's close to the
lowest low over a caller-specified number of subsequent observed sessions. The
horizon and threshold are required inputs; the app does not assume illustrative
10%/21-session defaults. A threshold flag derived from a drawdown estimate is
not a calibrated crisis probability.

Features are causal SMA, EMA, MACD, RSI, rolling volatility, volume change, and
returns. Feature selection uses XGBoost-backed RFE plus Boruta with a random
forest, selecting the intersection and falling back to the union. XGBoost uses
the supplied notebook's 500 estimators and 0.05 learning rate. The PyTorch DNN
uses 128/64 ReLU layers, 0.2/0.1 dropout, MSE, Adam at 0.001, and validation-based
early stopping. Scaling and feature selection are fit on training data only.
Chronological split boundaries are 70% and 85%; horizon-sized gaps are purged
around them. Validation controls early stopping; test rows are reserved for
MSE, MAE, and RMSE.

These are project choices based on the supplied notebook and brief, not verified
paper settings. The accessible paper abstract describes XGBoost/DNN regression
and MSE/MAE/RMSE but does not establish exact stock labels, dataset constituents,
HFS configuration, or model parameters. The full text was inaccessible during
implementation, so this application does not claim exact paper reproduction.

## Deployment scope

This is a local, single-user research application. It has no authentication and
uses Flask's development server; do not expose it directly to the public
internet. Configure a production WSGI server, authentication, and access controls
before deployment beyond a trusted local machine.