
# Stock Crisis Prediction Using XGBoost and Deep Neural Networks

> A Flask-based stock market analysis application using XGBoost, PyTorch, and an AI-powered assistant built with LangGraph and Groq.

## Project Overview

This project implements a stock crisis prediction and analysis application for historical Indian stock market data. It combines machine learning, deep learning, and an AI-powered conversational interface to analyze historical stock movements and estimate potential downside risk.

The project is inspired by the research paper:

**Novel Stock Crisis Prediction Technique – A Study on Indian Stock Market**

Naik, Nagaraj and Mohan, Biju (2021). IEEE Access.  
DOI: [10.1109/ACCESS.2021.3088999](https://doi.org/10.1109/ACCESS.2021.3088999)

The application currently follows the supplied regression notebook as its implementation baseline. It predicts future downside drawdown estimates using XGBoost and a PyTorch-based Deep Neural Network (DNN).

**Research note:** The current implementation should not be interpreted as an exact reproduction of the published paper. The paper's complete methodology and experimental configurations require independent verification.

## Key Features

- Historical stock data collection using Yahoo Finance.
- Interactive dashboard for stock ticker and date-range selection.
- Technical indicator generation and feature engineering.
- Feature selection using XGBoost-backed Recursive Feature Elimination and Boruta.
- XGBoost regression model.
- PyTorch-based Deep Neural Network.
- Model evaluation using MSE, MAE, and RMSE.
- Interactive stock analysis and visualization.
- Persistent analysis reports using SQLAlchemy and SQLite.
- AI-powered stock analysis assistant using LangGraph and Groq.
- Conversation persistence using LangGraph's SQLite checkpointer.
- Historical report retrieval and comparison.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Database | SQLite |
| ORM | Flask-SQLAlchemy |
| Machine Learning | Scikit-learn, XGBoost |
| Deep Learning | PyTorch |
| Feature Selection | RFE, Boruta |
| Stock Data | Yahoo Finance (yfinance) |
| AI Agent | LangGraph, LangChain |
| LLM Provider | Groq |
| Visualization | Chart.js |
| Testing | Pytest |

## Methodology

### 1. Data Collection and Preprocessing

Historical stock market data is collected using Yahoo Finance.

The application supports Indian stock ticker conventions, including:

- RELIANCE.NS
- TCS.NS
- INFY.NS
- HDFCBANK.NS
- ^NSEI

The dataset contains the following OHLCV attributes:

- Date
- Open
- High
- Low
- Close
- Volume

Data preprocessing includes chronological sorting, duplicate handling, missing-value handling, and dataset validation.

### 2. Crisis and Drawdown Estimation

The current implementation uses a regression-based downside drawdown target.

For each trading session, the target is calculated using the maximum non-negative decline from the current closing price to the lowest low over a user-specified number of subsequent observed trading sessions.

The prediction horizon and drawdown threshold are supplied by the user through the application.

The threshold is subsequently used to identify whether the predicted downside drawdown exceeds the specified risk level.

The illustrative values of 10% and 21 trading sessions are not assumed as fixed research-paper defaults.

A threshold-based drawdown estimate should not be interpreted as a calibrated crisis probability.

### 3. Feature Engineering

The project uses technical indicators commonly applied in financial time-series analysis:

- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Moving Average Convergence Divergence (MACD)
- Relative Strength Index (RSI)
- Rolling Volatility
- Volume Change
- Historical Returns
- Momentum Returns

Feature calculations use historical information to avoid future-data leakage.

### 4. Hybrid Feature Selection

The current implementation uses a combination of feature selection techniques:

- XGBoost-backed Recursive Feature Elimination (RFE).
- Boruta feature selection using a Random Forest estimator.
- Intersection of selected features, with a union fallback when necessary.

Feature selection is fitted using training data only.

This is the project's implemented feature-selection approach and should not be assumed to be identical to the original paper's HFS methodology without verification.

### 5. Machine Learning Model: XGBoost

An XGBoost regression model is implemented to estimate future downside drawdown.

The current configuration follows the supplied notebook baseline:

- Number of estimators: 500
- Learning rate: 0.05

The model is trained using selected features and chronological training data.

### 6. Deep Learning Model: PyTorch DNN

A fully connected Deep Neural Network is implemented using PyTorch.

Current baseline architecture:

- First hidden layer: 128 neurons.
- Second hidden layer: 64 neurons.
- ReLU activation functions.
- Dropout regularization: 0.2 and 0.1.
- Mean Squared Error (MSE) loss.
- Adam optimizer.
- Learning rate: 0.001.
- Validation-based early stopping.

The DNN produces regression-based drawdown estimates.

### 7. Model Evaluation

Both models are evaluated using:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

The dataset is split chronologically:

- Training: First 70%.
- Validation: Next 15%.
- Testing: Final 15%.

Horizon-sized gaps are purged around split boundaries to reduce leakage caused by overlapping future-label windows.

Feature selection and preprocessing are fitted using training data only. Validation data is used for model selection and early stopping, while test data is reserved for final evaluation.

## AI-Powered Stock Analysis Agent

The application includes an AI-powered conversational assistant built using LangGraph and Groq.

The agent allows users to interact with the stock analysis system using natural language.

Example requests:

- Analyze RELIANCE.NS for a specified date range.
- Explain the predicted downside drawdown.
- Compare XGBoost and DNN predictions.
- Retrieve a previously generated analysis report.
- Explain the features influencing a prediction.
- Compare two historical analysis reports.

### Agent Architecture

The agent uses:

- LangGraph for workflow orchestration.
- LangChain for LLM integration and tool calling.
- Groq API for language model inference.
- SQLite checkpointer for persistent conversation state.
- SQLAlchemy for retrieving stored analysis reports.

The agent consumes actual machine learning outputs and stored reports rather than inventing predictions or evaluation metrics.

If Groq is unavailable, the application provides a clearly identified fallback based on available stored analysis results.

## Database and Report Persistence

The application uses SQLite and Flask-SQLAlchemy.

The database supports:

### AnalysisReport

Stores:

- Unique report ID.
- Stock ticker.
- Analysis date range.
- Prediction configuration.
- XGBoost results.
- DNN results.
- Evaluation metrics.
- Selected features.
- Predictions.
- Model metadata.
- Creation timestamp.

### Conversation

Stores conversation identifiers and timestamps.

### ConversationReport

Associates generated analysis reports with their respective conversations.

Analysis reports are stored separately from LangGraph checkpoint state to ensure that historical results remain independently accessible.

## Project Structure

```text
stock_crisis_prediction/
│
├── app/
│   ├── routes/
│   ├── models/
│   ├── services/
│   ├── agent/
│   ├── templates/
│   └── static/
│
├── data/
├── saved_models/
├── reports/
├── instance/
├── tests/
│
├── config.py
├── requirements.txt
├── .env.example
├── .gitignore
├── run.py
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/jagadeesh-repos/Feature-Optimized-Stock-Prediction.git
cd Feature-Optimized-Stock-Prediction
```

### 2. Create a Virtual Environment

Using Python's built-in environment manager:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Alternatively, if you already have the project environment configured, continue using the existing environment.

### 3. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file from the example:

```powershell
Copy-Item .env.example .env
```

Configure the required values:

```env
SECRET_KEY=your_secret_key

GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_supported_groq_model
```

Additional database configuration can be specified if required.

Never commit the `.env` file or expose API credentials.

### 5. Run the Application

```powershell
python run.py
```

Alternatively, without activating the environment:

```powershell
& .\.venv\Scripts\python.exe run.py
```

Open the application:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

The application initializes its report database and separate LangGraph checkpoint database under the `instance/` directory.

## Application Pages and API Endpoints

### Main Pages

| Route | Description |
|---|---|
| `/` | Stock analysis dashboard |
| `/reports` | Previously generated reports |
| `/assistant` | AI assistant interface |
| `/about` | Project information and research limitations |
| `/health` | Application readiness check |

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/analyze` | POST | Perform stock analysis |
| `/api/reports` | GET | Retrieve saved reports |
| `/api/reports/<report_id>` | GET | Retrieve a specific report |
| `/api/reports/<report_id>/metrics` | GET | Retrieve evaluation metrics |
| `/api/reports/<report_id>/predictions` | GET | Retrieve predictions |
| `/api/reports/compare` | POST | Compare two reports |
| `/api/chat` | POST | Interact with the AI assistant |
| `/api/conversations` | GET/POST | Retrieve or create conversations |
| `/api/conversations/<thread_id>` | GET | Retrieve conversation details |

## Testing

The project uses Pytest for automated testing.

Run the complete test suite:

```powershell
python -m pytest -q
```

Or use the configured virtual environment directly:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

Tests use synthetic datasets where appropriate and do not require live Yahoo Finance or Groq connections for ordinary unit testing.

## Research Reference

Naik, Nagaraj & Mohan, Biju. (2021).

"Novel Stock Crisis Prediction Technique – A Study on Indian Stock Market."

IEEE Access.

DOI: [10.1109/ACCESS.2021.3088999](https://doi.org/10.1109/ACCESS.2021.3088999)

## Research Limitations

The current application uses the supplied regression notebook as its implementation baseline.

Its regression target, feature-selection configuration, model parameters, and evaluation workflow should not be presented as verified details of the published research paper.

The complete paper methodology, original dataset, exact crisis-label definition, HFS configuration, and model settings require verification before claiming exact reproduction.

## Future Work

- Verify and reproduce the complete methodology of the original IEEE Access paper.
- Extend the regression baseline to a dedicated binary crisis-classification pipeline if required.
- Explore additional macroeconomic and market indicators.
- Improve model explainability.
- Implement walk-forward validation and more extensive out-of-sample evaluation.
- Extend the AI assistant with additional report-grounded analysis tools.

## Deployment Scope

This is currently a local, single-user research application.

It uses Flask's development server and does not include production authentication or access controls.

Before public deployment, configure a production WSGI server, authentication, secure database access, and appropriate security controls.

## License

Add the appropriate license before distributing or reusing the project.
