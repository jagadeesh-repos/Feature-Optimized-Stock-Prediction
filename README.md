# Stock Crisis Prediction Using XGBoost and Deep Neural Networks

##  Project Overview
This project implements a **Stock Crisis Prediction** system that detects potential **market crisis events** (significant price crashes) using **machine learning** and **deep learning** models.  
It reproduces and extends the methodology from the research paper *“Novel Stock Crisis Prediction Technique – A Study on Indian Stock Market”*.

The objective is to classify time periods in which a stock or index is at **high risk of a crisis**, defined as a **price drop exceeding a threshold** (e.g., ≥ 10% within 21 trading days).

---

##  Key Features
- **Crisis Labeling** – Automatically labels *crisis* vs. *normal* periods based on future drawdowns.  
- **Hybrid Feature Selection (HFS)** – Combines filter, wrapper, and embedded methods to select the most informative technical indicators.  
- **Dual Modeling Framework:**
  - **XGBoost** for gradient-boosted decision trees.
  - **Deep Neural Network (DNN)** for nonlinear feature learning.
- **Imbalanced Data Handling** – Uses SMOTE and class weighting for rare crisis event detection.  
- **Evaluation Metrics** – Focuses on recall, F1-score, and precision-recall AUC for the crisis class.  

---

##  Methodology

### 1️Data Preparation
- **Input:** Daily stock or index data (`Date`, `Open`, `High`, `Low`, `Close`, `Volume`).  
- **Source:** Yahoo Finance, NSE/BSE archives, or custom datasets.  
- **Processing:** Data is sorted chronologically and cleaned using missing value interpolation.

---

### 2️Crisis Definition
A sample crisis condition:

> A *“crisis”* occurs when the **minimum price in the next 21 trading days** falls by **10% or more** relative to the current closing price.

You can adjust this via the configuration constants:
```python
CRISIS_DROP = 0.10        # 10% threshold
HORIZON_DAYS = 21         # lookahead window
```

## Feature Engineering

The project includes several technical indicators commonly used in financial modeling:

- Moving Averages (SMA, EMA)
- MACD
- RSI
- Rolling Volatility
- Volume Change
- Momentum Returns (1-day, 5-day, etc.)

---

## Hybrid Feature Selection (HFS)

A three-phase process:

1. **Filter:** Remove highly correlated features (> 0.95) and rank remaining features by mutual information.
2. **Wrapper/Embedded:** Use an XGBoost model to estimate feature importance.
3. **Selection:** Keep top-N features for the final modeling phase.

This hybrid process ensures that only the most relevant and non-redundant features are retained.

---

## Modeling

### XGBoost Classifier
- Efficiently handles nonlinear interactions.
- Incorporates `scale_pos_weight` to manage class imbalance.
- Provides feature importance for interpretability.

### Deep Neural Network (DNN)
- 4-layer fully connected feed-forward model with ReLU activations and dropout regularization.
- Trained with binary cross-entropy loss and the Adam optimizer.

---

## Evaluation

**Metrics used:**
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Precision-Recall AUC (PR-AUC) (recommended for imbalanced datasets)

---

## Future Work
- Incorporate macroeconomic and sentiment features (e.g., VIX, news-based indicators).
- Experiment with sequential models (LSTM, GRU, Transformer).
- Implement walk-forward validation and backtesting for real-world performance evaluation.

---

## Citation
If you use this project or its methodology, please cite:

Naik, N. & Mohan, B. (2021). Novel Stock Crisis Prediction Technique — A Study on Indian Stock Market. *IEEE Access*. pp. 1–1. doi:10.1109/ACCESS.2021.3088999

