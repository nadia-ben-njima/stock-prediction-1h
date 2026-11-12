# AAPL Stock Prediction — Hourly Time Series

A machine-learning project that explores next-hour Apple (AAPL) closing-price prediction using historical hourly market data. It combines a notebook-based data and model workflow with a FastAPI inference service and a Flask dashboard for visualizing market features and model predictions.

> **Disclaimer:** This is an educational machine-learning project, not investment research or financial advice. Model outputs and dashboard signals should not be used to make trading decisions.

## Project overview

The repository covers several stages of a time-series ML workflow:

1. Load and prepare hourly AAPL market data.
2. Create lagged, return-based, trend, momentum, and volatility features.
3. Train an LSTM model and save the model and preprocessing scalers.
4. Serve predictions through a FastAPI endpoint.
5. Display market indicators, prediction output, and a rolling actual-versus-predicted comparison in a Flask dashboard.

The prediction target is the next hourly close derived from the current close series.

## Architecture

```text
Hourly AAPL CSV
      |
      v
Feature engineering
      |
      v
LSTM model + fitted scalers
      |
      v
FastAPI inference service (:8000)
      ^
      | HTTP requests
      |
Flask dashboard (:5000)
      |
      v
Charts, indicators, prediction and backtest view
```

## Tech stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** TensorFlow / Keras, scikit-learn
- **Market data:** yfinance (listed in the project dependencies)
- **API:** FastAPI, Uvicorn
- **Dashboard:** Flask
- **Visualization and analysis:** Jupyter, Matplotlib, Seaborn, SciPy

## Repository structure

```text
.
├── aapl_dashboard/
│   ├── templates/             # Dashboard HTML templates
│   ├── app.py                 # Flask dashboard and chart endpoints
│   └── requirements.txt
├── data/
│   ├── raw/                   # AAPL_1h.csv input data
│   └── processed/
├── notebooks/
│   ├── 01_load_aapl_data.ipynb
│   ├── 02_make_target_and_features.ipynb
│   ├── lstm_model.keras       # Trained model artifact
│   ├── scaler_X.pkl           # Input feature scaler
│   └── scaler_y.pkl           # Target scaler
├── reports/
├── src/
│   └── app.py                 # FastAPI inference service
├── requirements.txt
└── README.md
```

The API also loads `notebooks/metrics.json` for its `/metrics` endpoint; make sure this artifact is present in the checked-out branch.

## Data and feature engineering

The dashboard's feature pipeline derives technical and statistical features from the hourly OHLCV data, including:

| Feature group | Examples |
|---|---|
| Price history | Previous closes and close lags |
| Returns | One-hour percentage return and log returns |
| Candle features | High-low range and open-close change |
| Trend | Simple and exponential moving averages, rolling slope |
| Volatility | Rolling standard deviation and variance, intraday variance |
| Momentum | Five-period momentum, rolling highs and lows |
| Technical indicators | RSI (14), Bollinger-style upper/lower bands |
| Daily aggregation | Daily true range and ATR (3) |

The dashboard's LSTM inference path uses a fixed list of 27 input features and a 10-row sequence. The feature order must remain consistent with the order used to fit `scaler_X` and train the saved model.

## Setup

### Prerequisites

- Python 3.10 or newer
- The repository's data and model artifacts
- A working environment for TensorFlow

Clone the repository:

```bash
git clone https://github.com/nadia-ben-njima/stock-prediction-1h.git
cd stock-prediction-1h
```

Create and activate a virtual environment:

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
# .venv\\Scripts\\activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Ensure the following files are available before starting the services:

- `data/raw/AAPL_1h.csv`
- `notebooks/lstm_model.keras`
- `notebooks/scaler_X.pkl`
- `notebooks/scaler_y.pkl`
- `notebooks/metrics.json`

The model and scalers are loaded when the FastAPI application starts.

## Run the application

Start the model API in one terminal:

```bash
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```

Start the dashboard in a second terminal:

```bash
python aapl_dashboard/app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

The dashboard calls the model API at `http://127.0.0.1:8000/predict` by default. To use another API address, set `MODEL_API_URL` before starting the Flask app:

```bash
export MODEL_API_URL=http://127.0.0.1:8000/predict
```

## API endpoints

### FastAPI model service

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Health check |
| GET | `/metrics` | Returns the saved metrics JSON |
| POST | `/predict` | Scales a supplied sequence, runs the LSTM, and inverse-transforms the prediction |

Example prediction request:

```json
{
  "sequence": [
    [/* 27 feature values for one hour */],
    [/* 27 feature values for the next hour */]
  ]
}
```

The sequence should contain 10 time steps, each with 27 values in the exact training feature order. The response contains a `prediction` value.

### Flask dashboard

The dashboard exposes chart and analysis routes under `/api`, including price/return charts, RSI, volume, Bollinger bands, momentum, feature-table data, missing-value statistics, and `/api/prediction` for the LSTM-backed prediction and rolling backtest view.

## Model evaluation

The dashboard computes error and directional metrics for the rolling prediction comparison when enough valid points are available:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Directional accuracy

No headline accuracy or performance score is stated here because results depend on the saved model, input data, and evaluation window. Evaluate on a strictly held-out chronological period before drawing conclusions about generalization.

## Limitations and responsible use

- Financial time series are noisy and non-stationary; past patterns do not guarantee future performance.
- A model prediction is not a calibrated probability or a reliable buy/sell recommendation.
- The dashboard's BUY / SELL / HOLD label is a simple threshold derived from predicted percentage change, not a validated trading strategy.
- Avoid random train/test shuffling for time-series evaluation; use chronological splits and guard against look-ahead leakage.
- Results can change with the dataset, feature definitions, scaler fit, and model version.

## Future work

- Add reproducible chronological train/validation/test evaluation.
- Report baseline comparisons and metrics for a clearly defined test window.
- Track experiment parameters and model versions.
- Add automated tests for feature ordering and API input validation.
- Package the API and dashboard for reproducible deployment.

## License

No license is currently specified. Contact the repository owner before reusing or redistributing this project.
