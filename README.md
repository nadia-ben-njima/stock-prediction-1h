# AAPL Hourly Stock Prediction

An end-to-end machine-learning project exploring **next-hour Apple (AAPL) price prediction** from hourly market data. The project combines Jupyter notebooks for data preparation and LSTM modeling, a FastAPI inference service, and a Flask dashboard for interactive time-series analysis.

> **Disclaimer:** This is an educational project, not financial advice or investment research. Predictions and dashboard signals are experimental and should not be used to make trading decisions.

## Project at a glance

- **Problem:** Predict the next hourly AAPL closing price and return from recent market observations.
- **Model:** Long Short-Term Memory (LSTM) neural network.
- **Input window:** 10 consecutive hourly observations.
- **Inference API:** FastAPI.
- **Dashboard:** Flask, HTML/CSS/JavaScript, and Chart.js.
- **Workflow:** Data acquisition → feature engineering → sequence generation → model training → API inference → dashboard visualization.

## Architecture

```text
Yahoo Finance hourly data
          |
          v
  Jupyter data preparation
          |
          v
 Feature engineering
          |
          v
 10-step input sequences
          |
          v
    LSTM model
          |
     +----+----+
     |         |
     v         v
 FastAPI    Saved model
 /predict   + scalers
     |
     v
 Flask dashboard
     |
     v
 Charts, indicators,
 prediction and comparison
```

The FastAPI service loads the trained Keras model and the fitted feature/target scalers from `notebooks/`. The Flask dashboard prepares the latest feature window and sends it to the API.

## Tech stack

| Area | Tools |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy |
| Data source | yfinance |
| Modeling | TensorFlow / Keras, LSTM |
| Preprocessing | scikit-learn, MinMaxScaler |
| API | FastAPI, Uvicorn |
| Dashboard | Flask, HTML, CSS, JavaScript |
| Visualization | Chart.js, Matplotlib, Seaborn |
| Experimentation | Jupyter Notebook |

## Repository structure

```text
.
├── aapl_dashboard/
│   ├── app.py                  # Flask dashboard and data endpoints
│   └── templates/
│       └── index.html          # Dashboard interface
├── data/
│   └── raw/                    # Hourly AAPL CSV (generated locally)
├── notebooks/
│   ├── 01_load_aapl_data.ipynb
│   ├── 02_make_target_and_features.ipynb
│   ├── lstm_model.keras        # Trained model
│   ├── scaler_X.pkl            # Input feature scaler
│   └── scaler_y.pkl            # Target scaler
├── src/
│   └── app.py                  # FastAPI inference service
├── requirements.txt
└── README.md
```

The market-data CSV is not included in the repository. The trained model and scaler artifacts are included.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/nadia-ben-njima/stock-prediction-1h.git
cd stock-prediction-1h
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use a Python version supported by the TensorFlow version in the requirements file.

### 4. Prepare the data

Open and run `notebooks/01_load_aapl_data.ipynb`. It downloads hourly AAPL data using yfinance and writes the dataset to the project's `data/raw/` directory.

Before starting the applications, verify that this file exists:

```text
data/raw/AAPL_1h.csv
```

If it is saved elsewhere, update the data path in the notebook or dashboard so both use the same CSV.

### 5. Start the FastAPI service

From the repository root, run:

```bash
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at:

- Health check: http://127.0.0.1:8000/
- Interactive API docs: http://127.0.0.1:8000/docs
- Prediction endpoint: http://127.0.0.1:8000/predict

The model and scalers are loaded when the API starts.

### 6. Start the dashboard

In a second terminal, activate the same virtual environment, move to the repository root, and run:

```bash
python aapl_dashboard/app.py
```

Open http://127.0.0.1:5000 in your browser. The dashboard sends prediction requests to the FastAPI service at `http://127.0.0.1:8000/predict` by default.

To use a different API URL, set `MODEL_API_URL` before starting Flask:

```bash
export MODEL_API_URL=http://127.0.0.1:8000/predict
```

## Modeling workflow

The notebooks document the main experimentation pipeline:

1. Download hourly OHLCV data for AAPL.
2. Define the next-hour close and return as prediction targets.
3. Engineer price-history, return, candle, moving-average, volatility, and momentum features.
4. Scale the feature and target data with `MinMaxScaler`.
5. Build rolling sequences using the previous 10 time steps.
6. Train an LSTM network with 32 units, dropout, and a dense output layer.
7. Use early stopping and a chronological 80/20 sequence split.
8. Save the trained model and scalers for API inference.

The dashboard also visualizes indicators such as RSI, moving averages, Bollinger-style bands, momentum, volume, and rolling actual-versus-predicted values.

## API contract

The `POST /predict` endpoint accepts a JSON object containing a sequence of 10 time steps. Each time step must contain the 27 features in the exact order used during training.

Example request shape:

```json
{
  "sequence": [
    [/* 27 feature values for time step 1 */],
    [/* 27 feature values for time step 2 */]
  ]
}
```

The example is abbreviated: send exactly 10 rows, each with 27 numeric values. The response contains the model's prediction output.

## Current limitations and next steps

This repository is an experimental implementation, and its results should be interpreted carefully.

- **Preprocessing leakage:** In the current training notebook, the scalers are fitted before the chronological train/test split. Fit them on the training portion only, then transform the test portion, before using the evaluation as an unbiased estimate.
- **Evaluation:** Add clearly reported MAE, RMSE, and directional accuracy on a strictly held-out chronological test period, and compare against simple baselines such as persistence (predicting the next close equals the latest close).
- **Reproducibility:** Record data download dates, feature definitions, random seeds, and model configuration for each experiment.
- **API robustness:** Add request-schema validation, clearer error responses, and automated tests for feature count and ordering.
- **Metrics endpoint:** The API defines `GET /metrics`, which expects `notebooks/metrics.json`; that file is not currently included, so this endpoint needs the metrics artifact to be generated before it can be used.
- **Deployment:** Add environment-based configuration and a reproducible deployment setup for the API and dashboard.

## Project origin and attribution

This repository is a personal portfolio copy of a collaborative codebase. The Git history has been retained so the project's development history and contributor attribution remain visible. The original project repository is [sinex-cloud/stock-prediction-1h](https://github.com/sinex-cloud/stock-prediction-1h).

## License

No license is currently provided. Please contact the relevant project contributors before reusing or redistributing the code.
