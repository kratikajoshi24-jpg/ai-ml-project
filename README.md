# Demand Forecasting with Uncertainty Quantification

A reproducible AI/ML project that forecasts product demand and reports prediction intervals, helping retailers reason about inventory risk instead of relying on a single point forecast.

## What this project does
- Builds a demo daily store-item demand dataset (so the project runs without a large external download).
- Engineers time-series features: lags, rolling averages, day-of-week, month, and promotion indicator.
- Trains quantile gradient-boosting models to estimate the 10th, 50th, and 90th demand percentiles.
- Evaluates point accuracy (MAE and RMSE) and interval coverage/width.
- Creates forecast plots and a simple inventory-planning example based on the upper forecast quantile.

## Quick start

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python src/run_project.py
```

The run creates:
- `outputs/metrics.json` — evaluation metrics
- `outputs/forecast_results.csv` — actuals and forecast quantiles on the test period
- `outputs/forecast_plot.png` — actual demand and prediction interval
- `outputs/inventory_example.json` — illustrative inventory decision
- `outputs/feature_importance.csv` — permutation importance, when available

## Dataset
By default, the script generates a deterministic synthetic dataset with daily demand, weekly/yearly patterns, and promotions. This makes the repository easy to reproduce without downloading a large dataset. It is a demonstration dataset, not real Walmart sales data.

For a real-world extension, replace the demo data-generation function with a prepared store-item time-series dataset such as the M5 Forecasting dataset from Kaggle. Make sure you comply with the dataset's terms and document the source.

## Method
1. Generate/prepare daily demand records.
2. Create lag and rolling-window features using past observations only.
3. Use a chronological train/test split to avoid random time-series leakage.
4. Fit quantile models at 0.10, 0.50, and 0.90.
5. Evaluate MAE/RMSE for the median forecast and coverage/width for the 10th–90th percentile interval.
6. Illustrate a simple replenishment quantity using the upper quantile.

## Notes and limitations
- The demo dataset is synthetic; results should not be interpreted as real retail performance.
- The inventory example is illustrative and does not model lead times, holding costs, lost-sales costs, or supplier constraints.
- The interval coverage is measured on the held-out test period. A nominal 80% interval may not achieve 80% empirical coverage; that difference is part of the evaluation.
- This baseline is intended as an educational project and can be improved with real data, cross-validation, richer features, and cost-based inventory optimisation.

## Repository structure
```text
demand-forecasting-uncertainty/
├── README.md
├── requirements.txt
├── .gitignore
├── report.md
├── data/
│   └── README.md
├── outputs/
│   └── .gitkeep
└── src/
    └── run_project.py
```
