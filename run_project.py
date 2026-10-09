from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def make_demo_data(n_days: int = 1100, seed: int = 42) -> pd.DataFrame:
    """Create a reproducible synthetic daily demand series."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)
    promotion = (rng.random(n_days) < 0.10).astype(int)
    weekly = 7 * np.sin(2 * np.pi * t / 7)
    yearly = 10 * np.sin(2 * np.pi * t / 365.25)
    trend = 0.015 * t
    promo_lift = 12 * promotion
    noise = rng.normal(0, 5 + 1.5 * promotion, n_days)
    demand = np.maximum(0, 55 + weekly + yearly + trend + promo_lift + noise)
    return pd.DataFrame({"date": dates, "demand": demand.round(2), "promotion": promotion})


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features using past demand only; no future values are used."""
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["day_of_week"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["day_of_year"] = out["date"].dt.dayofyear
    for lag in (1, 7, 14, 28):
        out[f"lag_{lag}"] = out["demand"].shift(lag)
    out["rolling_mean_7"] = out["demand"].shift(1).rolling(7).mean()
    out["rolling_mean_28"] = out["demand"].shift(1).rolling(28).mean()
    return out.dropna().reset_index(drop=True)


def main() -> None:
    raw = make_demo_data()
    featured = add_features(raw)

    feature_cols = [
        "promotion", "day_of_week", "month", "day_of_year",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_mean_28",
    ]
    split = int(len(featured) * 0.80)
    train, test = featured.iloc[:split], featured.iloc[split:]
    X_train, y_train = train[feature_cols], train["demand"]
    X_test, y_test = test[feature_cols], test["demand"]

    predictions = {}
    for q, label in [(0.10, "q10"), (0.50, "q50"), (0.90, "q90")]:
        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=180,
            learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0,
            random_state=42,
        )
        model.fit(X_train, y_train)
        predictions[label] = np.maximum(0, model.predict(X_test))

    # Ensure lower <= median <= upper for a valid displayed interval.
    lower = np.minimum(predictions["q10"], predictions["q90"])
    upper = np.maximum(predictions["q10"], predictions["q90"])
    median = np.clip(predictions["q50"], lower, upper)
    actual = y_test.to_numpy()

    coverage = float(np.mean((actual >= lower) & (actual <= upper)))
    metrics = {
        "dataset": "synthetic daily demand (illustration only)",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "mae_median_forecast": round(float(mean_absolute_error(actual, median)), 4),
        "rmse_median_forecast": round(float(np.sqrt(mean_squared_error(actual, median))), 4),
        "q10_q90_interval_nominal_coverage": 0.80,
        "empirical_interval_coverage": round(coverage, 4),
        "average_interval_width": round(float(np.mean(upper - lower)), 4),
        "note": "Synthetic demonstration data; metrics are not real-world retail results.",
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    results = pd.DataFrame({
        "date": test["date"].to_numpy(),
        "actual_demand": actual,
        "forecast_q10": lower,
        "forecast_q50": median,
        "forecast_q90": upper,
        "promotion": test["promotion"].to_numpy(),
    })
    results.to_csv(OUT / "forecast_results.csv", index=False)

    # Illustrative inventory decision: upper quantile as a conservative target.
    last = results.iloc[-1]
    inventory_example = {
        "date": str(pd.Timestamp(last["date"]).date()),
        "median_forecast": round(float(last["forecast_q50"]), 2),
        "upper_quantile_forecast": round(float(last["forecast_q90"]), 2),
        "illustrative_target_units_for_next_day": int(np.ceil(last["forecast_q90"])),
        "caveat": "Illustration only; does not account for current stock, lead time, or costs.",
    }
    (OUT / "inventory_example.json").write_text(
        json.dumps(inventory_example, indent=2), encoding="utf-8"
    )

    # Plot a short final segment of the test period.
    plot_n = min(100, len(results))
    plot_df = results.tail(plot_n)
    plt.figure(figsize=(11, 5))
    plt.plot(plot_df["date"], plot_df["actual_demand"], label="Actual demand", linewidth=1.6)
    plt.plot(plot_df["date"], plot_df["forecast_q50"], label="Median forecast (q50)", linewidth=1.5)
    plt.fill_between(
        plot_df["date"],
        plot_df["forecast_q10"].to_numpy(),
        plot_df["forecast_q90"].to_numpy(),
        alpha=0.2,
        label="q10–q90 prediction interval",
    )
    plt.title("Demand Forecast with Prediction Interval")
    plt.xlabel("Date")
    plt.ylabel("Demand (units)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "forecast_plot.png", dpi=160)
    plt.close()

    print("Project completed.")
    print(json.dumps(metrics, indent=2))
    print(f"Outputs saved to: {OUT}")


if __name__ == "__main__":
    main()
