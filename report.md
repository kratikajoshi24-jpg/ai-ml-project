# Project Write-up: Demand Forecasting with Uncertainty Quantification

## Problem statement
Retail demand is uncertain. A single forecast can hide the possibility that actual demand will be substantially higher or lower, contributing to stockouts or excess inventory. This project demonstrates a forecasting workflow that provides both a central prediction and a range of plausible demand values.

## Objective
Build a reproducible baseline that:
- creates time-based demand features;
- predicts the 10th, 50th, and 90th demand quantiles;
- evaluates point accuracy and interval reliability; and
- demonstrates how an upper demand quantile can inform a simple inventory decision.

## Data
The default workflow uses a generated synthetic daily demand series with trend, weekly seasonality, annual seasonality, promotions, and random variation. Synthetic data makes the project reproducible without external downloads. It is not real retail data, so the results are illustrative.

## Approach
1. Generate daily observations and a promotion flag.
2. Construct lag features and rolling averages using historical values only.
3. Split observations chronologically into training and test periods.
4. Train quantile gradient-boosting models for the 10th, 50th, and 90th percentiles.
5. Report MAE and RMSE for the median prediction, empirical coverage and average width of the 10th–90th interval.
6. Save a plot and CSV results, and show an illustrative replenishment calculation.

## Evaluation
Run `python src/run_project.py`. The script writes actual metrics to `outputs/metrics.json` and the forecast table to `outputs/forecast_results.csv`. Use the generated metrics in any presentation; do not insert unmeasured performance claims.

- **MAE:** average absolute difference between actual and median forecast.
- **RMSE:** square root of mean squared forecast error; it penalises larger errors more strongly.
- **Interval coverage:** fraction of actual observations that fall between the lower and upper quantiles.
- **Average interval width:** average upper quantile minus lower quantile. Narrower intervals are more precise, but coverage must be considered too.

## Inventory illustration
The script uses the upper forecast quantile as a conservative demand estimate for a simple illustrative inventory target. This is not a complete inventory optimisation model: real decisions should account for lead time, costs, existing stock, and supplier constraints.

## Limitations and future improvements
- Evaluate on a real dataset such as a suitably licensed store-item sales dataset.
- Use rolling-origin backtesting instead of a single holdout split.
- Calibrate prediction intervals and analyse coverage by season, promotion, and item.
- Incorporate lead times, inventory levels, and different overstock/stockout costs.
- Compare quantile models with residual-based intervals or probabilistic forecasting approaches.

## Reproducibility
Install dependencies with `pip install -r requirements.txt` and run `python src/run_project.py`. Outputs are written to the `outputs/` directory.
