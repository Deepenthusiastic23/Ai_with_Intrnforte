# Temperature Forecast Project

Short-term (next-day) maximum temperature forecasting using historical daily weather data and regression models.

## Problem Statement
Given the last few days of weather observations (temperature, humidity, pressure, wind speed, rainfall, cloud cover), predict tomorrow's maximum temperature and compare the results against a naive baseline.

## Dataset
Daily weather data from 2018 to 2025 with these columns: `date`, `temp_max`, `temp_min`, `humidity`, `pressure`, `wind_speed`, `rainfall_mm`, `cloud_cover`.

**Note:** the data is synthetically generated (seasonal pattern, monsoon effect, autocorrelated noise). To use real data, replace the data-generation cell with `pd.read_csv('your_file.csv', parse_dates=['date'])`.

## Approach
1. Exploratory data analysis (trend, monthly pattern, correlation heatmap)
2. Feature engineering: lag features, 3-day and 7-day rolling statistics, calendar features
3. Chronological train/test split (85% / 15%) to avoid data leakage
4. Models: Naive baseline, Linear Regression, Random Forest, Gradient Boosting
5. Evaluation with MAE, RMSE and R2

## Results

| Model | MAE (°C) | RMSE (°C) | R2 |
|---|---|---|---|
| Linear Regression | 1.356 | 1.673 | 0.976 |
| Gradient Boosting | 1.367 | 1.709 | 0.975 |
| Random Forest | 1.388 | 1.730 | 0.975 |
| Naive Baseline | 1.745 | 2.180 | 0.960 |

All models beat the naive baseline. Linear Regression performed best, and the small gap between models shows that the engineered lag and rolling features drive most of the accuracy.

## Limitations
- The data is synthetic, so scores show that the pipeline works, not real-world accuracy.
- The 7-day rolling forecast only updates the temperature lag features, so its predictions flatten out after the first day. A production version should update all features at each step or train a separate model per forecast horizon.

## How to Run
```
pip install pandas numpy scikit-learn matplotlib seaborn ipykernel
```
Open `Temperature_Forecast_Project.ipynb` in VS Code or Jupyter and click **Run All**.

## Files
- `Temperature_Forecast_Project.ipynb`: main notebook
- `Temperature_Forecast_Project.html`: exported notebook with outputs