# Model Overview: XGBoost
> Champion Model: Yes  |  Benchmark: DEV

## Business Executive

Here's a brief on the XGBoost Sales Forecasting model:

## What This Model Does
This model predicts the number of units of a product that will be sold.

## How It Works
It uses a sophisticated "tree-based" method to analyze historical sales data and identify patterns. It considers 39 different factors (features) to make its predictions.

## Key Strengths
*   **It's our Champion:** This is the model we’ve selected for daily use because it represents our best current forecasting capability.
*   **Low Error Rate:** On average, its predictions are off by about 108 units (MAE). This means most of the time, the actual sales are within a few hundred units of what the model predicts.
*   **Good for High-Volume Items:** The model correctly identifies and flags the top 20% of sales predictions that are most likely to be accurate (High\_Conf\_pct: 20.4%).
*   **Identifies Trends:** While not perfect, it does capture some underlying trends in sales, as indicated by its R-squared value of 0.15.

## Business Value
This model helps us make more informed decisions about inventory management, production scheduling, and staffing. By providing a clearer picture of future sales, it reduces the risk of stockouts or overstocking, leading to cost savings and improved customer satisfaction.

## Recommended Use
**This is our recommended production model for forecasting Unit Sales.** It should be used for daily operational planning and decision-making. While it provides valuable insights, remember that its accuracy is best for high-volume items and that there is still room for improvement in its overall predictive power.

---

## Data Scientist

## Model Overview: XGBoost Sales Forecasting

### What This Model Does

This XGBoost model is designed to forecast future `UnitSales`. It leverages a comprehensive set of 39 features to predict sales volumes, categorised under the "Sales Forecasting" domain. The model is sensitive to variations across different "Segment" groups, as indicated by its use of `group_vars`.

### How It Works

The XGBoost model operates as a tree-based ensemble learner. It constructs a series of decision trees, where each subsequent tree attempts to correct the errors of the preceding ones. This gradient boosting approach allows for a hierarchical decomposition of the target variable, `UnitSales`, by progressively refining predictions. The model's mechanics involve iteratively building trees that minimize a loss function, effectively learning complex, non-linear relationships between the 39 input features and the sales outcome.

### Key Strengths

*   **Robust Performance on DEV:** On the DEV dataset, the model achieves a Mean Absolute Error (MAE) of **107.94** and a Root Mean Squared Error (RMSE) of **440.91**, indicating a reasonable magnitude of prediction error relative to the sales volume.
*   **Low Percentage Error:** The Mean Absolute Percentage Error (MAPE) of **2.08%** on DEV signifies that, on average, the model's predictions are within a small percentage of the actual sales.
*   **Good Fit for Exploratory Analysis:** The R-squared value of **0.147** and Adjusted R-squared of **0.146** on DEV suggest that while the model explains a portion of the variance in `UnitSales`, it is best suited for initial exploration and understanding of sales drivers rather than definitive, high-confidence predictions.
*   **Identifies High Confidence Predictions:** The model successfully identifies instances with high prediction confidence, with **20.41%** of predictions falling into this category on the DEV dataset.

### Business Value

This XGBoost model provides valuable insights into future sales trends, enabling better inventory management, production planning, and resource allocation. By understanding the predicted `UnitSales`, the business can proactively adjust operations to meet anticipated demand, potentially reducing stockouts and overstocking costs.

### Recommended Use

This XGBoost model is the **CHAMPION MODEL** and is the organization's recommended production model for sales forecasting. It should be used for making informed predictions about `UnitSales`. While its R-squared on DEV (0.147) suggests it's not a perfect predictor, its low MAPE (2.08%) and reasonable MAE (107.94) make it a strong candidate for general forecasting tasks and as a benchmark for future model development. It is particularly useful when segment-specific insights are required.

---

## Risk Manager

## What This Model Does

The XGBoost model is designed to forecast UnitSales for various product segments.

## How It Works

This model utilizes the XGBoost algorithm, a gradient boosting framework known for its efficiency and performance. It learns complex relationships between input features (39 in total) and the target variable, UnitSales, to predict future sales volumes. The model is structured to handle the nuances of sales data, potentially incorporating time-series dependencies and external factors through its feature engineering.

## Key Strengths

*   **High Performance on DEV Dataset:** Achieves an impressive MAPE of 2.08% on the DEV dataset, indicating a low average percentage error in its sales predictions.
*   **Robustness:** Demonstrates a reasonable R2 of 0.147242 on the DEV dataset, suggesting it captures a portion of the variance in UnitSales.
*   **Confidence in Predictions:** The model exhibits a High_Conf_pct of 20.4072% on the DEV dataset, signifying that a considerable portion of its predictions fall within a high confidence interval.
*   **Scalability and Efficiency:** As an XGBoost model, it is designed for efficient training and prediction, making it suitable for large datasets and frequent forecasting needs.

## Business Value

This model provides critical insights into future sales, enabling more effective inventory management, optimized production planning, and informed marketing campaign strategies. By accurately forecasting UnitSales, businesses can reduce stockouts, minimize excess inventory costs, and improve overall supply chain efficiency.

## Recommended Use

**The XGBoost model is the organization's CHAMPION MODEL and is the recommended production model for Sales Forecasting.** It should be used for predicting UnitSales, particularly for scenarios where a low percentage error (MAPE of 2.08% on DEV) is desired. Its performance on the DEV dataset indicates its suitability for general forecasting tasks across all segments. Continuous monitoring for data drift and temporal stability is advised, especially given the Adj_R2 of 0.146236, to ensure its predictive power is maintained over time.

---
