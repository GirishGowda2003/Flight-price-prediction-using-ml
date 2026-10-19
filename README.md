# Flight-price-prediction-using-ml
A machine learning application that predicts flight ticket prices based on airline, source, destination, departure time, duration, and other flight-related features.

# ✈️ Flight Fare Prediction — End-to-End Machine Learning Project

**A beginner-friendly, step-by-step regression project**

In this notebook we build a model that predicts flight ticket prices based on
features like airline, source, destination, duration, number of stops, and date of journey.

We go through the **complete data science workflow**:

1. Load & explore the raw dataset (EDA)
2. Clean the data and engineer new features
3. Handle outliers and encode/scale variables
4. Build, tune, and compare multiple regression models
5. Summarise results and save the best model

> **Revision note:** This version of the notebook fixes the issues raised in review —
> missing interpretation of charts, LabelEncoder misuse on nominal features, data-leakage in
> scaling, broken/duplicated model-building cells (e.g. `XGRegressor` typo, invalid `SVR`
> parameters), and the lack of a final model-comparison summary.
