"""
Comparison of Machine Learning Models for F1 Race Time Prediction

This script compares different ML models (GradientBoosting, Ridge, XGBoost, SVR) for predicting F1 race times
based on qualifying data, weather, team performance, and historical sector times. It trains models,
evaluates MAE, displays podiums, and visualizes feature importance and predictions.
"""

import fastf1
import pandas as pd
import numpy as np
import seaborn as sns
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,       # MAE
    mean_squared_error,        # MSE
    r2_score,                  # R²
    median_absolute_error      # MedAE
)
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Enable F1 data caching
fastf1.Cache.enable_cache("f1_cache")

# Load 2024 Emilia Romagna Qualifying Session
# Fetch and load qualifying data for the 2024 Emilia Romagna GP (round 7)
session_2024 = fastf1.get_session(2024, 7, "Q")
session_2024.load()

# Extract relevant lap data and drop rows with missing values
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# Convert timedelta columns to seconds for numerical processing
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Aggregate sector times by driver to get average sector performances
sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()

# Calculate total sector time as sum of individual sectors
sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"] +
    sector_times_2024["Sector2Time (s)"] +
    sector_times_2024["Sector3Time (s)"]
)

# Pre-computed clean air race pace for each driver (average race lap time in seconds)
clean_air_race_pace = {
    "VER": 88.13859652333029, "HAM": 87.99858076923077, "LEC": 88.19860530973452,
    "NOR": 87.95796681222707, "ALO": 89.79116370808678, "PIA": 88.11267857142857,
    "RUS": 88.25423928571429, "SAI": 88.11452128666036, "STR": 89.53571047008548,
    "HUL": 90.5244616915423, "OCO": 89.89775745118198, "TSU": 89.37616685456595,
    "GAS": 88.91867071823204, "ALB": 89.65372986369269
}

# Wet weather adjustment factors for qualifying times
wet_weather_factor = {
    "VER": 0.8828256162720444, "HAM": 0.8550248519900767, "LEC": 0.8447992944831761,
    "NOR": 0.922467828177502, "ALO": 0.9160402229972915, "PIA": 0.8632501469172694,
    "RUS": 0.8276233764085413, "SAI": 0.8284226284248292, "STR": 0.9494004296179431,
    "HUL": 0.8871134455802793, "OCO": 0.9996818544695274, "TSU": 0.961373412451465,
    "GAS": 0.9691866544335193, "ALB": 0.9147229519070319
}

# Create DataFrame for 2025 qualifying times (simulated data)
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "RUS", "SAI", "ALB", "LEC", "OCO",
               "HAM", "STR", "GAS", "ALO", "HUL"],
    "QualifyingTime (s)": [
        74.704, 74.962, 74.670, 74.807, 75.432, 75.473, 75.604, 76.613,
        75.765, 75.581, 75.787, 75.431, 76.518
    ]
})

# Map clean air race pace to qualifying data
qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

# Fetch weather forecast for Emilia Romagna GP location
API_KEY = "6659192f0eeaf84f10720b9d60458a75"
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat=44.3439&lon=11.7167&appid={API_KEY}&units=metric"
response = requests.get(weather_url)
weather_data = response.json()

# Find forecast closest to race time
forecast_time = "2025-05-18 06:00:00"
forecast_data = next((f for f in weather_data["list"] if f["dt_txt"] == forecast_time), None)

# Extract rain probability and temperature
rain_probability = forecast_data["pop"] if forecast_data else 0
temperature = forecast_data["main"]["temp"] if forecast_data else 20

# Adjust qualifying times if rain is expected (threshold 75%)
if rain_probability >= 0.75:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"] * qualifying_2025["Driver"].map(wet_weather_factor)
else:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]

# Constructor standings points for team performance scoring
team_points = {
    "McLaren": 246, "Mercedes": 141, "Red Bull": 105, "Williams": 37, "Ferrari": 94,
    "Haas": 20, "Aston Martin": 14, "Kick Sauber": 6, "Racing Bulls": 8, "Alpine": 7
}

# Normalize team points to create performance scores
max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

# Map drivers to their teams
driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
    "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Racing Bulls",
    "SAI": "Ferrari", "HUL": "Kick Sauber", "OCO": "Alpine", "STR": "Aston Martin"
}

# Add team and performance score to qualifying data
qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)

# Merge qualifying data with historical sector times
merged_data = qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

# Define features (X) and target (y) for ML models
X = merged_data[[
    "QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore", "CleanAirRacePace (s)"
]]
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

# Impute missing values with median
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=34)

# Define ML models for comparison
models = {
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=34),
    "Ridge": Ridge(alpha=1.0),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=34),
    "SVR": make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1))
}

# Train each model, make predictions, and evaluate MAE
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    merged_data[f"Predicted_{name} (s)"] = model.predict(X_imputed)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # percentage
    
    results[name] = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
        "MedAE": medae,
        "MAPE (%)": mape
    }
    
    print(f"\n{name} Performance:")
    print(f"  MAE:   {mae:.3f} s")
    print(f"  MSE:   {mse:.3f}")
    print(f"  RMSE:  {rmse:.3f} s")
    print(f"  R²:    {r2:.3f}")
    print(f"  MedAE: {medae:.3f} s")
    print(f"  MAPE:  {mape:.2f} %")
    
# ✅ Now build the heatmap once (outside the loop)
results_df = pd.DataFrame(results).T
results_df_rounded = results_df.round(3)

# Normalized version for color scaling
norm_df = (results_df - results_df.min()) / (results_df.max() - results_df.min())

plt.figure(figsize=(10, 5))
sns.heatmap(norm_df, annot=results_df_rounded, fmt=".3f", cmap="viridis", cbar_kws={'label': 'Normalized (0-1)'})
plt.title("Model comparison across metrics (normalized colors, annotated with raw values)")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
plot_df = results_df_rounded.reset_index().melt(id_vars="index", var_name="Metric", value_name="Value")
plot_df.rename(columns={"index": "Model"}, inplace=True)

plt.figure(figsize=(12, 6))
sns.barplot(data=plot_df, x="Model", y="Value", hue="Metric", palette="viridis")

for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height():.3f}',
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='bottom', fontsize=8, color='black', xytext=(0, 3),
                       textcoords='offset points')

plt.yscale("log")  # optional for clearer visual range
plt.title("Comparison of ML Models Across All Metrics", fontsize=14, weight='bold')
plt.xlabel("Model", fontsize=12)
plt.ylabel("Metric Value (log scale)", fontsize=12)
plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print("\n📋 Summary Table of All Metrics:\n")
print(results_df_rounded)

# Display predicted podium for each model
for name in models.keys():
    podium = merged_data.sort_values(f"Predicted_{name} (s)").reset_index(drop=True).loc[:2, ["Driver", f"Predicted_{name} (s)"]]
    print(f"\n🏆 Podium for {name} 🏆")
    print(f"🥇 P1: {podium.iloc[0]['Driver']}")
    print(f"🥈 P2: {podium.iloc[1]['Driver']}")
    print(f"🥉 P3: {podium.iloc[2]['Driver']}")
    

# Plot predicted race times vs clean air race pace for each model
for name in models.keys():
    plt.figure(figsize=(10,6))
    plt.scatter(merged_data["CleanAirRacePace (s)"], merged_data[f"Predicted_{name} (s)"])
    for i, driver in enumerate(merged_data["Driver"]):
        plt.annotate(driver,
                     (merged_data["CleanAirRacePace (s)"].iloc[i], merged_data[f"Predicted_{name} (s)"].iloc[i]),
                     xytext=(5,5), textcoords='offset points')
    plt.xlabel("Clean Air Race Pace (s)")
    plt.ylabel(f"Predicted Race Time ({name}) (s)")
    plt.title(f"Effect of Clean Air Race Pace on Predicted Race Times ({name})")
    plt.tight_layout()
    plt.show()

# Import permutation importance for SVR
from sklearn.inspection import permutation_importance

# Calculate and plot feature importance for each model
for name, model in models.items():
    print(f"\nFeature Importance / Contribution for {name}:")

    if name in ["GradientBoosting", "XGBoost"]:
        feature_importance = model.feature_importances_
    elif name == "Ridge":
        feature_importance = np.abs(model.coef_)
    elif name == "SVR":
        # Use permutation importance for SVR since it doesn't have built-in feature_importances_
        result = permutation_importance(model, X_imputed, y, n_repeats=10, random_state=34)
        feature_importance = result.importances_mean

    features = X.columns
    plt.figure(figsize=(8,5))
    plt.barh(features, feature_importance, color='skyblue')
    plt.xlabel("Importance")
    plt.title(f"Feature Importance in Race Time Prediction ({name})")
    plt.tight_layout()
    plt.show()

