"""
Terminal Output for F1 Race Prediction

This script provides a terminal-based interface for predicting F1 race outcomes based on user-input qualifying times,
weather data, and historical performance. It uses a GradientBoostingRegressor model to predict race times,
displays podium predictions, and visualizes feature importance and race pace effects.
"""

import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


# Enable F1 data caching
fastf1.Cache.enable_cache("f1_cache")



# Grand Prix schedule with round numbers, dates, times, and coordinates for weather API
gp_schedule = {
    "Bahrain Grand Prix": {"round": 1, "date": "April 13", "time": "15:00 ", "lat": 26.0325, "lon": 50.5106},
    "Saudi Arabian Grand Prix": {"round": 2, "date": "April 20", "time": "17:00 ", "lat": 24.4686, "lon": 39.6111},
    "Australian Grand Prix": {"round": 3, "date": "March 16", "time": "04:00 ", "lat": -27.4969, "lon": 153.0170},
    "Japanese Grand Prix": {"round": 4, "date": "April 6", "time": "05:00 ", "lat": 35.8497, "lon": 139.2610},
    "Chinese Grand Prix": {"round": 5, "date": "March 23", "time": "07:00 ", "lat": 31.3389, "lon": 121.22},
    "Miami Grand Prix": {"round": 6, "date": "May 4", "time": "20:00 ", "lat": 25.9580, "lon": -80.2389},
    "Emilia Romagna Grand Prix": {"round": 7, "date": "May 18", "time": "13:00 ", "lat": 44.3439, "lon": 11.7167},
    "Monaco Grand Prix": {"round": 8, "date": "May 25", "time": "13:00 ", "lat": 43.7347, "lon": 7.4206},
    "Canadian Grand Prix": {"round": 9, "date": "June 15", "time": "18:00 ", "lat": 45.5033, "lon": -73.5673},
    "Spanish Grand Prix": {"round": 10, "date": "June 1", "time": "13:00 ", "lat": 41.5700, "lon": 2.2619},
    "Austrian Grand Prix": {"round": 11, "date": "June 29", "time": "13:00 ", "lat": 47.2197, "lon": 14.7647},
    "British Grand Prix": {"round": 12, "date": "July 6", "time": "14:00 ", "lat": 52.0786, "lon": -1.0169},
    "Hungarian Grand Prix": {"round": 13, "date": "August 3", "time": "13:00 ", "lat": 47.5789, "lon": 19.2486},
    "Belgian Grand Prix": {"round": 14, "date": "July 27", "time": "13:00 ", "lat": 50.4372, "lon": 5.9714},
    "Dh Grand Prix": {"round": 15, "date": "August 31", "time": "13:00 ", "lat": 51.9116, "lon": 4.1623},
    "Italian Grand Prix": {"round": 16, "date": "September 7", "time": "13:00 ", "lat": 45.6156, "lon": 9.2811},
    "Azerbaijan Grand Prix": {"round": 17, "date": "September 21", "time": "11:00 ", "lat": 40.3725, "lon": 49.8533},
    "Singapore Grand Prix": {"round": 18, "date": "October 5", "time": "12:00 ", "lat": 1.2914, "lon": 103.8642},
    "United States Grand Prix": {"round": 19, "date": "October 19", "time": "19:00 ", "lat": 36.0908, "lon": -115.1762},
    "Mexico City Grand Prix": {"round": 20, "date": "October 26", "time": "20:00 ", "lat": 19.4040, "lon": -99.0810},
    "São Paulo Grand Prix": {"round": 21, "date": "November 9", "time": "17:00 ", "lat": -23.7036, "lon": -46.6997},
    "Las Vegas Grand Prix": {"round": 22, "date": "November 23", "time": "04:00 ", "lat": 36.0908, "lon": -115.1762},
    "Qatar Grand Prix": {"round": 23, "date": "November 30", "time": "16:00 ", "lat": 25.2880, "lon": 51.4375},
    "Abu Dhabi Grand Prix": {"round": 24, "date": "December 7", "time": "13:00 ", "lat": 24.4672, "lon": 54.6031},
}

# Prompt user to select a Grand Prix for prediction
print("Select a Grand Prix for prediction:")
for i, gp in enumerate(gp_schedule.keys()):
    print(f"{i+1}. {gp}")
gp_choice_idx = int(input("Enter the number of the Grand Prix: ")) - 1
gp_choice = list(gp_schedule.keys())[gp_choice_idx]
gp_data = gp_schedule[gp_choice]

year = 2024  # Year for fetching historical FastF1 data
round_number = gp_data["round"]
lat = gp_data["lat"]
lon = gp_data["lon"]

# Fetch qualifying session data from FastF1
session_choice = "Q"  # Use qualifying session
session = fastf1.get_session(year, round_number, session_choice)
session.load()

laps_2024 = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# Convert timedelta columns to seconds for analysis
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Aggregate sector times by driver for historical performance
sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()

sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"] +
    sector_times_2024["Sector2Time (s)"] +
    sector_times_2024["Sector3Time (s)"]
)

# Pre-computed clean air race pace for each driver
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

# List of drivers and prompt for qualifying times
drivers = ["VER", "TSU", "NOR", "PIA", "RUS", "LEC", "HAM", "SAI", "ALB", "ALO", "STR", "OCO", "GAS", "HUL"]

qualifying_times = []
print("\nEnter qualifying times in seconds for each driver (or type 'DNF' if driver did not set a time):")
for driver in drivers:
    while True:
        time = input(f"{driver}: ")
        try:
            if time.strip().upper() == "DNF":
                qualifying_times.append(None)  # Handle DNF as None
            else:
                qualifying_times.append(float(time))
            break
        except ValueError:
            print("Invalid input! Enter a number in seconds or 'DNF'.")

# Impute DNF times with a penalty
max_time = max([t for t in qualifying_times if t is not None])
qualifying_times = [t if t is not None else max_time + 5 for t in qualifying_times]

# Create DataFrame for qualifying data
qualifying_2025 = pd.DataFrame({
    "Driver": drivers,
    "QualifyingTime (s)": qualifying_times
})

qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

import datetime

# Fetch weather forecast using OpenWeatherMap API
API_KEY = "6659192f0eeaf84f10720b9d60458a75"

lat = gp_data["lat"]
lon = gp_data["lon"]
gp_date_str = gp_data.get("date", "2025-05-18")  # Default date
gp_time_str = gp_data.get("time", "06:00").strip()

curr_year = 2025
gp_datetime = datetime.datetime.strptime(f"{curr_year} {gp_date_str} {gp_time_str}", "%Y %B %d %H:%M")

weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
response = requests.get(weather_url)
weather_data = response.json()

# Find forecast closest to GP time
forecast_data = min(weather_data["list"], key=lambda f: abs(datetime.datetime.strptime(f["dt_txt"], "%Y-%m-%d %H:%M:%S") - gp_datetime))

rain_probability = forecast_data.get("pop", 0)
temperature = forecast_data.get("main", {}).get("temp", 20)

print(f"Weather for {gp_choice} at {gp_datetime} -> Rain Probability: {rain_probability}, Temperature: {temperature}°C")

# Adjust qualifying times for wet conditions if rain is likely
if rain_probability >= 0.75:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"] * qualifying_2025["Driver"].map(wet_weather_factor)
else:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]

# Constructor standings points for team performance
team_points = {
    "McLaren": 246, "Mercedes": 141, "Red Bull": 105, "Williams": 37, "Ferrari": 94,
    "Haas": 20, "Aston Martin": 14, "Kick Sauber": 6, "Racing Bulls": 8, "Alpine": 7
}
max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

# Map drivers to teams
driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
    "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Racing Bulls",
    "SAI": "Ferrari", "HUL": "Kick Sauber", "OCO": "Alpine", "STR": "Aston Martin"
}
qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)

# Merge data and prepare features for ML model
merged_data = qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature
merged_data["QualifyingTime"] = merged_data["QualifyingTime"]

X = merged_data[["QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore", "CleanAirRacePace (s)"]]
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=34)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=34)
model.fit(X_train, y_train)
merged_data["PredictedRaceTime (s)"] = model.predict(X_imputed)

# Driver information for display
drivers_info = [
        {"CarNo":1,"Driver":"Max Verstappen","Team":"Red Bull Racing","Nationality":"Netherlands","Code":"VER"},
        {"CarNo":22,"Driver":"Yuki Tsunoda","Team":"Red Bull Racing","Nationality":"Japan","Code":"TSU"},
        {"CarNo":4,"Driver":"Lando Norris","Team":"McLaren","Nationality":"United Kingdom","Code":"NOR"},
        {"CarNo":81,"Driver":"Oscar Piastri","Team":"McLaren","Nationality":"Australia","Code":"PIA"},
        {"CarNo":63,"Driver":"George Russell","Team":"Mercedes","Nationality":"United Kingdom","Code":"RUS"},
        {"CarNo":16,"Driver":"Charles Leclerc","Team":"Ferrari","Nationality":"Monaco","Code":"LEC"},
        {"CarNo":44,"Driver":"Lewis Hamilton","Team":"Ferrari","Nationality":"United Kingdom","Code":"HAM"},
        {"CarNo":14,"Driver":"Fernando Alonso","Team":"Aston Martin","Nationality":"Spain","Code":"ALO"},
        {"CarNo":18,"Driver":"Lance Stroll","Team":"Aston Martin","Nationality":"Canada","Code":"STR"},
        {"CarNo":31,"Driver":"Esteban Ocon","Team":"Haas","Nationality":"France","Code":"OCO"},
        {"CarNo":55,"Driver":"Carlos Sainz","Team":"Williams","Nationality":"Spain","Code":"SAI"},
        {"CarNo":23,"Driver":"Alexander Albon","Team":"Williams","Nationality":"Thailand","Code":"ALB"},
        {"CarNo":27,"Driver":"Nico Hulkenberg","Team":"Kick Sauber","Nationality":"Germany","Code":"HUL"},
        {"CarNo":10,"Driver":"Pierre Gasly","Team":"Alpine","Nationality":"France","Code":"GAS"}
    ]

# Sort and display predicted results
final_results = merged_data.sort_values("PredictedRaceTime (s)")
print(f"\n🏁 Predicted {gp_choice} 2025 GP Winner 🏁\n")
print(final_results[["Driver", "PredictedRaceTime (s)"]])
y_pred = model.predict(X_test)
print(f"Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# Plot predicted race times vs clean air race pace
plt.figure(figsize=(12, 8))
plt.scatter(final_results["CleanAirRacePace (s)"], final_results["PredictedRaceTime (s)"])
for i, driver in enumerate(final_results["Driver"]):
    plt.annotate(driver, (final_results["CleanAirRacePace (s)"].iloc[i], final_results["PredictedRaceTime (s)"].iloc[i]),
                 xytext=(5, 5), textcoords='offset points')
plt.xlabel("clean air race pace (s)")
plt.ylabel("predicted race time (s)")
plt.title("effect of clean air race pace on predicted race results")
plt.tight_layout()
plt.show()

# Plot feature importance
feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance in Race Time Prediction")
plt.tight_layout()
plt.show()

# Get and display podium
final_results = merged_data.sort_values("PredictedRaceTime (s)").reset_index(drop=True)
podium = final_results.loc[:2, ["Driver", "PredictedRaceTime (s)"]]

print("\n🏆 Predicted Podium 🏆")
print(f"🥇 P1: {podium.iloc[0]['Driver']}")
print(f"🥈 P2: {podium.iloc[1]['Driver']}")
print(f"🥉 P3: {podium.iloc[2]['Driver']}")
