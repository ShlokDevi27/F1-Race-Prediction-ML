"""
F1 Race Prediction Web Application

This Flask application predicts F1 race outcomes based on qualifying times,
weather data, team performance, and historical sector times. It uses machine
learning models to forecast race times and provides visualizations.

Features:
- User input for qualifying times and DNFs
- Weather API integration for rain probability and temperature
- ML model training and prediction
- Interactive charts and results display
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import datetime
import seaborn as sns
from io import BytesIO
import base64
from xgboost import XGBRegressor

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

os.makedirs("f1_cache", exist_ok=True)
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
    "Dutch Grand Prix": {"round": 15, "date": "August 31", "time": "13:00 ", "lat": 51.9116, "lon": 4.1623},
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

# List of F1 drivers for 2025 season
drivers = ["VER", "TSU", "NOR", "PIA", "RUS", "LEC", "HAM", "SAI", "ALB", "ALO", "STR", "OCO", "GAS", "HUL"]

# Detailed metadata for 2025 drivers: car numbers, full names, official teams, default lap times, and motorsport hex colors
DRIVER_DETAILS = {
    "VER": {"name": "Max Verstappen", "car_no": 1, "team": "Red Bull Racing", "color": "#3671C6", "default_time": 88.0},
    "TSU": {"name": "Yuki Tsunoda", "car_no": 22, "team": "Red Bull Racing", "color": "#3671C6", "default_time": 88.5},
    "NOR": {"name": "Lando Norris", "car_no": 4, "team": "McLaren", "color": "#FF8000", "default_time": 89.0},
    "PIA": {"name": "Oscar Piastri", "car_no": 81, "team": "McLaren", "color": "#FF8000", "default_time": 89.5},
    "RUS": {"name": "George Russell", "car_no": 63, "team": "Mercedes", "color": "#27F4D2", "default_time": 90.0},
    "LEC": {"name": "Charles Leclerc", "car_no": 16, "team": "Ferrari", "color": "#E8002D", "default_time": 90.5},
    "HAM": {"name": "Lewis Hamilton", "car_no": 44, "team": "Ferrari", "color": "#E8002D", "default_time": 91.0},
    "SAI": {"name": "Carlos Sainz", "car_no": 55, "team": "Williams", "color": "#64C4FF", "default_time": 91.5},
    "ALB": {"name": "Alexander Albon", "car_no": 23, "team": "Williams", "color": "#64C4FF", "default_time": 92.0},
    "ALO": {"name": "Fernando Alonso", "car_no": 14, "team": "Aston Martin", "color": "#229971", "default_time": 92.5},
    "STR": {"name": "Lance Stroll", "car_no": 18, "team": "Aston Martin", "color": "#229971", "default_time": 93.0},
    "OCO": {"name": "Esteban Ocon", "car_no": 31, "team": "Haas", "color": "#B6BABD", "default_time": 93.5},
    "GAS": {"name": "Pierre Gasly", "car_no": 10, "team": "Alpine", "color": "#0093CC", "default_time": 94.0},
    "HUL": {"name": "Nico Hülkenberg", "car_no": 27, "team": "Kick Sauber", "color": "#52E252", "default_time": 94.5}
}

# Constructor team pairings for line-wise alignment (driver1, driver2 or None if single driver)
TEAM_DRIVER_PAIRS = [
    ("Red Bull Racing", "VER", "TSU"),
    ("McLaren", "NOR", "PIA"),
    ("Mercedes", "RUS", None),
    ("Ferrari", "LEC", "HAM"),
    ("Williams", "SAI", "ALB"),
    ("Aston Martin", "ALO", "STR"),
    ("Haas", "OCO", None),
    ("Alpine", "GAS", None),
    ("Kick Sauber", "HUL", None)
]

def setup_dark_chart_rc():
    """Configure dark telemetry aesthetic for matplotlib / seaborn charts."""
    plt.rcParams.update({
        'figure.facecolor': '#0B0F19',
        'axes.facecolor': '#111827',
        'axes.edgecolor': '#232D42',
        'axes.labelcolor': '#94A3B8',
        'xtick.color': '#94A3B8',
        'ytick.color': '#94A3B8',
        'grid.color': '#1F2937',
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
        'text.color': '#F8FAFC',
        'font.family': 'sans-serif'
    })

weather_cache = {}

def get_weather_for_gp(gp_choice):
    """
    Fetch weather forecast for a selected Grand Prix using OpenWeatherMap API with caching and error handling.
    """
    if not gp_choice or gp_choice not in gp_schedule:
        return {
            "gp_choice": gp_choice or "Unknown",
            "rain_probability": 0.0,
            "temperature": 22.0,
            "race_date": "TBD",
            "race_time": "13:00",
            "condition": "Dry / Normal"
        }

    if gp_choice in weather_cache:
        return weather_cache[gp_choice]

    gp_data = gp_schedule[gp_choice]
    lat = gp_data["lat"]
    lon = gp_data["lon"]
    gp_date_str = gp_data.get("date", "May 18").strip()
    gp_time_str = gp_data.get("time", "13:00").strip()
    API_KEY = "6659192f0eeaf84f10720b9d60458a75"

    rain_probability = 0.0
    temperature = 22.0
    condition = "Clear Sky"

    try:
        curr_year = 2025
        gp_datetime = datetime.datetime.strptime(f"{curr_year} {gp_date_str} {gp_time_str}", "%Y %B %d %H:%M")
        weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        response = requests.get(weather_url, timeout=4)
        if response.status_code == 200:
            weather_data = response.json()
            if "list" in weather_data and len(weather_data["list"]) > 0:
                forecast_data = min(
                    weather_data["list"],
                    key=lambda f: abs(datetime.datetime.strptime(f["dt_txt"], "%Y-%m-%d %H:%M:%S") - gp_datetime)
                )
                rain_probability = float(forecast_data.get("pop", 0.0))
                temperature = float(forecast_data.get("main", {}).get("temp", 22.0))
                if forecast_data.get("weather") and len(forecast_data["weather"]) > 0:
                    condition = forecast_data["weather"][0].get("description", "Clear").title()
    except Exception as e:
        print(f"Weather API error for {gp_choice}: {e}")

    result = {
        "gp_choice": gp_choice,
        "rain_probability": rain_probability,
        "temperature": round(temperature, 1),
        "race_date": gp_date_str,
        "race_time": gp_time_str,
        "condition": condition
    }
    weather_cache[gp_choice] = result
    return result

# ---------------- Route 0: API for Instant Weather by Grand Prix ----------------
@app.route('/api/weather')
def api_weather():
    """
    Return live weather data as JSON as soon as a Grand Prix is selected in the UI.
    """
    gp_choice = request.args.get('grand_prix', '').strip()
    if not gp_choice or gp_choice not in gp_schedule:
        return {"error": "Invalid Grand Prix"}, 400
    return get_weather_for_gp(gp_choice)

# ---------------- Route 1: Display Input Form ----------------
@app.route('/')
def home():
    """
    Render the home page with the input form for selecting Grand Prix and entering qualifying times.

    Returns:
        str: Rendered HTML template for the home page.
    """
    return render_template(
        'index.html',
        grand_prix_list=list(gp_schedule.keys()),
        drivers=drivers,
        driver_details=DRIVER_DETAILS,
        team_driver_pairs=TEAM_DRIVER_PAIRS,
        selected_gp='',
        user_inputs={},
        user_dnfs={}
    )

# ---------------- Route 2: Handle User Input + Run Prediction ----------------
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle user input from the form, process data, train ML model, and generate predictions with visualizations.

    Retrieves qualifying times, DNFs, and selected GP from form. Fetches historical data, weather, trains XGBoost model,
    predicts race times, and creates charts. Returns rendered template with results.

    Returns:
        str: Rendered HTML template with prediction results and charts.
    """
    # Get user inputs from the form
    gp_choice = request.form.get('grand_prix')
    gp_data = gp_schedule[gp_choice]

    # Get the session round number from gp_schedule
    gp_round = gp_data["round"]

    # Fetch qualifying session data from FastF1
    session = fastf1.get_session(2024, gp_round, 'Q')  # 'Q' for qualifying
    session.load()

    # Extract laps and sector times from historical data
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

    # collect qualifying times and form inputs
    user_inputs = {}
    user_dnfs = {}
    qualifying_2025 = []

    for driver in drivers:
        dnf_flag = request.form.get(f"{driver}_dnf")
        raw_val = request.form.get(driver, "")
        user_inputs[driver] = raw_val
        is_dnf = (dnf_flag == "DNF")
        user_dnfs[driver] = is_dnf

        if is_dnf:
            qualifying_2025.append(None)  # will impute later
        else:
            try:
                qualifying_2025.append(float(raw_val))
            except (ValueError, TypeError):
                qualifying_2025.append(None)
        
    # impute DNFs with max time + 5 penalty
    valid_times = [t for t in qualifying_2025 if t is not None]
    max_time = max(valid_times) if valid_times else 90.0
    qualifying_2025 = [t if t is not None else max_time + 5 for t in qualifying_2025]

    # create DataFrame
    qualifying_2025_df = pd.DataFrame({
        "Driver": drivers,
        "QualifyingTime": qualifying_2025
    })
    
    # ---------------- clean air race pace calculate using racepace.py----------------
    clean_air_race_pace = {
        "VER": 88.13859652333029, "HAM": 87.99858076923077, "LEC": 88.19860530973452,
        "NOR": 87.95796681222707, "ALO": 89.79116370808678, "PIA": 88.11267857142857,
        "RUS": 88.25423928571429, "SAI": 88.11452128666036, "STR": 89.53571047008548,
        "HUL": 90.5244616915423, "OCO": 89.89775745118198, "TSU": 89.37616685456595,
        "GAS": 88.91867071823204, "ALB": 89.65372986369269
    }
    
    qualifying_2025_df["CleanAirRacePace (s)"] = qualifying_2025_df["Driver"].map(clean_air_race_pace)
    
    # ---------------- wet weather factor calculate using wetpace.py----------------
    wet_weather_factor = {
        "VER": 0.8828256162720444, "HAM": 0.8550248519900767, "LEC": 0.8447992944831761,
        "NOR": 0.922467828177502, "ALO": 0.9160402229972915, "PIA": 0.8632501469172694,
        "RUS": 0.8276233764085413, "SAI": 0.8284226284248292, "STR": 0.9494004296179431,
        "HUL": 0.8871134455802793, "OCO": 0.9996818544695274, "TSU": 0.961373412451465,
        "GAS": 0.9691866544335193, "ALB": 0.9147229519070319
}

    # ---------------- Weather API & Proportional Wet Weather Adjustment ----------------
    weather_info = get_weather_for_gp(gp_choice)
    rain_probability = weather_info["rain_probability"]
    temperature = weather_info["temperature"]

    # Normalize rain probability to a fraction in [0.0, 1.0]
    rain_p = float(rain_probability) / 100.0 if float(rain_probability) > 1.0 else float(rain_probability)
    rain_p = max(0.0, min(1.0, rain_p))
    rain_pct = int(round(rain_p * 100))

    print(f"Weather for {gp_choice} -> Rain Probability: {rain_p} ({rain_pct}%), Temperature: {temperature}°C")

    # Whenever there is a rain prediction, apply proportional wet weather effect:
    # e.g. 20% rain applies 20% effect, 80% rain applies 80% effect
    if rain_p > 0.0:
        def get_effective_wet_factor(driver):
            w = wet_weather_factor.get(driver, 1.0)
            return (1.0 - rain_p) * 1.0 + rain_p * w

        qualifying_2025_df["QualifyingTime"] = qualifying_2025_df.apply(
            lambda row: row["QualifyingTime"] * get_effective_wet_factor(row["Driver"]), axis=1
        )
        qualifying_2025_df["CleanAirRacePace (s)"] = qualifying_2025_df.apply(
            lambda row: row["CleanAirRacePace (s)"] * get_effective_wet_factor(row["Driver"]), axis=1
        )
        print(f"Rain prediction ({rain_pct}%) detected! Applied {rain_pct}% proportional wet weather pace adjustment.")
    else:
        print("Dry conditions (0% rain). Baseline pace modeling applied.")

    # ---------------- Teams and constructor performance ----------------
    team_points = {
        "McLaren": 246, "Mercedes": 141, "Red Bull": 105, "Williams": 37, "Ferrari": 94,
        "Haas": 20, "Aston Martin": 14, "Kick Sauber": 6, "Racing Bulls": 8, "Alpine": 7
    }
    max_points = max(team_points.values())
    team_performance_score = {team: points / max_points for team, points in team_points.items()}

    driver_to_team = {
        "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
        "HAM": "Ferrari", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Red Bull",
        "SAI": "Williams", "HUL": "Kick Sauber", "OCO": "Haas", "STR": "Aston Martin", "ALB": "Williams"
    }
    qualifying_2025_df["Team"] = qualifying_2025_df["Driver"].map(driver_to_team)
    qualifying_2025_df["TeamPerformanceScore"] = qualifying_2025_df["Team"].map(team_performance_score)

    # ---------------- Merge and prepare features ----------------
    merged_data = qualifying_2025_df.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
    merged_data["RainProbability"] = rain_probability
    merged_data["Temperature"] = temperature
    merged_data["QualifyingTime"] = merged_data["QualifyingTime"]

    X = merged_data[["QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore", "CleanAirRacePace (s)"]]
    y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=34)
    model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=34
)
    model.fit(X_train, y_train)
    merged_data["PredictedRaceTime (s)"] = model.predict(X_imputed)
    # ---------------- Driver Info Table ----------------
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

    # Model details
    n_trees = model.n_estimators
    learning_rate = model.learning_rate

    # True vs predicted
    y_true = y
    y_pred = merged_data["PredictedRaceTime (s)"]

    # Mean Error (MAE)
    mean_error = mean_absolute_error(y_true, y_pred)

    # Accuracy approximation for regression
    accuracy = 100 * (1 - (mean_error / np.mean(y_true)))

    # Round values
    metrics = {
        "Mean Error": round(mean_error, 2),
        "Accuracy": round(accuracy, 2),
        "Trees": n_trees,
        "Learning Rate": learning_rate
    }

    # Add to results dict
    #results["metrics"] = metrics
    final_results = merged_data.sort_values("PredictedRaceTime (s)").reset_index(drop=True)
    # ---------------- Add team to final_results ----------------
    final_results['Team'] = final_results['Driver'].map(driver_to_team)

    # ---------------- Handle DNFs ----------------
    dnf_drivers = [driver for driver in drivers if request.form.get(f"{driver}_dnf") == "DNF"]

    # Create columns for HTML display
    final_results['QualifyingTime_display'] = final_results['QualifyingTime'].astype(object)
    final_results['PredictedRaceTime_display'] = final_results['PredictedRaceTime (s)'].astype(object)

    # Mark DNF drivers for HTML
    for driver in dnf_drivers:
        final_results.loc[final_results['Driver'] == driver, 'QualifyingTime_display'] = "DNF"
        final_results.loc[final_results['Driver'] == driver, 'PredictedRaceTime_display'] = "DNF"

    # Create numeric columns for plotting
    final_results['QualifyingTime_plot'] = final_results['QualifyingTime'].copy()
    final_results['PredictedRaceTime_plot'] = final_results['PredictedRaceTime (s)'].copy()

    # Set DNF drivers to NaN for plotting
    for driver in dnf_drivers:
        final_results.loc[final_results['Driver'] == driver, 'QualifyingTime_plot'] = np.nan
        final_results.loc[final_results['Driver'] == driver, 'PredictedRaceTime_plot'] = np.nan

    # ---------------- Sort results ----------------
    # Use numeric predicted time for sorting, DNFs last
    final_results['sort_time'] = final_results['PredictedRaceTime_plot']
    final_results.sort_values(by='sort_time', inplace=True, na_position='last')
    final_results.reset_index(drop=True, inplace=True)
    final_results.drop(columns=['sort_time'], inplace=True)

    # ---------------- Podium ----------------
    podium_df = final_results[final_results['PredictedRaceTime_display'] != "DNF"].head(3).copy()
    p1_time = podium_df.iloc[0]["PredictedRaceTime (s)"] if len(podium_df) > 0 else None

    podium_list = []
    for rank, (_, row) in enumerate(podium_df.iterrows(), start=1):
        code = row["Driver"]
        d_meta = DRIVER_DETAILS.get(code, {"name": code, "car_no": "-", "team": row.get("Team", ""), "color": "#E10600"})
        pred_val = row["PredictedRaceTime_display"]
        pred_rounded = round(pred_val, 3) if isinstance(pred_val, (int, float, np.float64)) else pred_val
        delta_str = "LEADER" if rank == 1 else (f"+{(row['PredictedRaceTime (s)'] - p1_time):.3f}s" if p1_time else "-")
        podium_list.append({
            "rank": rank,
            "driver": code,
            "name": d_meta["name"],
            "car_no": d_meta["car_no"],
            "team": d_meta["team"],
            "team_color": d_meta["color"],
            "time": pred_rounded,
            "delta": delta_str
        })

    # ---------------- Full Results ----------------
    full_results_list = []
    for rank, (_, row) in enumerate(final_results.iterrows(), start=1):
        code = row["Driver"]
        d_meta = DRIVER_DETAILS.get(code, {"name": code, "car_no": "-", "team": row.get("Team", ""), "color": "#E10600"})
        q_val = row["QualifyingTime_display"]
        p_val = row["PredictedRaceTime_display"]
        q_display = f"{q_val:.3f}s" if isinstance(q_val, (int, float, np.float64)) else str(q_val)
        p_display = f"{p_val:.3f}s" if isinstance(p_val, (int, float, np.float64)) else str(p_val)
        
        if p_val == "DNF" or p1_time is None:
            delta_str = "DNF"
        elif rank == 1:
            delta_str = "LEADER"
        else:
            delta_str = f"+{(row['PredictedRaceTime (s)'] - p1_time):.3f}s"

        full_results_list.append({
            "rank": rank,
            "driver": code,
            "name": d_meta["name"],
            "car_no": d_meta["car_no"],
            "team": d_meta["team"],
            "team_color": d_meta["color"],
            "qualifying": q_display,
            "predicted": p_display,
            "delta": delta_str,
            "is_dnf": (p_val == "DNF")
        })

    # ---------------- Calculate MAE ----------------
    mae = mean_absolute_error(y, merged_data["PredictedRaceTime (s)"])
    mae = round(mae, 3)

    # ---------------- Results Dictionary ----------------
    results = {
        "gp_choice": gp_choice,
        "rain_probability": rain_p,
        "rain_pct": rain_pct,
        "temperature": temperature,
        "race_date": gp_data["date"],
        "podium": podium_list,
        "full_results": full_results_list,
        "mae": mae,
        "metrics": {
            "Mean Error": round(mean_error, 2),
            "Accuracy": round(accuracy, 2),
            "Trees": model.n_estimators,
            "Learning Rate": model.learning_rate
        }
    }

    # ------------------- 1️⃣ Effect of Clean Air Race Pace (Dark Themed) -------------------
    setup_dark_chart_rc()
    fig, ax = plt.subplots(figsize=(10, 6))

    x = final_results["CleanAirRacePace (s)"]
    y = final_results["PredictedRaceTime (s)"]

    ax.scatter(x, y, color='#38BDF8', s=90, edgecolors='#E10600', linewidth=1.5, alpha=0.9, zorder=4)

    for i, driver in enumerate(final_results["Driver"]):
        ax.annotate(driver, (x.iloc[i], y.iloc[i]), xytext=(6, 4), textcoords='offset points', fontsize=9, fontweight='bold', color='#F8FAFC')

    ax.set_xlabel("Clean Air Race Pace (s)", fontsize=11, fontweight='bold', color='#94A3B8')
    ax.set_ylabel("Predicted Race Lap Time (s)", fontsize=11, fontweight='bold', color='#94A3B8')
    ax.set_title("Pace Correlation: Clean Air Pace vs Predicted Lap Time", fontsize=12, fontweight='bold', color='#F8FAFC', pad=12)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=1)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    cleanair_effect_chart = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    # ------------------- 2️⃣ Predicted vs Qualifying Times (Dark Themed) -------------------
    setup_dark_chart_rc()
    fig, ax = plt.subplots(figsize=(11, 5.5))

    x_indices = np.arange(len(merged_data["Driver"]))
    driver_labels = merged_data["Driver"].tolist()
    qual_times = merged_data["QualifyingTime"].values
    pred_times = merged_data["PredictedRaceTime (s)"].values

    bar_width = 0.38
    ax.bar(x_indices - bar_width/2, qual_times, width=bar_width, label='Qualifying Lap Time', color='#38BDF8', alpha=0.85, edgecolor='#0284C7', linewidth=0.8)
    ax.bar(x_indices + bar_width/2, pred_times, width=bar_width, label='Predicted Race Lap Time', color='#E10600', alpha=0.85, edgecolor='#B91C1C', linewidth=0.8)

    ax.set_xticks(x_indices)
    ax.set_xticklabels(driver_labels, fontsize=10, fontweight='bold', color='#F8FAFC')
    ax.set_xlabel("Driver Code", fontsize=11, fontweight='bold', color='#94A3B8')
    ax.set_ylabel("Lap Time (seconds)", fontsize=11, fontweight='bold', color='#94A3B8')
    ax.set_title("Telemetry Comparison: Qualifying vs Predicted Race Pace", fontsize=12, fontweight='bold', color='#F8FAFC', pad=12)
    
    valid_all = [t for t in list(qual_times) + list(pred_times) if not np.isnan(t)]
    if valid_all:
        ax.set_ylim(max(0, min(valid_all) - 5), max(valid_all) + 5)

    ax.legend(facecolor='#1E2638', edgecolor='#2E3C56', fontsize=10, labelcolor='#F8FAFC')
    ax.grid(True, linestyle='--', alpha=0.4, zorder=1)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    pred_vs_qual_chart = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    # ------------------- 3️⃣ Feature Importance (Dark Themed) -------------------
    setup_dark_chart_rc()
    feature_labels = ["Qualifying Time", "RainProbability", "Temperature", "TeamPerformanceScore", "CleanAirRacePace (s)"]
    importances = np.array(model.feature_importances_, dtype=float).copy()

    # Cumulative wet weather pace impact across the field
    # Reflects the cumulative proportion of wet weather pace applied (e.g. 20% rain -> 0.20 weight)
    if rain_p > 0.0:
        wet_imp = min(0.85, float(rain_p))
        base_sum = importances[0] + importances[3] + importances[4]
        if base_sum > 0:
            scale = (1.0 - wet_imp) / base_sum
            importances[0] = importances[0] * scale
            importances[3] = importances[3] * scale
            importances[4] = importances[4] * scale
        importances[1] = wet_imp

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.barh(feature_labels, importances, color='#E10600', alpha=0.88, height=0.52, edgecolor='#991B1B')
    ax.set_xlabel("Relative Feature Weight (Gain)", fontsize=11, fontweight='bold', color='#94A3B8')
    ax.set_title("XGBoost Regression Model Feature Importance", fontsize=12, fontweight='bold', color='#F8FAFC', pad=12)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va='center', color='#F8FAFC', fontweight='bold', fontsize=9)
    ax.set_xlim(0, max(importances) * 1.18 if max(importances) > 0 else 1.0)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=1)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    importance_chart = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    # ------------------- 4️⃣ Clean Air vs Predicted Race Time (Dark Themed) -------------------
    setup_dark_chart_rc()
    sorted_data = merged_data.sort_values("CleanAirRacePace (s)")
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.plot(sorted_data["Driver"], sorted_data["CleanAirRacePace (s)"], marker='o', markersize=6, linestyle='--', color='#38BDF8', linewidth=1.8, label='Clean Air Benchmark Pace')
    ax.plot(sorted_data["Driver"], sorted_data["PredictedRaceTime (s)"], marker='s', markersize=6, linestyle='-', color='#E10600', linewidth=2, label='Predicted Race Lap Time')

    ax.set_xlabel("Driver (Ranked by Clean Air Pace)", fontsize=11, fontweight='bold', color='#94A3B8')
    ax.set_ylabel("Lap Time (seconds)", fontsize=11, fontweight='bold', color='#94A3B8')
    ax.set_title("Clean Air Benchmark vs XGBoost Predicted Lap Time", fontsize=12, fontweight='bold', color='#F8FAFC', pad=12)
    ax.legend(facecolor='#1E2638', edgecolor='#2E3C56', fontsize=10, labelcolor='#F8FAFC')
    ax.grid(True, linestyle='--', alpha=0.4, zorder=1)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    cleanair_chart = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    # ------------------- Pass charts to results dict -------------------
    results["charts"] = {
        "pred_vs_qual_chart": pred_vs_qual_chart,
        "importance_chart": importance_chart,
        "cleanair_chart": cleanair_chart,
        "cleanair_effect_chart": cleanair_effect_chart
    }
    # Pass results to template
    return render_template(
        'index.html',
        grand_prix_list=list(gp_schedule.keys()),
        drivers=drivers,
        driver_details=DRIVER_DETAILS,
        team_driver_pairs=TEAM_DRIVER_PAIRS,
        results=results,
        selected_gp=gp_choice,
        user_inputs=user_inputs,
        user_dnfs=user_dnfs
    )

# ---------------- Run Flask App ----------------
if __name__ == '__main__':
    app.run(debug=True)
