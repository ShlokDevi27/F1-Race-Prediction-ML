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

from flask import Flask, render_template, request
import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from xgboost import XGBRegressor

# Initialize Flask app
app = Flask(__name__)

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

# ---------------- Route 1: Display Input Form ----------------
@app.route('/')
def home():
    """
    Render the home page with the input form for selecting Grand Prix and entering qualifying times.

    Returns:
        str: Rendered HTML template for the home page.
    """
    return render_template('index.html', grand_prix_list=gp_schedule.keys(), drivers=drivers)

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

    qualifying_2025 = []

    # collect qualifying times from form
    for driver in drivers:
        dnf_flag = request.form.get(f"{driver}_dnf")
        if dnf_flag == "DNF":
            qualifying_2025.append(None)  # will impute later
        else:
            value = request.form.get(driver)
            qualifying_2025.append(float(value))
        
    # impute DNFs with max time + 5 penalty
    max_time = max([t for t in qualifying_2025 if t is not None])
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

    # ---------------- Weather API ----------------
    API_KEY = "6659192f0eeaf84f10720b9d60458a75"

    lat = gp_data["lat"]
    lon = gp_data["lon"]
    # GP date and time (from gp_schedule)
    # You can extend gp_schedule to include exact 'date' and 'time' strings
    gp_date_str = gp_data.get("date", "2025-05-18")  # fallback if date not set
    gp_time_str = gp_data.get("time", "06:00")   # fallback if time not set
    gp_time_str = gp_time_str.strip()

    curr_year = 2025
    # Convert to datetime object
    # Remove 'UTC' and parse
    # Correct datetime parsing
    gp_datetime = datetime.datetime.strptime(f"{curr_year} {gp_date_str} {gp_time_str}", "%Y %B %d %H:%M")


    # OpenWeatherMap API call
    weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(weather_url)
    weather_data = response.json()

    # Find the closest forecast entry to the GP datetime
    forecast_data = min(weather_data["list"], key=lambda f: abs(datetime.datetime.strptime(f["dt_txt"], "%Y-%m-%d %H:%M:%S") - gp_datetime))

    rain_probability = forecast_data.get("pop", 0)
    temperature = forecast_data.get("main", {}).get("temp", 20)

    print(f"Weather for {gp_choice} at {gp_datetime} -> Rain Probability: {rain_probability}, Temperature: {temperature}°C")


# ---------------- Adjust qualifying for weather (if needed) ----------------
    if rain_probability >= 0.75:
    # Multiply each driver's qualifying time by their wet weather factor
        qualifying_2025_df["QualifyingTime"] = qualifying_2025_df.apply(
        lambda row: row["QualifyingTime"] * wet_weather_factor.get(row["Driver"], 1), axis=1
    )
        print("Rain expected! Adjusted qualifying times applied.")
    else:
    # No rain: keep original qualifying times
        qualifying_2025_df["QualifyingTime"] = qualifying_2025_df["QualifyingTime"]
        print("No rain expected. Qualifying times remain unchanged.")

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
        "SAI": "Willams", "HUL": "Kick Sauber", "OCO": "Hass", "STR": "Aston Martin" , "ALB" : "Willams"
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
    final_results['QualifyingTime_display'] = final_results['QualifyingTime']
    final_results['PredictedRaceTime_display'] = final_results['PredictedRaceTime (s)']

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
    podium = final_results[final_results['PredictedRaceTime_display'] != "DNF"].head(3)
    podium = podium[["Driver", "Team", "PredictedRaceTime_display"]].copy()
    podium.rename(columns={"Driver": "driver", "Team": "team", "PredictedRaceTime_display": "time"}, inplace=True)
    podium['time'] = podium['time'].apply(lambda x: round(x, 2) if isinstance(x, (int, float, np.float64)) else x)

    # ---------------- Full Results ----------------
    full_results = final_results[["Driver", "Team", "QualifyingTime_display", "PredictedRaceTime_display"]].copy()
    full_results.rename(columns={
        "Driver": "driver",
        "Team": "team",
        "QualifyingTime_display": "qualifying",
        "PredictedRaceTime_display": "predicted"
    }, inplace=True)

    # Round numeric times only, leave DNFs as is
    full_results["qualifying"] = full_results["qualifying"].apply(lambda x: round(x,2) if isinstance(x,(int,float,np.float64)) else x)
    full_results["predicted"] = full_results["predicted"].apply(lambda x: round(x,2) if isinstance(x,(int,float,np.float64)) else x)

    full_results_list = full_results.to_dict(orient="records")

    # ---------------- Calculate MAE ----------------
    mae = mean_absolute_error(y, merged_data["PredictedRaceTime (s)"])
    mae = round(mae, 3)

    # ---------------- Results Dictionary ----------------
    results = {
        "gp_choice": gp_choice,
        "rain_probability": rain_probability,
        "temperature": temperature,
        "race_date": gp_data["date"],
        "podium": podium.to_dict(orient="records"),
        "full_results": full_results_list,
        "mae": mae,
        "metrics": {
            "Mean Error": round(mean_error, 2),
            "Accuracy": round(accuracy, 2),
            "Trees": model.n_estimators,
            "Learning Rate": model.learning_rate
        }
    }

    # ------------------- 5️⃣ Effect of Clean Air Race Pace -------------------
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(12,8))

    x = final_results["CleanAirRacePace (s)"]
    y = final_results["PredictedRaceTime (s)"]

    ax.scatter(x, y, color='#1d3557', s=100)

    # Annotate each driver
    for i, driver in enumerate(final_results["Driver"]):
        ax.annotate(driver, (x.iloc[i], y.iloc[i]), xytext=(5,5), textcoords='offset points', fontsize=10, fontweight='bold')

    ax.set_xlabel("Clean Air Race Pace (s)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Predicted Race Time (s)", fontsize=12, fontweight='bold')
    ax.set_title("Effect of Clean Air Race Pace on Predicted Race Results", fontsize=14, fontweight='bold', color='#f77f00')
    ax.grid(True, linestyle='--', alpha=0.5)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    cleanair_effect_chart = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    # ------------------- 1️⃣ Predicted vs Qualifying Times -------------------
    
    
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(10,6))

    x = merged_data["Driver"]
    qual_times = merged_data["QualifyingTime"]
    pred_times = merged_data["PredictedRaceTime (s)"]

    bar_width = 0.35
    ax.bar(x, qual_times, width=bar_width, label='Qualifying Time', color='#f77f00', alpha=0.8)
    ax.bar([i for i in range(len(x))], pred_times, width=bar_width, label='Predicted Race Time', color='#e63946', alpha=0.8, align='edge')

    ax.set_xlabel("Driver", fontsize=12, fontweight='bold')
    ax.set_ylabel("Time (s)", fontsize=12, fontweight='bold')
    ax.set_title("Predicted vs Qualifying Times", fontsize=14, fontweight='bold', color='#f77f00')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    pred_vs_qual_chart = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # ------------------- 2️⃣ Feature Importance -------------------
    sns.set_style("darkgrid")
    features = ["QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore", "CleanAirRacePace (s)"]
    importances = model.feature_importances_

    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(features, importances, color='#457b9d', alpha=0.85)
    ax.set_xlabel("Importance", fontsize=12, fontweight='bold')
    ax.set_title("Feature Importance", fontsize=14, fontweight='bold', color='#f77f00')
    for i, v in enumerate(importances):
        ax.text(v + 0.005, i, f"{v:.2f}", color='black', fontweight='bold')

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    importance_chart = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # ------------------- 3️⃣ Clean Air vs Predicted Race Time -------------------
    sns.set_style("darkgrid")
    sorted_data = merged_data.sort_values("CleanAirRacePace (s)")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(sorted_data["Driver"], sorted_data["CleanAirRacePace (s)"], marker='o', linestyle='--', color='#1d3557', label='Clean Air Pace')
    ax.plot(sorted_data["Driver"], sorted_data["PredictedRaceTime (s)"], marker='s', linestyle='-', color='#e63946', label='Predicted Race Time')

    ax.set_xlabel("Driver", fontsize=12, fontweight='bold')
    ax.set_ylabel("Time (s)", fontsize=12, fontweight='bold')
    ax.set_title("Clean Air vs Predicted Race Time", fontsize=14, fontweight='bold', color='#f77f00')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    cleanair_chart = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # ------------------- Pass charts to results dict -------------------
    results["charts"] = {
        "pred_vs_qual_chart": pred_vs_qual_chart,
        "importance_chart": importance_chart,
        "cleanair_chart": cleanair_chart,
        "cleanair_effect_chart" : cleanair_effect_chart
    }
    # Pass results to template
    return render_template('index.html', grand_prix_list=gp_schedule.keys(), drivers=drivers, results=results)

# ---------------- Run Flask App ----------------
if __name__ == '__main__':
    app.run(debug=True)
