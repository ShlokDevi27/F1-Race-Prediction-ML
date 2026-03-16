import fastf1
import pandas as pd
import numpy as np

# Enable FastF1 cache (so we don't re-download every time)
fastf1.Cache.enable_cache("f1_cache")

SEASON = 2024
ROUNDS = range(1, 23)   # 22 races in 2024 season (adjust if different)

# Detect if lap is wet
def is_wet_lap(lap):
    tyre = lap.get('Compound', None)
    if tyre in ["WET", "INTERMEDIATE"]:
        return True
    # fallback heuristic: very slow lap might be wet
    lap_time = lap["LapTime"].total_seconds() if pd.notna(lap["LapTime"]) else None
    if lap_time and lap_time > 110:
        return True
    return False

all_data = []

for rnd in ROUNDS:
    try:
        session = fastf1.get_session(SEASON, rnd, 'R')  # Race
        session.load(weather=True, laps=True)

        laps = session.laps.pick_quicklaps()  # ignore pit/out laps
        for drv in session.drivers:
            drv_laps = laps.pick_driver(drv)
            wet_mask = drv_laps.apply(is_wet_lap, axis=1)

            n_wet = wet_mask.sum()
            n_dry = len(drv_laps) - n_wet

            mean_wet = drv_laps[wet_mask]["LapTime"].dropna().dt.total_seconds().mean()
            mean_dry = drv_laps[~wet_mask]["LapTime"].dropna().dt.total_seconds().mean()

            drv_info = session.get_driver(drv)

            # Penalize if retired in a wet race
            dnf = 0
            if drv_info["Status"] != "Finished" and n_wet > 0:
                dnf = 1

            all_data.append({
                "Driver": drv_info["Abbreviation"],
                "Race": session.event["EventName"],
                "n_wet": n_wet,
                "n_dry": n_dry,
                "mean_wet": mean_wet,
                "mean_dry": mean_dry,
                "dnf": dnf
            })
        print(f"✅ Processed {session.event['EventName']}")
    except Exception as e:
        print(f"⚠️ Skipped round {rnd}: {e}")

df = pd.DataFrame(all_data)

# Aggregate by driver
agg = df.groupby("Driver").apply(lambda g: pd.Series({
    "n_wet": int(g["n_wet"].sum()),
    "mean_wet": np.nansum(g["mean_wet"] * g["n_wet"]) / g["n_wet"].sum() if g["n_wet"].sum() > 0 else np.nan,
    "mean_dry": np.nansum(g["mean_dry"] * g["n_dry"]) / g["n_dry"].sum() if g["n_dry"].sum() > 0 else np.nan,
    "dnf_count": g["dnf"].sum()
})).reset_index()

# Experience score (main weight, grows with wet laps)
agg["exp_score"] = 1 + np.log10(1 + agg["n_wet"])

# Performance score (lap time ratio, smaller importance)
agg["perf_score"] = agg["mean_wet"] / agg["mean_dry"]

# Final wet factor = experience × performance
agg["wet_factor_adjusted"] = agg["exp_score"] * agg["perf_score"]

# Apply DNF penalty (2% per wet DNF)
agg["wet_factor_adjusted"] *= (1 + 0.02 * agg["dnf_count"])

# Normalize wet factor to be below 1
max_value = agg["wet_factor_adjusted"].max()
if max_value > 0:
    agg["wet_factor_adjusted"] = agg["wet_factor_adjusted"] / (max_value + 0.001)  # avoids exact 1

# Save results
agg.to_csv("wet_pace.csv", index=False)

print("\n✅ Wet pace factors saved to wet_pace.csv")
print(agg[["Driver", "n_wet", "exp_score", "perf_score", "wet_factor_adjusted"]].head())
