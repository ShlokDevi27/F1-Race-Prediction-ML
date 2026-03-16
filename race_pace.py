import fastf1
import pandas as pd
from tqdm import tqdm  # for progress bar

fastf1.Cache.enable_cache("f1_cache")

rounds_2024 = list(range(1, 24))  # list of rounds
all_lap_times = {}

for rnd in tqdm(rounds_2024, desc="Processing 2024 races"):
    try:
        session = fastf1.get_session(2024, rnd, 'R')
        session.load()

        laps = session.laps.pick_quicklaps().copy()
        laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()

        clean_laps = laps[
            (laps["PitOutTime"].isna()) &
            (laps["PitInTime"].isna()) &
            (laps["IsAccurate"])
        ]

        for driver, group in clean_laps.groupby("Driver"):
            if driver not in all_lap_times:
                all_lap_times[driver] = []
            all_lap_times[driver].extend(group["LapTime (s)"].tolist())

    except Exception as e:
        print(f"Skipping round {rnd} due to error: {e}")

average_race_pace = {driver: sum(times)/len(times) for driver, times in all_lap_times.items()}

race_pace_df = pd.DataFrame(list(average_race_pace.items()), columns=["Driver", "AvgRacePace (s)"])
race_pace_df = race_pace_df.sort_values("AvgRacePace (s)").reset_index(drop=True)

print("\n🏎 2024 Average Clean Air Race Pace per Driver 🏎")
print(race_pace_df)

# Save to CSV
race_pace_df.to_csv("race_pace.csv", index=False)
print("\n✅ Data saved to race_pace.csv")