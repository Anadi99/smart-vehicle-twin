# scripts/aggregate_laps.py
import pandas as pd
import os

HIST_PATH = "data/history.csv"
OUT_PATH  = "data/lap_features.csv"
TICK_SEC  = 2  # simulator tick duration

if not os.path.exists(HIST_PATH):
    raise FileNotFoundError(f"{HIST_PATH} not found. Run the simulator first to generate history.")

df = pd.read_csv(HIST_PATH)
# Basic cleaning
df = df.dropna(subset=["lap"])
df["lap"] = df["lap"].astype(int)

laps = []
for lap_id, g in df.groupby("lap"):
    if len(g) < 10:
        # too few ticks; skip partial laps
        continue

    g = g.reset_index(drop=True)
    ticks = len(g)
    duration_sec = ticks * TICK_SEC

    speed_mean = g["speed(km/h)"].mean()
    speed_max  = g["speed(km/h)"].max()
    temp_mean  = g["Temperature (°C)"].mean()
    temp_max   = g["Temperature (°C)"].max()

    soc_start = g["Battery SOC (%)"].iloc[0]
    soc_end   = g["Battery SOC (%)"].iloc[-1]
    soc_drop  = max(0.0, soc_start - soc_end)

    pad_start = g["Brake Pad (fraction)"].iloc[0]
    pad_end   = g["Brake Pad (fraction)"].iloc[-1]
    pad_wear  = max(0.0, pad_start - pad_end)

    laps.append({
        "lap": lap_id,
        "ticks": ticks,
        "duration_sec": duration_sec,
        "speed_mean": speed_mean,
        "speed_max":  speed_max,
        "temp_mean":  temp_mean,
        "temp_max":   temp_max,
        "soc_start": soc_start, "soc_end": soc_end, "soc_drop": soc_drop,
        "pad_start": pad_start, "pad_end": pad_end, "pad_wear": pad_wear,
    })

lap_df = pd.DataFrame(laps).sort_values("lap")
if lap_df.empty:
    raise RuntimeError("No complete laps found. Let the simulator run longer.")

lap_df.to_csv(OUT_PATH, index=False)
print(f"Wrote per-lap features -> {OUT_PATH}")
print(lap_df.tail(3))
