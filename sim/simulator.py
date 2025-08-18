# sim/simulator.py
import os
import csv
import random
from datetime import datetime

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
LAPS_FILE = os.path.join(DATA_DIR, "laps.csv")

def simulate_lap(lap_id: int):
    """Simulate one lap of telemetry data"""
    ticks = random.randint(50, 150)  # number of samples per lap
    speed = [random.uniform(80, 300) for _ in range(ticks)]
    temp = [random.uniform(70, 120) for _ in range(ticks)]

    duration = ticks * 0.5  # seconds (0.5 sec per tick)
    pad_wear = random.uniform(0.01, 0.05) * ticks
    soc_drop = random.uniform(0.05, 0.2)

    return {
        "lap_id": lap_id,
        "ticks": ticks,
        "speed_mean": sum(speed) / len(speed),
        "speed_max": max(speed),
        "temp_mean": sum(temp) / len(temp),
        "temp_max": max(temp),
        "duration_sec": duration,
        "pad_wear": pad_wear,
        "soc_drop": soc_drop,
        "timestamp": datetime.now().isoformat()
    }

def main():
    laps = []
    for lap_id in range(1, 21):  # simulate 20 laps
        laps.append(simulate_lap(lap_id))

    fieldnames = list(laps[0].keys())
    with open(LAPS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(laps)

    print(f"✅ Simulated {len(laps)} laps → {LAPS_FILE}")

if __name__ == "__main__":
    main()
