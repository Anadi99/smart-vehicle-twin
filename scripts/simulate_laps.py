# scripts/simulate_laps.py
import os, csv, random, time

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "lap_data.csv")

# columns: lap, ticks, speed, temp, tire_wear, soc
header = ["lap", "ticks", "speed", "temp", "tire_wear", "soc"]

# If file doesn't exist, create with header
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

lap = 0
while lap < 50:  # simulate 50 laps
    ticks = random.randint(80, 120)
    speed = random.uniform(180, 280)  # km/h
    temp = random.uniform(60, 120)    # °C
    tire_wear = max(0, 1 - lap * 0.015 + random.uniform(-0.01, 0.01))  # degrade
    soc = max(0, 100 - lap * 1.5 + random.uniform(-0.5, 0.5))          # battery drop %

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([lap, ticks, speed, temp, tire_wear, soc])

    lap += 1
    time.sleep(0.1)  # small delay to mimic real telemetry

print(f"✅ Simulation finished. Data saved to {CSV_PATH}")
