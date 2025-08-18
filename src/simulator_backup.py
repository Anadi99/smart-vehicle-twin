import json, random, time, os
import pandas as pd

CONFIG_PATH = "configs/bmw_i4.json"
DATA_DIR = "data"
LATEST_PATH = os.path.join(DATA_DIR, "latest.csv")
HIST_PATH   = os.path.join(DATA_DIR, "history.csv")

os.makedirs(DATA_DIR, exist_ok=True)

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

model = cfg.get("model", "Unknown")
temp_max_C = float(cfg.get("temp_max_C", 90))
brake_wear_threshold = float(cfg.get("brake_wear_threshold", 0.3))

print(f"Starting telemetry simulator for {model} with lap + speed logic...")

# --- lap model (each lap ~ 90 sec at 2s ticks = 45 ticks) ---
LAP_TICKS = 45
TICK_SEC  = 2

# braking zones inside a lap
BRAKE_ZONES = {8, 9, 10, 22, 23, 24, 36, 37, 38}
STRAIGHTS   = {3, 4, 5, 15, 16, 17, 28, 29, 30}  # high speed zones

lap = 1
tick_in_lap = 0

# initialize states
battery_soc = random.uniform(70, 100)
brake_pad_fraction = random.uniform(0.6, 1.0)
temperature = random.uniform(30, 45)
speed = 0.0  # km/h

def risk_and_reason(temp_c, pad_frac, battery, spd):
    reasons = []
    risk = 0.0

    # temperature
    if temp_c > temp_max_C:
        over = min((temp_c - temp_max_C) / 30.0, 1.0)
        risk += 0.6 * over
        reasons.append(f"temperature high ({temp_c:.1f}°C>{temp_max_C}°C)")
    elif temp_c > (temp_max_C - 10):
        risk += 0.3
        reasons.append(f"temperature nearing limit ({temp_c:.1f}°C)")

    # brake wear
    if pad_frac < brake_wear_threshold:
        under = min((brake_wear_threshold - pad_frac) / 0.3, 1.0)
        risk += 0.5 * under
        reasons.append(f"brake pad low ({pad_frac:.2f}<{brake_wear_threshold})")

    # low battery
    if battery < 15:
        risk += 0.4
        reasons.append(f"battery low ({battery:.1f}%)")

    # overspeed risk
    if spd > 230:
        risk += 0.3
        reasons.append(f"overspeed ({spd:.0f} km/h)")

    # clamp
    risk = max(0.0, min(1.0, risk))
    if not reasons:
        reasons.append("nominal")
    return risk, reasons

# prepare history.csv header if missing
if not os.path.exists(HIST_PATH):
    pd.DataFrame(columns=[
        "ts","lap","speed(km/h)","Battery SOC (%)","Brake Pad (fraction)","Temperature (°C)","risk","reasons"
    ]).to_csv(HIST_PATH, index=False)

while True:
    # baseline drift
    battery_soc -= random.uniform(0.05, 0.25)
    battery_soc = max(0.0, battery_soc)

    # brake wear
    brake_pad_fraction -= random.uniform(0.0005, 0.002)

    # base temperature variation
    temperature += random.uniform(-0.6, 0.6)

    # speed profile
    if tick_in_lap in STRAIGHTS:
        speed = random.uniform(180, 250)   # long straight
    elif tick_in_lap in BRAKE_ZONES:
        speed = random.uniform(60, 120)    # heavy braking
    else:
        speed = random.uniform(100, 200)   # corners

    # braking zones heat + wear
    if tick_in_lap in BRAKE_ZONES:
        temperature += random.uniform(3.0, 8.0)
        brake_pad_fraction -= random.uniform(0.002, 0.006)

    # keep sane bounds
    brake_pad_fraction = max(0.05, min(1.0, brake_pad_fraction))
    temperature = max(15.0, min(130.0, temperature))

    risk, reasons = risk_and_reason(temperature, brake_pad_fraction, battery_soc, speed)

    row = {
        "ts": pd.Timestamp.utcnow().isoformat(),
        "lap": lap,
        "speed(km/h)": round(speed, 1),
        "Battery SOC (%)": round(battery_soc, 2),
        "Brake Pad (fraction)": round(brake_pad_fraction, 3),
        "Temperature (°C)": round(temperature, 2),
        "risk": round(risk, 3),
        "reasons": "; ".join(reasons),
    }

    # write latest
    pd.DataFrame([row]).to_csv(LATEST_PATH, index=False)

    # append to history
    pd.DataFrame([row])[["ts","lap","speed(km/h)","Battery SOC (%)","Brake Pad (fraction)","Temperature (°C)","risk","reasons"]] \
        .to_csv(HIST_PATH, mode="a", header=False, index=False)

    print(f"Lap {lap:02d} T{tick_in_lap:02d} ->", row)

    # tick / lap rollover
    tick_in_lap += 1
    if tick_in_lap >= LAP_TICKS:
        lap += 1
        tick_in_lap = 0
        # cool-down
        temperature -= random.uniform(3.0, 6.0)

    time.sleep(TICK_SEC)
