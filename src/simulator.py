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

print(f"Starting telemetry simulator for {model} with lap logic...")

# --- simple lap model (each lap ~ 90 sec at 2s ticks = 45 ticks) ---
LAP_TICKS = 45
TICK_SEC  = 2

# braking zones inside a lap (tick indices where we heat things more)
BRAKE_ZONES = {8, 9, 10, 22, 23, 24, 36, 37, 38}

lap = 1
tick_in_lap = 0

# initialize states
battery_soc = random.uniform(70, 100)
brake_pad_fraction = random.uniform(0.6, 1.0)   # 1.0 = new pads
temperature = random.uniform(30, 45)            # "general" temp proxy

def risk_and_reason(temp_c, pad_frac):
    reasons = []
    risk = 0.0
    # temp component
    if temp_c > temp_max_C:
        over = min((temp_c - temp_max_C) / 30.0, 1.0)
        risk += 0.6 * over
        reasons.append(f"temperature high ({temp_c:.1f}°C>{temp_max_C}°C)")
    elif temp_c > (temp_max_C - 10):
        risk += 0.3
        reasons.append(f"temperature nearing limit ({temp_c:.1f}°C)")
    # brake wear component
    if pad_frac < brake_wear_threshold:
        under = min((brake_wear_threshold - pad_frac) / 0.3, 1.0)
        risk += 0.5 * under
        reasons.append(f"brake pad low ({pad_frac:.2f}<{brake_wear_threshold})")
    # clamp
    risk = max(0.0, min(1.0, risk))
    if not reasons:
        reasons.append("nominal")
    return risk, reasons

# prepare history.csv header if missing
if not os.path.exists(HIST_PATH):
    pd.DataFrame(columns=[
        "ts","lap","Battery SOC (%)","Brake Pad (fraction)","Temperature (°C)","risk","reasons"
    ]).to_csv(HIST_PATH, index=False)

while True:
    # baseline drift
    battery_soc -= random.uniform(0.05, 0.25)         # SOC drains
    battery_soc = max(0.0, battery_soc)

    # pad wear light each tick
    brake_pad_fraction -= random.uniform(0.0005, 0.002)

    # temp relaxes slightly
    temperature += random.uniform(-0.6, 0.6)

    # braking zone boost
    if tick_in_lap in BRAKE_ZONES:
        temperature += random.uniform(3.0, 8.0)
        brake_pad_fraction -= random.uniform(0.002, 0.006)

    # keep sane bounds
    brake_pad_fraction = max(0.05, min(1.0, brake_pad_fraction))
    temperature = max(15.0, min(120.0, temperature))

    risk, reasons = risk_and_reason(temperature, brake_pad_fraction)

    row = {
        "Battery SOC (%)": round(battery_soc, 2),
        "Brake Pad (fraction)": round(brake_pad_fraction, 3),
        "Temperature (°C)": round(temperature, 2),
        "risk": round(risk, 3),
        "reasons": "; ".join(reasons),
        "lap": lap,
        "ts": pd.Timestamp.utcnow().isoformat()
    }

    # write latest
    pd.DataFrame([{
        "Battery SOC (%)": row["Battery SOC (%)"],
        "Brake Pad (fraction)": row["Brake Pad (fraction)"],
        "Temperature (°C)": row["Temperature (°C)"],
        "risk": row["risk"],
        "reasons": row["reasons"],
        "lap": row["lap"],
        "ts": row["ts"]
    }]).to_csv(LATEST_PATH, index=False)

    # append to history
    pd.DataFrame([row])[["ts","lap","Battery SOC (%)","Brake Pad (fraction)","Temperature (°C)","risk","reasons"]] \
        .to_csv(HIST_PATH, mode="a", header=False, index=False)

    print(f"Lap {lap:02d} T{tick_in_lap:02d} ->", row)

    # tick / lap rollover
    tick_in_lap += 1
    if tick_in_lap >= LAP_TICKS:
        lap += 1
        tick_in_lap = 0
        # brief cool-down between laps
        temperature -= random.uniform(3.0, 6.0)

    time.sleep(TICK_SEC)
