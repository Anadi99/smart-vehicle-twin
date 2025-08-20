import os, json, time, math, random, csv
from datetime import datetime

import pandas as pd

# Optional MQTT publish (safe to run without a broker)
try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None

# ---------------- Config ----------------
CONFIG_PATH = "configs/bmw_i4.json"
DATA_DIR = "data"
LATEST = os.path.join(DATA_DIR, "latest.csv")
HIST   = os.path.join(DATA_DIR, "history.csv")

os.makedirs(DATA_DIR, exist_ok=True)

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

MODEL     = cfg.get("model", "Vehicle")
AMBIENT   = float(cfg.get("ambient_C", 30))
TEMP_MAX  = float(cfg.get("temp_max_C", 95))
PAD_MIN   = float(cfg.get("brake_wear_threshold", 0.20))
SOC_MIN   = float(cfg.get("soc_min_pct", 15))
LAP_LEN   = float(cfg.get("lap_length_m", 5100))
TICK_SEC  = int(cfg.get("tick_sec", 1))

MQTT_CFG      = cfg.get("mqtt", {"enabled": False})
MQTT_ENABLED  = bool(MQTT_CFG.get("enabled", False)) and mqtt is not None

print(f"▶ Starting simulator for {MODEL}")
print(f"   ambient={AMBIENT}C, temp_max={TEMP_MAX}C, pad_min={PAD_MIN}, soc_min={SOC_MIN}%")
print(f"   lap_length={LAP_LEN}m, tick={TICK_SEC}s, mqtt_enabled={MQTT_ENABLED}")

# ---------------- Track setup ----------------
# Track center near Nürburgring (approx)
LAT0, LON0 = 50.3357, 6.9470
R_EARTH    = 6371000.0  # meters
heading_deg = 0.0
radius_m   = 400.0      # fictitious oval radius for GPS drawing

# ---------------- Vehicle states ----------------
lap = 1
tick = 0
speed_kph = 0.0
prev_speed_kph = 0.0
accel_mps2 = 0.0
battery_soc = 98.0
brake_pad   = 1.00   # 1.0 new, goes down toward 0
brake_temp  = AMBIENT + 10
tire_wear   = 0.0
lap_distance = 0.0
session_distance = 0.0

# Braking zones (fractions of lap distance)
BRAKES = [(0.18, 0.24), (0.52, 0.58), (0.82, 0.86)]

# ---------------- MQTT setup ----------------
mqtt_client = None
if MQTT_ENABLED:
    mqtt_client = mqtt.Client()
    try:
        mqtt_client.connect(MQTT_CFG["host"], int(MQTT_CFG.get("port", 1883)), keepalive=30)
        mqtt_client.loop_start()
        print("   MQTT: connected")
    except Exception as e:
        print(f"   MQTT: connection failed ({e}), continuing without MQTT")
        mqtt_client = None

# ---------------- CSV headers ----------------
if not os.path.exists(HIST):
    with open(HIST, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ts","lap","tick","speed_kph","accel_mps2","distance_m","lat","lon",
            "battery_soc","brake_pad_frac","brake_temp_c","tire_wear_frac","risk","reasons"
        ])

if not os.path.exists(LATEST):
    pd.DataFrame(columns=[
        "ts","lap","tick","speed_kph","accel_mps2","distance_m","lat","lon",
        "battery_soc","brake_pad_frac","brake_temp_c","tire_wear_frac","risk","reasons"
    ]).to_csv(LATEST, index=False)

# ---------------- Helper functions ----------------
def in_brake_zone(frac: float) -> bool:
    for a, b in BRAKES:
        if a <= frac <= b:
            return True
    return False

def gps_from_angle(deg: float):
    rad = math.radians(deg)
    dx = radius_m * math.cos(rad)
    dy = radius_m * math.sin(rad)
    dlat = (dy / R_EARTH) * (180.0 / math.pi)
    dlon = (dx / (R_EARTH * math.cos(math.radians(LAT0)))) * (180.0 / math.pi)
    return LAT0 + dlat, LON0 + dlon

def risk_and_reasons():
    reasons = []
    risk = 0.0
    if brake_temp > TEMP_MAX:
        risk += min((brake_temp - TEMP_MAX) / 40.0, 1.0) * 0.6
        reasons.append(f"high brake temp {brake_temp:.1f}C")
    if brake_pad < PAD_MIN:
        risk += min((PAD_MIN - brake_pad) / 0.4, 1.0) * 0.6
        reasons.append(f"low pad {brake_pad:.2f}")
    if battery_soc < SOC_MIN:
        risk += min((SOC_MIN - battery_soc) / 20.0, 1.0) * 0.5
        reasons.append(f"low SOC {battery_soc:.1f}%")
    if not reasons:
        reasons.append("nominal")
    return max(0.0, min(1.0, risk)), "; ".join(reasons)

# ---------------- Main loop ----------------
print("▶ Running... Ctrl+C to stop")
try:
    while True:
        ts = datetime.utcnow().isoformat()

        # --- Speed profile ---
        target = random.uniform(140, 240)
        if in_brake_zone((lap_distance / LAP_LEN) if LAP_LEN > 0 else 0):
            target = random.uniform(60, 120)  # braking zones
        speed_kph += (target - speed_kph) * random.uniform(0.05, 0.15)
        speed_kph = max(0.0, min(280.0, speed_kph))

        # --- Acceleration ---
        accel_mps2 = ((speed_kph - prev_speed_kph) / 3.6) / max(1, TICK_SEC)
        prev_speed_kph = speed_kph

        # --- Distances ---
        dist_m = (speed_kph / 3.6) * TICK_SEC
        lap_distance += dist_m
        session_distance += dist_m

        # --- GPS position ---
        heading_deg = (heading_deg + (dist_m / (2 * math.pi * radius_m)) * 360.0) % 360.0
        lat, lon = gps_from_angle(heading_deg)

        # --- Battery drain ---
        drain = 0.005 * TICK_SEC + 0.00002 * (speed_kph ** 2) * (TICK_SEC / 1.0)
        drain += max(0.0, accel_mps2) * 0.01
        battery_soc = max(0.0, battery_soc - drain)

        # --- Brake system ---
        if in_brake_zone((lap_distance / LAP_LEN) if LAP_LEN > 0 else 0):
            brake_temp += 0.8 * (speed_kph / 100.0) + random.uniform(0.5, 1.5)
            brake_pad -= 0.0008 * (speed_kph / 120.0)
        else:
            brake_temp += -(brake_temp - (AMBIENT + 5)) * 0.04

        brake_temp = max(AMBIENT, min(160.0, brake_temp))
        brake_pad  = max(0.05, min(1.0, brake_pad))

        # --- Tires ---
        tire_wear = min(1.0, tire_wear + 0.0002 * (speed_kph / 150.0))

        # --- Failure injection ---
        if random.random() < 1.0 / 300.0:
            brake_temp += random.uniform(15, 30)
            battery_soc -= random.uniform(1.0, 2.0)

        # --- Risk evaluation ---
        risk, reasons = risk_and_reasons()

        # --- Row ---
        row = {
            "ts": ts,
            "lap": lap,
            "tick": tick,
            "speed_kph": round(speed_kph, 1),
            "accel_mps2": round(accel_mps2, 2),
            "distance_m": round(session_distance, 1),
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "battery_soc": round(battery_soc, 2),
            "brake_pad_frac": round(brake_pad, 3),
            "brake_temp_c": round(brake_temp, 1),
            "tire_wear_frac": round(tire_wear, 3),
            "risk": round(risk, 3),
            "reasons": reasons
        }

        # --- latest.csv ---
        pd.DataFrame([row]).to_csv(LATEST, index=False)

        # --- history.csv ---
        with open(HIST, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                row["ts"], row["lap"], row["tick"], row["speed_kph"], row["accel_mps2"],
                row["distance_m"], row["lat"], row["lon"], row["battery_soc"],
                row["brake_pad_frac"], row["brake_temp_c"], row["tire_wear_frac"],
                row["risk"], row["reasons"]
            ])

        # --- MQTT publish ---
        if mqtt_client:
            try:
                payload = json.dumps({
                    "ts": row["ts"],
                    "veh": {"brand": "BMW", "model": MODEL},
                    "lap": row["lap"],
                    "gps": {"lat": row["lat"], "lon": row["lon"]},
                    "dyn": {"speed_kph": row["speed_kph"], "accel_mps2": row["accel_mps2"]},
                    "temps": {"brake_c": row["brake_temp_c"]},
                    "battery": {"soc_pct": row["battery_soc"]},
                    "wear": {"brake_pad_frac": row["brake_pad_frac"], "tire_wear_frac": row["tire_wear_frac"]},
                    "edge_ai": {"risk": row["risk"], "explain": row["reasons"].split("; ")}
                })
                mqtt_client.publish(MQTT_CFG["topic"], payload, qos=0, retain=False)
            except Exception:
                pass

        # --- Lap rollover ---
        tick += 1
        if lap_distance >= LAP_LEN:
            lap += 1
            lap_distance = 0.0
            brake_temp = max(AMBIENT + 5, brake_temp - random.uniform(5, 10))

        # --- Console print ---
        print(f"Lap {lap:02d} T{tick:03d} v={row['speed_kph']:5.1f} km/h "
              f"T={row['brake_temp_c']:5.1f}C SOC={row['battery_soc']:5.1f}% "
              f"Pad={row['brake_pad_frac']:.3f} Risk={row['risk']:.2f} :: {row['reasons']}")
        time.sleep(TICK_SEC)

except KeyboardInterrupt:
    print("\nStopping simulator.")
finally:
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
