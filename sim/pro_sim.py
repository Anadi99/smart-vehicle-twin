#!/usr/bin/env python3
import os, time, json, math, random
import argparse, getpass
from datetime import datetime
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt

# ------------------ params ------------------
DEF_DT = 0.2            # seconds per tick
TRACK_KM = 5.3
LAP_M = TRACK_KM * 1000
MAX_SPEED = 240.0       # kph
MAX_ACCEL = 4.0         # m/s^2
MAX_BRAKE = 7.0         # m/s^2
BASE_SOC_DROP = 0.0004  # per tick base
TEMP_COOL_RATE = 0.25   # degC per tick cooling
TEMP_HEAT_RATE = 1.6    # degC per tick heating when braking/high load
PAD_WEAR_RATE = 0.00015 # per tick base
TIRE_WEAR_RATE = 0.00010
FAIL_PROB = 0.002       # failure chance per tick when --failures on

# Files expected by dashboard
LATEST = "data/latest.csv"
HIST   = "data/history.csv"
LAPF   = "data/lap_features.csv"

def ensure_csv(path, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)

def mqtt_client(host, port, topic):
    c = mqtt.Client()
    c.connect(host, port, 60)
    return c, topic

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laps", type=int, default=10)
    ap.add_argument("--dt", type=float, default=DEF_DT)
    ap.add_argument("--mqtt", type=str, default="localhost:1883")
    ap.add_argument("--topic", type=str, default=f"uvtwin/telemetry/{getpass.getuser()}")
    ap.add_argument("--failures", action="store_true")
    args = ap.parse_args()

    host, port = args.mqtt.split(":")[0], int(args.mqtt.split(":")[1])

    # init csvs
    cols = ["ts","lap","speed_kph","rpm","accel_mps2","distance_m","brake_temp_c",
            "battery_soc","brake_pad_frac","tire_wear_frac","risk","reasons","lat","lon"]
    ensure_csv(LATEST, cols)
    ensure_csv(HIST, cols)
    ensure_csv(LAPF, ["lap","lap_time_sec","speed_mean","speed_max","brake_temp_max","soc_drop"])

    client, topic = mqtt_client(host, port, args.topic)

    # state
    t = 0.0
    lap = 1
    lap_start_t = 0.0
    dist = 0.0
    speed = 0.0
    accel = 0.0
    rpm = 1200.0
    temp = 60.0
    soc = 100.0
    pad = 1.0    # 1.0 = new pads, goes down
    tire = 1.0   # fraction
    lat, lon = 37.42198, -122.08400  # base point, fake GPS

    # per-lap aggregators
    lap_dist_start = 0.0
    lap_soc_start = soc
    lap_temp_max = temp
    speed_hist = []

    # track shape for GPS (oval-ish)
    def gps_step(s):
        # move along a small oval; scale by speed
        r = 0.00008
        ang = (t * s * 0.001) % (2*math.pi)
        return 37.42198 + r*math.sin(ang), -122.08400 + r*math.cos(ang)

    # driving program: throttle & brake patterns
    def control_pattern(tt):
        # throttle oscillates; adds random driver variation
        throttle = 0.55 + 0.4*math.sin(tt*0.15) + random.uniform(-0.05, 0.05)
        throttle = max(0.0, min(1.0, throttle))
        # sometimes brake (corners)
        brake = 0.0
        if (tt % 28.0) > 23.0:
            brake = 0.6 + 0.3*random.random()
        return throttle, brake

    # runtime
    print(f"▶ Simulator → MQTT mqtt://{host}:{port}/{topic}  and CSV under data/")
    tick = 0
    while lap <= args.laps and soc > 3.0 and pad > 0.05 and tire > 0.05:
        throttle, brake = control_pattern(t)

        # dynamics
        a_plus = throttle * MAX_ACCEL
        a_minus = brake * MAX_BRAKE
        accel = (a_plus - a_minus)
        speed_mps = max(0.0, (speed*1000/3600) + accel*args.dt)
        # aero drag (nonlinear)
        drag = 0.00035 * (speed_mps**2)
        speed_mps = max(0.0, speed_mps - drag)
        speed = min(MAX_SPEED, speed_mps*3.6)

        # distance & lap
        d_step = speed_mps * args.dt
        dist += d_step
        speed_hist.append(speed)

        # engine rpm ~ proportional to speed (fake gearing)
        rpm = max(700.0, 35.0*speed + random.uniform(-150, 150))

        # temperatures
        temp = max(20.0, temp - TEMP_COOL_RATE)  # cool
        if brake > 0.05 or throttle > 0.85 or speed > 160:
            temp += TEMP_HEAT_RATE * (0.5 + brake + 0.4*throttle)

        # wear & battery
        # soc drops faster at higher speed & temp
        soc_drop = BASE_SOC_DROP * (1.0 + (speed/120.0)**2 + (temp/120.0)**1.2)
        soc = max(0.0, soc - soc_drop*100)  # convert to %
        pad = max(0.0, pad - (PAD_WEAR_RATE * (brake + 0.3*speed/200.0)))
        tire = max(0.0, tire - (TIRE_WEAR_RATE * (speed/220.0 + abs(accel)/MAX_BRAKE)))

        # random failures (optional)
        risk = 0.0
        reasons = "nominal"
        if args.failures and random.random() < FAIL_PROB:
            mode = random.choice(["brake_overheat","battery_spike","sensor_glitch"])
            if mode == "brake_overheat":
                temp += 30 + 40*random.random()
                risk = 0.9
                reasons = "random_failure: brake_overheat"
            elif mode == "battery_spike":
                soc = max(0.0, soc - (5 + 8*random.random()))
                risk = 0.7
                reasons = "random_failure: battery_spike"
            else:
                speed = max(0.0, speed + random.uniform(-30, 30))
                risk = 0.5
                reasons = "random_failure: sensor_glitch"

        # risk heuristics
        risk = max(risk, min(1.0, 0.4*(temp/120.0) + 0.5*(1.0-pad) + 0.25*(1.0-tire)))

        # GPS
        lat, lon = gps_step(speed)

        # write row
        now = datetime.utcnow().isoformat()
        row = {
            "ts": now,
            "lap": lap,
            "speed_kph": round(speed,2),
            "rpm": round(rpm,1),
            "accel_mps2": round(accel,3),
            "distance_m": round(dist,2),
            "brake_temp_c": round(temp,2),
            "battery_soc": round(soc,2),
            "brake_pad_frac": round(pad,4),
            "tire_wear_frac": round(tire,4),
            "risk": round(risk,3),
            "reasons": reasons,
            "lat": round(lat,6),
            "lon": round(lon,6),
        }
        df = pd.DataFrame([row])

        # append history & latest
        df.to_csv(HIST, mode="a", header=False, index=False)
        df.to_csv(LATEST, mode="w", header=True, index=False)

        # publish MQTT
        try:
            client.publish(topic, json.dumps(row), qos=0, retain=False)
        except Exception:
            pass

        # per-lap aggregations
        lap_temp_max = max(lap_temp_max, temp)

        # lap complete?
        if (dist - lap_dist_start) >= LAP_M:
            lap_time = t - lap_start_t
            speed_mean = float(np.mean(speed_hist)) if speed_hist else 0.0
            speed_max = float(np.max(speed_hist)) if speed_hist else 0.0
            soc_drop = max(0.0, lap_soc_start - soc)
            pd.DataFrame([{
                "lap": lap,
                "lap_time_sec": round(lap_time,2),
                "speed_mean": round(speed_mean,2),
                "speed_max": round(speed_max,2),
                "brake_temp_max": round(lap_temp_max,2),
                "soc_drop": round(soc_drop,2),
            }]).to_csv(LAPF, mode="a", header=not os.path.exists(LAPF) or os.path.getsize(LAPF)==0, index=False)

            # reset lap
            lap += 1
            lap_start_t = t
            lap_dist_start = dist
            lap_soc_start = soc
            lap_temp_max = temp
            speed_hist = []

        # next tick
        t += args.dt
        tick += 1
        time.sleep(max(0.0, args.dt - 0.02))
    print("■ Simulation finished.")
if __name__ == "__main__":
    main()
