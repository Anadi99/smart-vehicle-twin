#!/usr/bin/env python3
"""
scripts/predict_latest.py

Load trained models and make predictions for the latest lap/telemetry.
Outputs: data/predictions.json
"""
import os
import json
import math
import subprocess
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

# Paths
DATA_DIR = "data"
LATEST = os.path.join(DATA_DIR, "latest.csv")
HIST = os.path.join(DATA_DIR, "history.csv")
LAPF = os.path.join(DATA_DIR, "lap_features.csv")
PRED_OUT = os.path.join(DATA_DIR, "predictions.json")
MODELS_DIR = "models"
META_PATH = os.path.join(MODELS_DIR, "model_meta.json")

# thresholds (keep in sync with configs)
PAD_MIN = 0.20
SOC_MIN = 15.0

def ensure_lap_features():
    if os.path.exists(LAPF):
        return True
    # try to run process_laps.py to generate lap_features
    if os.path.exists("scripts/process_laps.py"):
        print("Generating lap_features.csv via scripts/process_laps.py ...")
        subprocess.run(["python", "scripts/process_laps.py"], check=False)
    return os.path.exists(LAPF)

def normalize_cols(df):
    """Return mapping-friendly df (lower->original) and helper to pick columns."""
    col_map = {c.lower(): c for c in df.columns}
    return col_map

def pick_col(df, candidates, default=None):
    # candidates: list of possible column names (case-insensitive)
    lowmap = {c.lower(): c for c in df.columns}
    for choice in candidates:
        if choice is None: 
            continue
        if choice in df.columns:
            return choice
        if choice.lower() in lowmap:
            return lowmap[choice.lower()]
    return default

def load_models():
    models = {}
    try:
        models['wear'] = joblib.load(os.path.join(MODELS_DIR, "brake_wear_v1.pkl"))
    except Exception:
        models['wear'] = None
    try:
        models['soc'] = joblib.load(os.path.join(MODELS_DIR, "soc_drop_v1.pkl"))
    except Exception:
        models['soc'] = None
    try:
        models['rul'] = joblib.load(os.path.join(MODELS_DIR, "rul_best.pkl"))
    except Exception:
        models['rul'] = None
    meta = {}
    if os.path.exists(META_PATH):
        try:
            meta = json.load(open(META_PATH))
        except Exception:
            meta = {}
    return models, meta

def build_feature_vector_from_lap(last_lap_row, feature_cols):
    # last_lap_row is a pandas Series; feature_cols is list of column names expected
    X = []
    for c in feature_cols:
        val = last_lap_row.get(c, None)
        if pd.isna(val) or val is None:
            # fallback to 0
            X.append(0.0)
        else:
            X.append(float(val))
    return np.array(X).reshape(1, -1)

def fallback_features_from_history(feature_cols):
    # compute simple stats from history.csv
    if not os.path.exists(HIST):
        return np.zeros((1, len(feature_cols)))
    h = pd.read_csv(HIST, on_bad_lines="skip")
    # try to map likely names
    m = { 'speed_mean': ['speed(km/h)', 'speed_kph', 'speed'],
          'speed_max': ['speed(km/h)', 'speed_kph', 'speed'],
          'temp_mean': ['temperature (°c)', 'brake_temp_c', 'battery_temp', 'temperature'],
          'temp_max' : ['temperature (°c)', 'brake_temp_c', 'battery_temp', 'temperature'],
          'duration_sec': ['duration_sec','ticks']
        }
    vals = []
    for c in feature_cols:
        found = None
        # if exact present
        if c in h.columns:
            found = c
        else:
            # try mapping
            for cand in m.get(c, []):
                if cand in h.columns:
                    found = cand
                    break
                # case-insensitive
                lowmap = {col.lower(): col for col in h.columns}
                if cand.lower() in lowmap:
                    found = lowmap[cand.lower()]
                    break
        if found is None:
            vals.append(0.0)
        else:
            if c.startswith("speed_"):
                if 'max' in c:
                    vals.append(float(h[found].max()))
                else:
                    vals.append(float(h[found].mean()))
            elif c.startswith("temp_"):
                if 'max' in c:
                    vals.append(float(h[found].max()))
                else:
                    vals.append(float(h[found].mean()))
            elif c == "duration_sec":
                # if ticks present, assume each tick=1s (or use ticks->sec conversion from file if present)
                if 'ticks' in h.columns:
                    vals.append(float(h['ticks'].mean()))
                else:
                    vals.append(60.0)
            else:
                vals.append(float(h[found].mean()))
    return np.array(vals).reshape(1, -1)

def get_latest_values():
    if not os.path.exists(LATEST):
        return {}
    L = pd.read_csv(LATEST, on_bad_lines="skip")
    if L.empty:
        return {}
    last = L.iloc[-1]
    # pick common names
    pad = pick_col(L, ["brake_pad_frac", "Brake Pad (fraction)", "brake pad (fraction)"], default=None)
    soc = pick_col(L, ["battery_soc", "Battery SOC (%)", "soc_pct", "soc"], default=None)
    temp = pick_col(L, ["brake_temp_c", "Temperature (°C)", "temperature"], default=None)
    speed = pick_col(L, ["speed_kph", "speed(km/h)", "speed"], default=None)
    out = {
        "pad_col": pad,
        "soc_col": soc,
        "temp_col": temp,
        "speed_col": speed,
        "pad": float(last[pad]) if pad and not pd.isna(last[pad]) else None,
        "soc": float(last[soc]) if soc and not pd.isna(last[soc]) else None,
        "temp": float(last[temp]) if temp and not pd.isna(last[temp]) else None,
        "speed": float(last[speed]) if speed and not pd.isna(last[speed]) else None
    }
    return out

def main():
    ensure_lap_features()
    models, meta = load_models()
    feature_cols = meta.get("feature_cols", ["speed_mean","speed_max","temp_mean","temp_max","duration_sec"])

    # get features from last lap (preferred)
    if os.path.exists(LAPF):
        laps = pd.read_csv(LAPF, on_bad_lines="skip")
        if not laps.empty:
            last_lap = laps.iloc[-1]
            # if some feature names don't exactly match, try case-insensitive mapping
            # direct use if columns exist
            if all(c in last_lap.index for c in feature_cols):
                X = build_feature_vector_from_lap(last_lap, feature_cols)
            else:
                # attempt case-insensitive mapping: lower-case map
                lower_map = {col.lower(): col for col in last_lap.index}
                mapped = {}
                for c in feature_cols:
                    if c in last_lap.index:
                        mapped[c] = last_lap[c]
                    elif c.lower() in lower_map:
                        mapped[c] = last_lap[lower_map[c.lower()]]
                    else:
                        mapped[c] = 0.0
                X = np.array([float(mapped[c]) for c in feature_cols]).reshape(1,-1)
        else:
            X = fallback_features_from_history(feature_cols)
    else:
        X = fallback_features_from_history(feature_cols)

    # predictions
    out = {
        "ts": datetime.utcnow().isoformat(),
        "predictions": {}
    }

    if models.get('wear') is not None:
        try:
            p = float(models['wear'].predict(X)[0])
            out["predictions"]["brake_wear_per_lap"] = float(np.round(p, 6))
        except Exception as e:
            out["predictions"]["brake_wear_per_lap"] = None
            out["predictions"]["error_wear"] = str(e)
    else:
        out["predictions"]["brake_wear_per_lap"] = None

    if models.get('soc') is not None:
        try:
            p = float(models['soc'].predict(X)[0])
            out["predictions"]["soc_drop_per_lap"] = float(np.round(p, 6))
        except Exception as e:
            out["predictions"]["soc_drop_per_lap"] = None
            out["predictions"]["error_soc"] = str(e)
    else:
        out["predictions"]["soc_drop_per_lap"] = None

    if models.get('rul') is not None:
        try:
            p = float(models['rul'].predict(X)[0])
            out["predictions"]["rul_laps"] = float(np.round(p,3))
        except Exception as e:
            out["predictions"]["rul_laps"] = None
            out["predictions"]["error_rul"] = str(e)
    else:
        out["predictions"]["rul_laps"] = None

    # current telemetry
    latest_vals = get_latest_values()
    curr_pad = latest_vals.get("pad")
    curr_soc = latest_vals.get("soc")

    # compute laps-until thresholds
    wear = out["predictions"].get("brake_wear_per_lap")
    soc_drop = out["predictions"].get("soc_drop_per_lap")
    rul_pred = out["predictions"].get("rul_laps")

    # laps until pad threshold
    if curr_pad is None:
        out["predictions"]["laps_until_pad"] = None
    elif wear is None or wear <= 0:
        out["predictions"]["laps_until_pad"] = None
    else:
        laps_pad = math.floor((curr_pad - PAD_MIN) / wear) if (curr_pad - PAD_MIN) > 0 else 0
        out["predictions"]["laps_until_pad"] = int(max(0, laps_pad))

    # laps until soc threshold
    if curr_soc is None:
        out["predictions"]["laps_until_soc"] = None
    elif soc_drop is None or soc_drop <= 0:
        out["predictions"]["laps_until_soc"] = None
    else:
        laps_soc = math.floor((curr_soc - SOC_MIN) / soc_drop) if (curr_soc - SOC_MIN) > 0 else 0
        out["predictions"]["laps_until_soc"] = int(max(0, laps_soc))

    # recommended action
    cand = [v for v in [
        out["predictions"].get("laps_until_pad"),
        out["predictions"].get("laps_until_soc"),
        out["predictions"].get("rul_laps")
    ] if v is not None]
    if not cand:
        recommended = "No prediction available"
    else:
        min_laps = min(cand)
        if min_laps <= 0:
            recommended = "Pit NOW"
        elif min_laps <= 2:
            recommended = "Consider Pit in 1-2 laps"
        elif min_laps <= 5:
            recommended = "Plan Pit in next 3-5 laps"
        else:
            recommended = "No immediate action"

    out["predictions"]["recommended_action"] = recommended

    # write JSON
    with open(PRED_OUT, "w") as f:
        json.dump(out, f, indent=2)

    print("Saved predictions ->", PRED_OUT)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
