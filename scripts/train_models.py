# scripts/train_models.py
import os, json, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import joblib

warnings.filterwarnings("ignore")

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

RAW_PATH = "data/lap_data.csv"
MODELS_DIR = "models"
META_PATH = os.path.join(MODELS_DIR, "model_meta.json")
os.makedirs(MODELS_DIR, exist_ok=True)

assert os.path.exists(RAW_PATH), "Run simulator first to create data/lap_data.csv"
df = pd.read_csv(RAW_PATH).copy()

# Basic sanity filter
df = df[(df["ticks"] >= 20)].reset_index(drop=True)
if len(df) < 10:
    raise SystemExit(f"Not enough laps to train (got {len(df)}). Run simulator longer.")

# --- Handle missing columns ---
if "battery_temp" not in df.columns:
    # synthetic: baseline temp + noise from speed
    df["battery_temp"] = 25 + 0.05 * df["speed"] + np.random.normal(0, 0.5, size=len(df))

if "soc_drop" not in df.columns:
    df["soc_drop"] = np.maximum(0, 0.05 * (df["ticks"] * 0.1))

if "tire_wear" not in df.columns and "pad_wear" not in df.columns:
    df["tire_wear"] = 1 - 0.001 * df["ticks"]

# --- Feature engineering ---
df["duration_sec"] = df["ticks"] * 0.1
df["speed_mean"] = df["speed"].rolling(5, min_periods=1).mean()
df["speed_max"] = df["speed"].rolling(5, min_periods=1).max()
df["temp_mean"] = df["battery_temp"].rolling(5, min_periods=1).mean()
df["temp_max"] = df["battery_temp"].rolling(5, min_periods=1).max()

# Handle missing/NaNs
df = df.fillna(method="bfill").fillna(method="ffill")

feature_cols = ["speed_mean", "speed_max", "temp_mean", "temp_max", "duration_sec"]
X = df[feature_cols].values

# --- 1) Brake wear predictor ---
if "tire_wear" in df.columns:
    y_wear = df["tire_wear"].values
elif "pad_wear" in df.columns:
    y_wear = df["pad_wear"].values
else:
    y_wear = 1 - 0.001 * df["ticks"]  # fallback synthetic

ts = max(0.2, min(0.5, 1.0 / len(df) * 5))
Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(X, y_wear, test_size=ts, random_state=42)
m_wear = RandomForestRegressor(n_estimators=300, random_state=42)
m_wear.fit(Xw_tr, yw_tr)
mae_w = mean_absolute_error(yw_te, m_wear.predict(Xw_te))
joblib.dump(m_wear, os.path.join(MODELS_DIR, "brake_wear_v1.pkl"))

# --- 2) Battery SOC drop predictor ---
y_soc = df["soc_drop"].values
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X, y_soc, test_size=ts, random_state=42)
m_soc = RandomForestRegressor(n_estimators=300, random_state=42)
m_soc.fit(Xs_tr, ys_tr)
mae_s = mean_absolute_error(ys_te, m_soc.predict(Xs_te))
joblib.dump(m_soc, os.path.join(MODELS_DIR, "soc_drop_v1.pkl"))

# --- 3) Remaining Useful Laps (RUL) predictor ---
PAD_MIN = 0.20
SOC_MIN = 15.0

df["pad_end"] = df.get("tire_wear", pd.Series(1 - 0.001 * df["ticks"]))
df["soc_end"] = df.get("soc_end", 100 - np.cumsum(df["soc_drop"].fillna(0)))

rul = []
for i in range(len(df)):
    future = df.iloc[i:][["pad_end", "soc_end", "lap"]].reset_index(drop=True)
    stop_idx = None
    for j in range(len(future)):
        if (future.loc[j, "pad_end"] <= PAD_MIN) or (future.loc[j, "soc_end"] <= SOC_MIN):
            stop_idx = j
            break
    if stop_idx is None:
        rul.append(max(0, len(df) - i - 1))
    else:
        rul.append(max(0, stop_idx))
df["rul_laps"] = rul

# Train multiple candidate models
candidates = [
    ("linear", LinearRegression()),
    ("rf", RandomForestRegressor(n_estimators=500, random_state=42))
]
if HAS_XGB:
    candidates.append(("xgb", XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=42
    )))

y_rul = df["rul_laps"].values
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X, y_rul, test_size=ts, random_state=42)

best_name, best_mae, best_model = None, 1e9, None
for name, model in candidates:
    model.fit(Xr_tr, yr_tr)
    pred = model.predict(Xr_te)
    mae = mean_absolute_error(yr_te, pred)
    if mae < best_mae:
        best_name, best_mae, best_model = name, mae, model

joblib.dump(best_model, os.path.join(MODELS_DIR, "rul_best.pkl"))

# --- Save metadata ---
meta = {
    "feature_cols": feature_cols,
    "rul_best_model": best_name,
    "mae": {
        "brake_wear": float(mae_w),
        "soc_drop": float(mae_s),
        "rul": float(best_mae)
    }
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print("✅ Saved models:")
print(f"  - models/brake_wear_v1.pkl   (MAE={mae_w:.4f})")
print(f"  - models/soc_drop_v1.pkl     (MAE={mae_s:.4f})")
print(f"  - models/rul_best.pkl        (best={best_name}, MAE={best_mae:.4f})")
print(f"ℹ Meta: {META_PATH}")
