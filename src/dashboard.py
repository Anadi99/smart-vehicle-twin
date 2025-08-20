# src/dashboard.py (updated to show predictions.json)
import os, json, time
import pandas as pd
import streamlit as st
import joblib

CONFIG_PATH = "configs/bmw_i4.json"
LATEST = "data/latest.csv"
HIST   = "data/history.csv"
LAPF   = "data/lap_features.csv"
PRED_F = "data/predictions.json"
MODELS_DIR = "models"

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

st.set_page_config(page_title="Smart Vehicle Twin", layout="wide")
st.title("🚗 Smart Vehicle Digital Twin Dashboard")
st.caption("📡 Live telemetry • per-lap analytics • Remaining Useful Life prediction")

st.write("Auto-refreshing every 2 seconds…")
time.sleep(0.5)

if not (os.path.exists(LATEST) and os.path.exists(HIST)):
    st.warning("No telemetry yet. Run the simulator:  `python sim/simulator.py`")
    st.stop()

# load latest + history robustly
latest = pd.read_csv(LATEST, on_bad_lines="skip")
hist = pd.read_csv(HIST, on_bad_lines="skip")

# normalize names used earlier
hist = hist.rename(columns={
    "speed(km/h)": "speed_kph",
    "Battery SOC (%)": "battery_soc",
    "Brake Pad (fraction)": "brake_pad_frac",
    "Temperature (°C)": "brake_temp_c"
})

# tail window
hist_tail = hist.tail(300).copy()
for c in ["speed_kph","brake_temp_c","battery_soc","brake_pad_frac","tire_wear_frac","lap","risk"]:
    if c in hist_tail.columns:
        hist_tail[c] = pd.to_numeric(hist_tail[c], errors="coerce")

last = latest.iloc[-1]

# KPIs
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Lap", int(last.get("lap", 1)))
k2.metric("Speed (km/h)", f"{float(last.get('speed_kph', last.get('speed(km/h)', 0))):.1f}")
k3.metric("Brake Temp (°C)", f"{float(last.get('brake_temp_c', last.get('Temperature (°C)', 0))):.1f}")
k4.metric("Battery SOC (%)", f"{float(last.get('battery_soc', last.get('Battery SOC (%)', 0))):.1f}")
k5.metric("Brake Pad (frac)", f"{float(last.get('brake_pad_frac', last.get('Brake Pad (fraction)', 0))):.3f}")

# Alerts (same as before)
TEMP_MAX = float(cfg.get("temp_max_C", 95))
PAD_MIN  = float(cfg.get("brake_wear_threshold", 0.2))
SOC_MIN  = float(cfg.get("soc_min_pct", 15))

alerts = []
temp_c = float(last.get("brake_temp_c", last.get("Temperature (°C)", 0)))
pad_f  = float(last.get("brake_pad_frac", last.get("Brake Pad (fraction)", 1.0)))
soc    = float(last.get("battery_soc", last.get("Battery SOC (%)", 100)))
risk   = float(last.get("risk", 0.0))
reasons = str(last.get("reasons", "nominal"))

if temp_c > TEMP_MAX:
    alerts.append(("critical", f"🔥 Brake overheat: {temp_c:.1f}°C > {TEMP_MAX}°C"))
elif temp_c > TEMP_MAX - 10:
    alerts.append(("warning", f"⚠️ Brake temp nearing limit: {temp_c:.1f}°C"))
if pad_f < PAD_MIN:
    alerts.append(("critical", f"🛑 Brake pad low: {pad_f:.3f} < {PAD_MIN}"))
elif pad_f < PAD_MIN + 0.1:
    alerts.append(("warning", f"🟡 Brake pad getting low: {pad_f:.3f}"))
if soc < SOC_MIN:
    alerts.append(("critical", f"🔋 SOC low: {soc:.1f}% < {SOC_MIN}%"))
elif soc < SOC_MIN + 5:
    alerts.append(("warning", f"🟡 SOC nearing low: {soc:.1f}%"))
if risk >= 0.7: alerts.append(("critical", f"Overall Risk HIGH ({risk:.2f}) • {reasons}"))
elif risk >= 0.4: alerts.append(("warning", f"Overall Risk MEDIUM ({risk:.2f}) • {reasons}"))
else: alerts.append(("ok", f"Overall Risk LOW ({risk:.2f}) • {reasons}"))

st.subheader("Health & Alerts")
for level,msg in alerts:
    if level == "critical": st.error(msg)
    elif level == "warning": st.warning(msg)
    else: st.success(msg)

# Charts
st.subheader("📈 Live Trends (last ~5 min)")
c1,c2,c3 = st.columns(3)
with c1:
    if "speed_kph" in hist_tail.columns:
        st.line_chart(hist_tail.set_index("ts")[["speed_kph"]])
with c2:
    if "brake_temp_c" in hist_tail.columns:
        st.line_chart(hist_tail.set_index("ts")[["brake_temp_c"]])
with c3:
    if "battery_soc" in hist_tail.columns:
        st.line_chart(hist_tail.set_index("ts")[["battery_soc"]])

c4,c5 = st.columns(2)
with c4:
    if "brake_pad_frac" in hist_tail.columns:
        st.line_chart(hist_tail.set_index("ts")[["brake_pad_frac"]])
with c5:
    if "tire_wear_frac" in hist_tail.columns:
        st.line_chart(hist_tail.set_index("ts")[["tire_wear_frac"]])

# Per-lap features + RUL (existing)
st.subheader("📊 Per-Lap Features")
if os.path.exists(LAPF):
    laps = pd.read_csv(LAPF, on_bad_lines="skip")
    if "lap_time_sec" not in laps.columns and "duration_sec" in laps.columns:
        laps["lap_time_sec"] = laps["duration_sec"]
    if "brake_temp_max" not in laps.columns and "temp_max" in laps.columns:
        laps["brake_temp_max"] = laps["temp_max"]
    st.dataframe(laps.tail(10), use_container_width=True)

    rul_model_path = os.path.join(MODELS_DIR, "rul_best.pkl")
    meta_path = os.path.join(MODELS_DIR, "model_meta.json")
    if os.path.exists(rul_model_path) and os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        feature_cols = meta.get("feature_cols", ["speed_mean","speed_max","temp_mean","temp_max","duration_sec"])
        if not laps.empty and all(c in laps.columns for c in feature_cols):
            row = laps.iloc[-1]
            X = row[feature_cols].values.reshape(1,-1)
        else:
            X = [[
                hist_tail["speed_kph"].mean() if "speed_kph" in hist_tail.columns else 0.0,
                hist_tail["speed_kph"].max() if "speed_kph" in hist_tail.columns else 0.0,
                hist_tail["brake_temp_c"].mean() if "brake_temp_c" in hist_tail.columns else 0.0,
                hist_tail["brake_temp_c"].max() if "brake_temp_c" in hist_tail.columns else 0.0,
                60.0
            ]]
        try:
            model = joblib.load(rul_model_path)
            pred_rul = float(model.predict(X)[0])
            mae_rul = float(meta.get("mae",{}).get("rul", 2.0))
            lo = max(0.0, pred_rul - max(1.0, 1.5*mae_rul))
            hi = pred_rul + max(1.0, 1.5*mae_rul)

            s_rul1, s_rul2, s_rul3 = st.columns(3)
            s_rul1.metric("Predicted RUL (laps)", f"{pred_rul:.1f}")
            s_rul2.metric("Confidence Low", f"{lo:.1f}")
            s_rul3.metric("Confidence High", f"{hi:.1f}")
        except Exception:
            st.info("RUL model exists but failed to predict.")
    else:
        st.info("Models not trained yet. Run: `python scripts/train_models.py`")
else:
    st.info("Run: `python scripts/process_laps.py` to create per-lap features.")

# ===== Show latest saved predictions.json (if present) =====
st.subheader("🤖 Live ML Predictions (from scripts/predict_latest.py)")
if os.path.exists(PRED_F):
    try:
        p = json.load(open(PRED_F))
        preds = p.get("predictions", {})
        c1,c2,c3 = st.columns(3)
        c1.metric("Pred brake wear / lap", f"{preds.get('brake_wear_per_lap', 'N/A')}")
        c2.metric("Pred SOC drop / lap", f"{preds.get('soc_drop_per_lap', 'N/A')}")
        c3.metric("Pred RUL (laps)", f"{preds.get('rul_laps', 'N/A')}")
        c4,c5 = st.columns(2)
        c4.metric("Laps until pad", f"{preds.get('laps_until_pad', 'N/A')}")
        c5.metric("Laps until SOC", f"{preds.get('laps_until_soc', 'N/A')}")
        st.info(f"Recommended action: {preds.get('recommended_action', 'N/A')}")
    except Exception as e:
        st.warning(f"Could not read predictions.json: {e}")
else:
    st.info("No predictions.json found. Run: `python scripts/predict_latest.py` to generate.")

st.divider()
st.caption("Tip: run the simulator continuously in another terminal and re-run scripts/predict_latest.py to refresh predictions.")

# ---------- Optional: Run inference from dashboard (button) ----------
import subprocess
if st.button("Run ML inference now", key="inference_btn1"):
    st.info("Running inference (scripts/predict_latest.py)...")
    try:
        subprocess.run(["python", "scripts/predict_latest.py"], check=True)
        st.success("Inference completed — predictions.json updated.")
    except Exception as e:
        st.error(f"Inference failed: {e}")

# ---------- Optional: Run inference from dashboard (button) ----------
import subprocess
if st.button("Run ML inference now", key="inference_btn2"):
    st.info("Running inference (scripts/predict_latest.py)...")
    try:
        subprocess.run(["python", "scripts/predict_latest.py"], check=True)
        st.success("Inference completed — predictions.json updated.")
    except Exception as e:
        st.error(f"Inference failed: {e}")

# --- Auto-run inference helper (safe) ---
import subprocess, threading, time

AUTO_FLAG = "data/_auto_infer.flag"

def start_auto_infer(interval_sec=30):
    """Background thread that runs scripts/predict_latest.py every interval_sec while flag exists."""
    def runner():
        while os.path.exists(AUTO_FLAG):
            try:
                subprocess.run(["python", "scripts/predict_latest.py"], check=False)
            except Exception:
                pass
            # avoid tight loop
            time.sleep(interval_sec)
    t = threading.Thread(target=runner, daemon=True)
    t.start()

# UI controls
st.subheader("ML inference controls")
colA, colB = st.columns([1,1])
if colA.button("Run ML inference now", key="inference_btn3"):
    st.info("Running inference...")
    try:
        subprocess.run(["python", "scripts/predict_latest.py"], check=True)
        st.success("Inference completed — predictions.json updated.")
    except Exception as e:
        st.error(f"Inference failed: {e}")

if colB.button("Start auto-inference (30s)"):
    # create the flag file and start the background thread
    open(AUTO_FLAG, "w").close()
    start_auto_infer(interval_sec=30)
    st.success("Auto-inference started (every 30s).")

if st.button("Stop auto-inference"):
    try:
        if os.path.exists(AUTO_FLAG):
            os.remove(AUTO_FLAG)
        st.info("Auto-inference stopped.")
    except Exception as e:
        st.error(f"Error stopping auto-inference: {e}")

