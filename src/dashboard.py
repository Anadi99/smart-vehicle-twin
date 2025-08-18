# src/dashboard.py
import streamlit as st
import pandas as pd
import json, os

CONFIG_DIR  = "configs"
LATEST_PATH = "data/latest.csv"
HIST_PATH   = "data/history.csv"

# === Vehicle Selection ===
configs = [f for f in os.listdir(CONFIG_DIR) if f.endswith(".json")]
vehicle = st.sidebar.selectbox("🚗 Select Vehicle Config", configs)

with open(os.path.join(CONFIG_DIR, vehicle)) as f:
    cfg = json.load(f)

st.title(f"📊 {cfg.get('model','Vehicle')} - Vehicle Telemetry Dashboard")
st.caption("Live telemetry feed, auto-refresh every 2 seconds")

temp_max_C = float(cfg.get("temp_max_C", 90.0))
brake_wear_threshold = float(cfg.get("brake_wear_threshold", 0.3))

# Auto-refresh dashboard every 2 seconds
st_autorefresh = st.sidebar.button("🔄 Refresh Data")


if not (os.path.exists(LATEST_PATH) and os.path.exists(HIST_PATH)):
    st.warning("⚠️ No telemetry yet. Run the simulator first: `python src/simulator.py`")
else:
    try:
        latest = pd.read_csv(LATEST_PATH)
        hist   = pd.read_csv(HIST_PATH)
    except Exception:
        st.error("❌ Error reading telemetry files")
        st.stop()

    # keep last 150 points for charts
    hist_tail = hist.tail(150)

    # === Alerts (rule-based “AI”) ===
    st.subheader("Health & Alerts")

    # Latest row fields
    last = latest.iloc[-1]
    temp_c = float(last["Temperature (°C)"])
    pad_f  = float(last["Brake Pad (fraction)"])
    risk   = float(last.get("risk", 0.0))
    reasons = str(last.get("reasons", "nominal"))
    lap     = int(last.get("lap", 1))

    cols = st.columns(4)
    cols[0].metric("Lap", lap)
    cols[1].metric("Risk (0–1)", f"{risk:.2f}")
    cols[2].metric("Temp (°C)", f"{temp_c:.1f}")
    cols[3].metric("Brake Pad", f"{pad_f:.2f}")

    # Risk gauge (progress bar)
    st.progress(min(max(risk,0),1))

    # Alert banners
    if temp_c > temp_max_C:
        st.error(f"🔥 Overheat risk: {temp_c:.1f}°C > {temp_max_C}°C")
    elif temp_c > temp_max_C - 10:
        st.warning(f"⚠️ Temperature nearing limit: {temp_c:.1f}°C")

    if pad_f < brake_wear_threshold:
        st.error(f"🛑 Brake pad low: {pad_f:.2f} < {brake_wear_threshold}")

    if risk >= 0.7:
        st.error(f"Overall Risk HIGH ({risk:.2f}) • Reasons: {reasons}")
    elif risk >= 0.4:
        st.warning(f"Overall Risk MEDIUM ({risk:.2f}) • Reasons: {reasons}")
    else:
        st.success(f"Overall Risk LOW ({risk:.2f}) • Reasons: {reasons}")

    # === Latest snapshot table ===
    st.subheader("Latest Telemetry")
    st.dataframe(latest)

    # === Trends ===
    st.subheader("Trends (last ~5 minutes)")
    st.line_chart(hist_tail.set_index("ts")[["Battery SOC (%)"]])
    st.line_chart(hist_tail.set_index("ts")[["Brake Pad (fraction)"]])
    st.line_chart(hist_tail.set_index("ts")[["Temperature (°C)"]])

    # === Session summary ===
    st.subheader("📈 Session Summary")
    st.write({
        "Max Temp (°C)": hist["Temperature (°C)"].max(),
        "Min Brake Pad": hist["Brake Pad (fraction)"].min(),
        "Avg Risk": hist["risk"].mean(),
        "Total Laps": hist["lap"].max()
    })
