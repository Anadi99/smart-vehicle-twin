#!/usr/bin/env python3
"""
dashboard/app.py

Streamlit dashboard for Smart Vehicle Twin
Shows per-lap features, predictions, telemetry, and alerts
"""

import streamlit as st
import pandas as pd
import joblib
import os

# =============================
# Config
# =============================
FEATURES_FILE = "data/lap_features.csv"
BATTERY_MODEL = "models/battery_model.pkl"
LATEST_FILE = "data/latest.csv"
HISTORY_FILE = "data/history.csv"

st.set_page_config(page_title="Smart Vehicle Twin Dashboard", layout="wide")

st.title("🚗 Smart Vehicle Twin Dashboard")

# =============================
# Section 1: Lap Features & Battery Prediction
# =============================
st.header("🏁 Per-Lap Analytics & Battery Prediction")

if os.path.exists(FEATURES_FILE) and os.path.exists(BATTERY_MODEL):
    try:
        df = pd.read_csv(FEATURES_FILE, on_bad_lines="skip")
        st.dataframe(df.tail(10))

        model = joblib.load(BATTERY_MODEL)
        preds = model.predict(df.drop(columns=["lap"], errors="ignore"))
        df["Predicted_Battery"] = preds

        st.line_chart(df[["Predicted_Battery"]])
    except Exception as e:
        st.warning(f"Error loading features/model: {e}")
else:
    st.info("No lap features or model found yet.")

# =============================
# Section 2: Live Telemetry
# =============================
st.header("📡 Live Telemetry (auto-refresh)")

try:
    if os.path.exists(LATEST_FILE):
        latest = pd.read_csv(LATEST_FILE, on_bad_lines="skip")

        # Keep history if available
        if os.path.exists(HISTORY_FILE):
            hist = pd.read_csv(HISTORY_FILE, on_bad_lines="skip").tail(200)
        else:
            hist = latest.tail(200)

        c1, c2, c3 = st.columns(3)

        if "speed(km/h)" in hist.columns:
            c1.line_chart(hist["speed(km/h)"], height=200)

        if "Temperature (°C)" in hist.columns:
            c2.line_chart(hist["Temperature (°C)"], height=200)

        if "Battery SOC (%)" in hist.columns:
            c3.line_chart(hist["Battery SOC (%)"], height=200)
    else:
        st.info("No telemetry yet. Start the simulator to see live data.")
except Exception as e:
    st.warning(f"Error reading telemetry: {e}")

# =============================
# Section 3: Alerts
# =============================
st.header("🚨 Alerts")
alerts = []

try:
    if os.path.exists(LATEST_FILE):
        latest = pd.read_csv(LATEST_FILE, on_bad_lines="skip")

        if "Temperature (°C)" in latest.columns and latest["Temperature (°C)"].iloc[-1] > 95:
            alerts.append("⚠️ Engine Overheating")

        if "Battery SOC (%)" in latest.columns and latest["Battery SOC (%)"].iloc[-1] < 15:
            alerts.append("🔋 Battery critically low")

        if "brake_pad_fraction" in latest.columns and latest["brake_pad_fraction"].iloc[-1] < 0.2:
            alerts.append("🛑 Brake pads worn out")

    if alerts:
        for a in alerts:
            st.error(a)
    else:
        st.success("✅ All systems normal")
except Exception as e:
    st.warning(f"Error checking alerts: {e}")
