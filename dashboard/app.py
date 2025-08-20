#!/usr/bin/env python3
"""
dashboard/app.py

Streamlit dashboard for Smart Vehicle Twin
Shows per-lap features, predictions, and alerts
"""

import streamlit as st
import pandas as pd
import joblib
import os

FEATURES_FILE = "data/lap_features.csv"
BATTERY_MODEL = "models/battery_model.pkl"
BRAKE_MODEL = "models/brake_model.pkl"

st.set_page_config(page_title="Smart Vehicle Twin", layout="wide")

st.title("🚗 Smart Vehicle Digital Twin Dashboard")

# Load features
if not os.path.exists(FEATURES_FILE):
    st.error("No lap_features.csv found. Please run `scripts/process_laps.py` first.")
    st.stop()

df = pd.read_csv(FEATURES_FILE)

st.subheader("📊 Per-Lap Features")
st.dataframe(df.tail(10))

# Load models
battery_model = joblib.load(BATTERY_MODEL) if os.path.exists(BATTERY_MODEL) else None
brake_model   = joblib.load(BRAKE_MODEL) if os.path.exists(BRAKE_MODEL) else None

if battery_model and brake_model:
    st.subheader("🔮 Predictions")
    features = df[["speed_mean", "temp_mean", "soc_drop", "pad_wear"]].fillna(0)

    df["pred_soc_drop"] = battery_model.predict(features)
    df["pred_pad_wear"] = brake_model.predict(features)

    st.line_chart(df[["lap", "soc_drop", "pred_soc_drop"]].set_index("lap"))
    st.line_chart(df[["lap", "pad_wear", "pred_pad_wear"]].set_index("lap"))

    # Alerts
    st.subheader("⚠️ Alerts")
    last = df.iloc[-1]
    if last["pred_soc_drop"] > 5:
        st.warning(f"High predicted battery drop ({last['pred_soc_drop']:.2f}%) on next lap.")
    if last["pred_pad_wear"] > 0.5:
        st.warning(f"Brake pad wear increasing quickly ({last['pred_pad_wear']:.2f}mm expected).")
else:
    st.info("Models not trained yet. Run `python scripts/train_models.py`.")
