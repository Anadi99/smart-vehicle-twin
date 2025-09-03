import pandas as pd
import streamlit as st

st.set_page_config(page_title="Smart Vehicle Twin", layout="wide")

# Load data
df = pd.read_csv("data/history.csv", on_bad_lines="skip")

st.title("🏎️ Smart Vehicle Twin – Dashboard")

# Sidebar filters
st.sidebar.header("Filters")

if df["lap"].nunique() > 1:
    lap_range = st.sidebar.slider(
        "Select Lap Range",
        int(df["lap"].min()),
        int(df["lap"].max()),
        (int(df["lap"].min()), int(df["lap"].max()))
    )
    df_filtered = df[(df["lap"] >= lap_range[0]) & (df["lap"] <= lap_range[1])]
else:
    st.sidebar.write("⚠️ Only one lap available in data.")
    df_filtered = df.copy()

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("🔋 Battery SOC (%)", f"{df_filtered['battery_soc'].iloc[-1]:.1f}")
col2.metric("🛑 Brake Pad Health (%)", f"{df_filtered['brake_pad_frac'].iloc[-1]*100:.1f}")
col3.metric("⚠️ Risk", f"{df_filtered['risk'].iloc[-1]:.2f}")

# Charts
st.subheader("📈 Vehicle Telemetry")
st.line_chart(df_filtered, x="lap", y="speed_kph")

st.subheader("🔋 Battery SOC over Laps")
st.line_chart(df_filtered, x="lap", y="battery_soc")

st.subheader("🛑 Brake Pad Health over Laps")
st.line_chart(df_filtered, x="lap", y="brake_pad_frac")

st.subheader("⚠️ Risk Analysis")
st.write(df_filtered[["lap", "risk", "reasons"]].tail(10))
