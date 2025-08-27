import pandas as pd
import streamlit as st

st.title("🏎️ Smart Vehicle Twin – Test Dashboard")

df = pd.read_csv("data/history.csv", on_bad_lines="skip")

st.write("### Sample Data", df.head())

st.line_chart(df, x="lap", y="speed_kph")
st.line_chart(df, x="lap", y="battery_soc")
