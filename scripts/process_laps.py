#!/usr/bin/env python3
"""
scripts/process_laps.py

Produces data/lap_features.csv from either:
 - data/laps.csv   (already aggregated per-lap)
 - data/history.csv (tick-level telemetry from simulator)

Output columns include:
 lap, ticks, duration_sec, speed_mean, speed_max,
 temp_mean, temp_max, soc_start, soc_end, soc_drop,
 pad_start, pad_end, pad_wear
"""
import os, sys
import pandas as pd

RAW_LAPS = "data/laps.csv"
HIST = "data/history.csv"
OUT = "data/lap_features.csv"
TICK_SEC = 2  # seconds per tick in tick-level simulator (adjust if needed)

def safe_find_column(df, keywords):
    """Return first column name in df that contains any keyword (case-insensitive)"""
    cols = df.columns.tolist()
    low = [c.lower() for c in cols]
    for kw in keywords:
        for i, c in enumerate(low):
            if kw.lower() in c:
                return cols[i]
    return None

def process_laps_csv():
    df = pd.read_csv(RAW_LAPS)
    # if the file already contains pad_wear / soc_drop and speed_mean, consider it ready
    if set(['pad_wear', 'soc_drop']).issubset(set(df.columns.str.lower())):
        # normalize column names (lower -> expected)
        # try to rename lap_id -> lap if present
        if 'lap_id' in df.columns:
            df = df.rename(columns={'lap_id': 'lap'})
        df_out = df.copy()
        want = [
            'lap', 'ticks', 'duration_sec', 'speed_mean', 'speed_max',
            'temp_mean', 'temp_max', 'soc_start', 'soc_end', 'soc_drop',
            'pad_start', 'pad_end', 'pad_wear'
        ]
        for col in want:
            if col not in df_out.columns:
                pass
        df_out.to_csv(OUT, index=False)
        print(f"✅ Copied aggregated laps -> {OUT} (from {RAW_LAPS})")
        return

def process_history_csv():
    df = pd.read_csv(HIST)
    if df.empty:
        print("❌ history.csv is empty. Run simulator longer.")
        sys.exit(1)

    # find likely column names
    lap_col   = safe_find_column(df, ['lap'])
    speed_col = safe_find_column(df, ['speed'])
    batt_col  = safe_find_column(df, ['battery','soc'])
    pad_col   = safe_find_column(df, ['brake pad','brake_pad','pad'])
    temp_col  = safe_find_column(df, ['temp','temperature','brake_temp'])

    if lap_col is None:
        print("❌ Could not find a 'lap' column in history.csv. Columns found:", df.columns.tolist())
        sys.exit(1)

    missing = []
    for c in [speed_col, batt_col, pad_col, temp_col]:
        if c is None:
            missing.append(c)
    if missing:
        print("⚠️ Warning: could not detect all telemetry columns. Detected columns:", {
            'lap': lap_col, 'speed': speed_col, 'battery': batt_col, 'pad': pad_col, 'temp': temp_col
        })

    # coerce lap to int
    df = df.dropna(subset=[lap_col])
    df[lap_col] = df[lap_col].astype(int)

    laps = []
    for lap_id, g in df.groupby(lap_col):
        if len(g) < 3:
            continue
        ticks = len(g)
        duration_sec = ticks * TICK_SEC

        def col_stats(col):
            if col is None or col not in g.columns:
                return (float('nan'), float('nan'))
            return (g[col].mean(), g[col].max())

        speed_mean, speed_max = col_stats(speed_col)
        temp_mean, temp_max   = col_stats(temp_col)

        def start_end(col):
            if col is None or col not in g.columns:
                return (float('nan'), float('nan'))
            s = float(g[col].iloc[0])
            e = float(g[col].iloc[-1])
            return (s, e)

        soc_start, soc_end = start_end(batt_col)
        pad_start, pad_end = start_end(pad_col)

        soc_drop = soc_start - soc_end if not pd.isna(soc_start) and not pd.isna(soc_end) else float('nan')
        pad_wear = pad_start - pad_end if not pd.isna(pad_start) and not pd.isna(pad_end) else float('nan')

        laps.append({
            "lap": int(lap_id),
            "ticks": int(ticks),
            "duration_sec": float(duration_sec),
            "speed_mean": float(speed_mean) if not pd.isna(speed_mean) else None,
            "speed_max":  float(speed_max) if not pd.isna(speed_max) else None,
            "temp_mean":  float(temp_mean) if not pd.isna(temp_mean) else None,
            "temp_max":   float(temp_max) if not pd.isna(temp_max) else None,
            "soc_start":  soc_start,
            "soc_end":    soc_end,
            "soc_drop":   soc_drop,
            "pad_start":  pad_start,
            "pad_end":    pad_end,
            "pad_wear":   pad_wear
        })

    if not laps:
        print("❌ No complete laps found in history.csv (after grouping).")
        sys.exit(1)

    out = pd.DataFrame(laps).sort_values("lap")
    out.to_csv(OUT, index=False)
    print(f"✅ Wrote per-lap features -> {OUT}")
    print(out.tail(5))

def main():
    if os.path.exists(RAW_LAPS):
        print("Found data/laps.csv — using that aggregated file.")
        process_laps_csv()
    elif os.path.exists(HIST):
        print("Found data/history.csv — aggregating tick-level telemetry into per-lap features.")
        process_history_csv()
    else:
        print("❌ No input data found. Please run simulator to generate data/laps.csv or data/history.csv.")
        sys.exit(1)

if __name__ == '__main__':
    main()
