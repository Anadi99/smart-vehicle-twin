# src/efficiency.py
import pandas as pd

def _normalize(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = float(s.min()), float(s.max())
    if pd.isna(lo) or pd.isna(hi) or hi - lo < 1e-9:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - lo) / (hi - lo)

def compute_efficiency(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a simple driver efficiency score per lap:
      - lower lap time is better
      - lower brake temp is better (less fade)
      - lower SOC drop is better (energy efficiency)

    The final score is in [0, 100].
    """
    if laps is None or laps.empty:
        return laps

    df = laps.copy()

    # Make sure we have the expected columns (derive if needed)
    if "lap_time_sec" not in df.columns and "duration_sec" in df.columns:
        df["lap_time_sec"] = df["duration_sec"]

    if "brake_temp_max" not in df.columns:
        # try to use temp_max if present
        if "temp_max" in df.columns:
            df["brake_temp_max"] = df["temp_max"]
        else:
            # fallback to a constant if missing
            df["brake_temp_max"] = 0.0

    if "soc_drop" not in df.columns:
        # derive a rough soc_drop from soc_start/end if available
        if {"soc_start", "soc_end"}.issubset(df.columns):
            df["soc_drop"] = df["soc_start"] - df["soc_end"]
        else:
            df["soc_drop"] = 0.0

    # Normalize (0..1)
    t_norm = _normalize(df["lap_time_sec"])
    b_norm = _normalize(df["brake_temp_max"])
    s_norm = _normalize(df["soc_drop"])

    # Weighted score (higher is better)
    df["efficiency_score"] = (
        (1 - t_norm) * 0.4
        + (1 - b_norm) * 0.3
        + (1 - s_norm) * 0.3
    ) * 100.0

    return df
