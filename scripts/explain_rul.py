#!/usr/bin/env python3
import os, json, joblib, numpy as np, pandas as pd
from pathlib import Path

MODEL = "models/rul_best.pkl"
META  = "models/model_meta.json"
LAPF  = "data/lap_features.csv"
OUT   = "data/explanations.json"

def try_shap(model, Xdf):
    try:
        import shap
        # TreeExplainer works well for RandomForest/XGB; fallback to Kernel for others
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = shap.Explainer(model.predict, Xdf, algorithm="auto")
        vals = explainer(Xdf)
        # vals.values shape: (1, n_features)
        shap_vals = np.array(vals.values).reshape(-1)
        base = float(np.array(getattr(vals, "base_values", [0.0])).reshape(-1)[0])
        return shap_vals.tolist(), base, "shap"
    except Exception:
        return None, None, None

def rf_importance(model, Xdf):
    try:
        imp = getattr(model, "feature_importances_", None)
        if imp is not None:
            return imp.tolist(), 0.0, "feature_importance"
    except Exception:
        pass
    return None, None, None

if not (Path(MODEL).exists() and Path(META).exists() and Path(LAPF).exists()):
    raise SystemExit("Missing model/meta/lap_features. Train and process laps first.")

meta = json.load(open(META))
feat_cols = meta.get("feature_cols", [])
laps = pd.read_csv(LAPF)
if laps.empty:
    raise SystemExit("lap_features.csv is empty")

row = laps.iloc[[-1]]  # keep DataFrame
if not set(feat_cols).issubset(row.columns):
    raise SystemExit(f"lap_features missing {set(feat_cols)-set(row.columns)}")

Xdf = row[feat_cols].astype(float)
model = joblib.load(MODEL)

# Prefer SHAP, else RF feature_importances_
vals, base, method = try_shap(model, Xdf)
if vals is None:
    vals, base, method = rf_importance(model, Xdf)

feat_vals = row.iloc[0][feat_cols].to_dict()
contribs = []
if vals is not None:
    for f, v, s in zip(feat_cols, [feat_vals[c] for c in feat_cols], vals):
        contribs.append({"feature": f, "value": float(v), "contribution": float(s)})
    contribs.sort(key=lambda d: abs(d["contribution"]), reverse=True)

out = {
    "method": method or "none",
    "base_value": float(base or 0.0),
    "prediction_hint": "rul_laps",
    "feature_values": {k: float(v) for k, v in feat_vals.items()},
    "contribs": contribs
}
Path(OUT).write_text(json.dumps(out, indent=2))
print(f"Saved {OUT} using {out['method']}")
