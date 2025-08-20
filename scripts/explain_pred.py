#!/usr/bin/env python3
"""
scripts/explain_pred.py

Compute SHAP explanations for the latest per-lap prediction (RUL and wear).
Writes data/explanations.json with text explanations and top features.
"""
import os, json
import numpy as np
import pandas as pd
import joblib
import shap

MODELS_DIR = "models"
LAPF = "data/lap_features.csv"
OUT = "data/explanations.json"

def load_model(name):
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(path):
        return joblib.load(path)
    return None

def main():
    models = {
        "wear": load_model("brake_wear_v1.pkl"),
        "rul" : load_model("rul_best.pkl")
    }
    if not os.path.exists(LAPF):
        print("No lap_features.csv found. Run scripts/process_laps.py")
        return
    laps = pd.read_csv(LAPF, on_bad_lines="skip")
    if laps.empty:
        print("lap_features empty")
        return
    last = laps.iloc[-1]
    # feature columns from model meta if present
    meta_path = os.path.join(MODELS_DIR, "model_meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        feature_cols = meta.get("feature_cols", ["speed_mean","speed_max","temp_mean","temp_max","duration_sec"])
    else:
        feature_cols = ["speed_mean","speed_max","temp_mean","temp_max","duration_sec"]

    X = last[feature_cols].values.reshape(1,-1)

    explanations = {"ts": pd.Timestamp.utcnow().isoformat(), "features": feature_cols, "explained": {}}

    for key in ["wear","rul"]:
        m = models.get(key)
        if m is None:
            explanations["explained"][key] = {"error":"model missing"}
            continue
        # shap for tree-based models; fallback to permutation if needed
        try:
            expl = shap.TreeExplainer(m)
            shap_vals = expl.shap_values(X)
            # shap_vals shape: for regressors typically (1, n_features)
            if isinstance(shap_vals, list):
                # classification can return list; reduce to first
                vals = np.array(shap_vals[0]).flatten()
            else:
                vals = np.array(shap_vals).flatten()
            # top contributors
            idx = np.argsort(-np.abs(vals))[:3]
            top = [{"feature": feature_cols[i], "shap": float(vals[i])} for i in idx]
            explanations["explained"][key] = {"top": top, "shap_values": [float(v) for v in vals]}
        except Exception as e:
            # fallback: return coefficients or simple feature importances
            try:
                importances = getattr(m, "feature_importances_", None)
                if importances is not None:
                    idx = np.argsort(-np.abs(importances))[:3]
                    top = [{"feature": feature_cols[i], "importance": float(importances[i])} for i in idx]
                    explanations["explained"][key] = {"top": top}
                else:
                    explanations["explained"][key] = {"error": str(e)}
            except Exception as e2:
                explanations["explained"][key] = {"error2": str(e2)}

    with open(OUT, "w") as f:
        json.dump(explanations, f, indent=2)
    print("Wrote explanations ->", OUT)

if __name__ == "__main__":
    main()
