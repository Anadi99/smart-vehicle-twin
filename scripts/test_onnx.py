#!/usr/bin/env python3
import json, joblib, numpy as np, onnxruntime as ort, pandas as pd
from pathlib import Path

META="models/model_meta.json"; SK="models/rul_best.pkl"; ONNX="models/rul_best.onnx"; LAPF="data/lap_features.csv"
meta=json.load(open(META)); cols=meta["feature_cols"]
row=pd.read_csv(LAPF).iloc[-1][cols].astype(float).values.reshape(1,-1)

sk=joblib.load(SK)
sk_pred=float(sk.predict(row)[0])

sess=ort.InferenceSession(Path(ONNX).read_bytes(), providers=["CPUExecutionProvider"])
inp_name=sess.get_inputs()[0].name
onnx_pred=float(sess.run(None, {inp_name: row.astype("float32")})[0].ravel()[0])

print(f"SKLearn: {sk_pred:.4f}  |  ONNX: {onnx_pred:.4f}")
