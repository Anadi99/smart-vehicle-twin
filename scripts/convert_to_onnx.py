#!/usr/bin/env python3
import os, json, joblib, numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from pathlib import Path

MODEL="models/rul_best.pkl"
META ="models/model_meta.json"
OUT  ="models/rul_best.onnx"

assert Path(MODEL).exists() and Path(META).exists(), "Train models first."
meta=json.load(open(META))
feat_cols=meta["feature_cols"]
n=len(feat_cols)

model=joblib.load(MODEL)
onnx_model = convert_sklearn(model, initial_types=[("input", FloatTensorType([None, n]))])
Path(OUT).write_bytes(onnx_model.SerializeToString())
print("Saved", OUT)
