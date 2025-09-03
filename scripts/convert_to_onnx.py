from pathlib import Path
import joblib
import xgboost as xgb
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxmltools

# Paths
model_path = Path("models/rul_best.pkl")
onnx_path = Path("models/rul_best.onnx")

# Load model
model = joblib.load(model_path)

# Convert based on type
if isinstance(model, xgb.XGBRegressor):
    print("🔄 Converting XGBoost model to ONNX...")
    initial_type = [("input", FloatTensorType([None, model.n_features_in_]))]
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
else:
    print("🔄 Converting sklearn model to ONNX...")
    initial_type = [("input", FloatTensorType([None, model.n_features_in_]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"✅ Exported ONNX model saved at: {onnx_path}")

