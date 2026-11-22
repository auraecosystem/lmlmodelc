import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import to_onnx
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Sample CSV-style DataFrame
df = pd.DataFrame({
    "age":,
    "income": [40000.0, 90000.0, 60000.0, 110000.0],
    "city": ["Lagos", "Abuja", "Lagos", "Port Harcourt"],
    "approved":,
})
X = df[["age", "income", "city"]]
y = df["approved"]

# 2. Build full preprocessing + neural network pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["age", "income"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["city"]),
])
clf = Pipeline([
    ("prep", preprocessor),
    ("mlp", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300)),
])
clf.fit(X, y)

# 3. Export BOTH preprocessing & model into one ONNX file
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType, StringTensorType

initial_types = [
    ("age", Int64TensorType([None, 1])),
    ("income", FloatTensorType([None, 1])),
    ("city", StringTensorType([None, 1])),
]
onnx_model = to_onnx(clf, initial_types=initial_types)
with open("tabular_pipeline.onnx", "wb") as f:
  f.write(onnx_model.SerializeToString())

# 4. Run inference directly on raw CSV columns
session = ort.InferenceSession("tabular_pipeline.onnx")
inputs = {
    "age": np.array([[32]], dtype=np.int64),
    "income": np.array([[75000.0]], dtype=np.float32),
    "city": np.array([["Lagos"]], dtype=object),
}
pred_label, pred_probs = session.run(None, inputs)
print("Prediction:", pred_label[0], "| Probabilities:", pred_probs[0])
