import numpy as np
from skl2onnx import to_onnx
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing => StandardScaler

pipe = Pipeline(
    [("scaler", StandardScaler()), ("clf", RandomForestClassifier())]
)
pipe.fit(np.random.randn(20, 4).astype(np.float32), np.random.randint(0, 3, 20))
onnx_model = to_onnx(pipe, np.zeros((1, 4), dtype=np.float32))
with open("sklearn_pipeline.onnx", "wb") as f:
  f.write(onnx_model.SerializeToString())
