import io
from fastapi import FastAPI, File, UploadFile
import numpy as np
import onnxruntime as ort
from PIL import Image

app = FastAPI(title="ONNX Digit Recognition API")

# Load ONNX session once at startup (thread-safe for concurrent requests)
session = ort.InferenceSession(
    "digit_cnn.onnx", providers=["CPUExecutionProvider"]
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  # Read uploaded image bytes
  contents = await file.read()
  img = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))

  # Preprocess
  arr = (np.asarray(img, dtype=np.float32) / 255.0 - 0.1307) / 0.3081
  input_tensor = arr[np.newaxis, np.newaxis, :, :]

  # Inference
  logits = session.run(["logits"], {"image": input_tensor})[0]
  probs = np.exp(logits - np.max(logits))
  probs /= np.sum(probs)
  pred = int(np.argmax(probs))

  return {
      "filename": file.filename,
      "predicted_digit": pred,
      "confidence": round(float(probs[0, pred]), 4),
  }
