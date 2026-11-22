import asyncio
import io
import os
from fastapi import FastAPI, File, HTTPException, UploadFile
import numpy as np
import onnxruntime as ort
from PIL import Image

app = FastAPI(title="Production ONNX Inference API")

# 1. Tune ONNX Runtime thread pool for containerized CPUs
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "2"))
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(
    "digit_cnn.onnx", sess_options=opts, providers=["CPUExecutionProvider"]
)


def run_inference_sync(image_bytes: bytes) -> dict:
  img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28))
  arr = (np.asarray(img, dtype=np.float32) / 255.0 - 0.1307) / 0.3081
  input_tensor = arr[np.newaxis, np.newaxis, :, :]

  logits = session.run(["logits"], {"image": input_tensor})
  probs = np.exp(logits - np.max(logits))
  probs /= np.sum(probs)
  pred = int(np.argmax(probs))
  return {"predicted_digit": pred, "confidence": round(float(probs[0, pred]), 4)}


@app.get("/health")
def health_check():
  return {"status": "healthy", "providers": session.get_providers()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  if not file.content_type or not file.content_type.startswith("image/"):
    raise HTTPException(status_code=400, detail="File must be an image.")

  contents = await file.read()
  # Offload CPU-bound inference to a worker thread so the event loop stays responsive
  result = await asyncio.to_thread(run_inference_sync, contents)
  return {"filename": file.filename, **result}
