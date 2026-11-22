import numpy as np
import onnxruntime as ort
from PIL import Image

session = ort.InferenceSession(
    "digit_cnn.onnx", providers=["CPUExecutionProvider"]
)


def predict_image_file(image_path: str):
  # 1. Load, convert to grayscale ("L"), and resize to 28x28
  img = Image.open(image_path).convert("L").resize((28, 28))

  # 2. Convert to float32, scale, and normalize
  arr = np.asarray(img, dtype=np.float32) / 255.0
  arr = (arr - 0.1307) / 0.3081
  input_tensor = arr[np.newaxis, np.newaxis, :, :]  # (1, 1, 28, 28)

  # 3. Predict & compute confidence
  logits = session.run(["logits"], {"image": input_tensor})[0]
  probs = np.exp(logits - np.max(logits))
  probs /= np.sum(probs)
  pred_class = int(np.argmax(probs))
  return {"digit": pred_class, "confidence": round(float(probs[0, pred_class]), 4)}

# Example: print(predict_image_file("my_digit.png"))
