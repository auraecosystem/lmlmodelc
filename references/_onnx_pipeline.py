import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 1. Define a CNN for 28x28 grayscale images
class DigitCNN(nn.Module):

  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),  # -> 16 x 14 x 14
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),  # -> 32 x 7 x 7
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 10),
    )

  def forward(self, x):
    return self.net(x)


# 2. Load MNIST dataset & train briefly (1 epoch for a fast demo)
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_ds = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_ds = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

model = DigitCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

model.train()
for images, labels in train_loader:
  optimizer.zero_grad()
  loss = loss_fn(model(images), labels)
  loss.backward()
  optimizer.step()

# 3. Export trained model to ONNX
model.eval()
onnx_path = "digit_cnn.onnx"
dummy_input = torch.randn(1, 1, 28, 28)

onnx_program = torch.onnx.export(
    model,
    (dummy_input,),
    input_names=["image"],
    output_names=["logits"],
    dynamic_shapes={"x": {0: "batch_size"}},
    dynamo=True,
)
onnx_program.save(onnx_path)
onnx.checker.check_model(onnx.load(onnx_path))

# 4. Run ONNX Inference with Preprocessing & Postprocessing
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def predict_digit(raw_uint8_image: np.ndarray):
  # Preprocess: scale [0, 255] ->, normalize, add batch/channel dims
  img = raw_uint8_image.astype(np.float32) / 255.0
  img = (img - 0.1307) / 0.3081
  input_tensor = img[np.newaxis, np.newaxis, :, :]  # Shape: (1, 1, 28, 28)

  # Run ONNX Runtime
  logits = session.run(["logits"], {"image": input_tensor})[0]

  # Postprocess: convert logits -> probabilities via Softmax
  exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
  probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
  pred_class = int(np.argmax(probs, axis=1))
  confidence = float(probs[0, pred_class])
  return pred_class, confidence


# Test on a real image from the MNIST test split
sample_raw_img = test_ds.data.numpy()  # Raw 28x28 uint8 image
actual_label = int(test_ds.targets)

pred, conf = predict_digit(sample_raw_img)
print(
    f"Actual digit: {actual_label} | Predicted: {pred} (Confidence: {conf:.2%})"
)
