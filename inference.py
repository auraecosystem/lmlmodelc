import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn


# 1. Define a PyTorch Neural Network
class NeuralClassifier(nn.Module):

  def __init__(self, in_features=8, hidden_dim=32, num_classes=3):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, num_classes),
    )

  def forward(self, x):
    return self.net(x)


# 2. Initialize & switch to evaluation mode
model = NeuralClassifier().eval()
dummy_input = torch.randn(1, 8)
onnx_path = "neural_classifier.onnx"

# 3. Export to ONNX (with dynamic batch size)
onnx_program = torch.onnx.export(
    model,
    (dummy_input,),
    input_names=["features"],
    output_names=["logits"],
    dynamic_shapes={"x": {0: "batch_size"}},
    dynamo=True,
)
onnx_program.save(onnx_path)

# 4. Validate the exported ONNX graph
onnx.checker.check_model(onnx.load(onnx_path))

# 5. Run inference with ONNX Runtime
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
test_batch = np.random.randn(4, 8).astype(np.float32)  # Batch of 4 samples

ort_logits = session.run(["logits"], {"features": test_batch})[0]

# 6. Verify PyTorch and ONNX outputs match
with torch.no_grad():
  torch_logits = model(torch.from_numpy(test_batch)).numpy()

np.testing.assert_allclose(torch_logits, ort_logits, rtol=1e-3, atol=1e-5)
print("Success! Predicted classes:", np.argmax(ort_logits, axis=1))
