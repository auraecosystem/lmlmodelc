import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

# --- A. Quick Training Loop ---
model = TabularNet(num_features=5, cat_cardinalities=)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

model.train()
for step in range(30):
  num_batch = torch.randn(32, 5)
  cat_batch = torch.stack([
      torch.randint(0, 4, (32,)),
      torch.randint(0, 10, (32,)),
      torch.randint(0, 3, (32,)),
  ], dim=1)
  targets = torch.randint(0, 2, (32, 1)).float()

  optimizer.zero_grad()
  loss = criterion(model(num_batch, cat_batch), targets)
  loss.backward()
  optimizer.step()

model.eval()
torch.onnx.export(
    model,
    (torch.randn(2, 5), torch.tensor([,], dtype=torch.long)),
    input_names=["numeric_features", "categorical_ids"],
    output_names=["prediction"],
    dynamic_shapes={"num_x": {0: "batch_size"}, "cat_x": {0: "batch_size"}},
    dynamo=True,
).save("tabular_net.onnx")

# --- B. ONNX Runtime Inference ---
session = ort.InferenceSession(
    "tabular_net.onnx", providers=["CPUExecutionProvider"]
)

inputs = {
    "numeric_features": np.array([[0.5, -1.1, 2.3, 0.0, 0.8]], dtype=np.float32),
    "categorical_ids": np.array([[]], dtype=np.int64),
}
raw_logit = session.run(["prediction"], inputs)

# Apply sigmoid to convert raw logit into a 0.0–1.0 probability
probability = float(1.0 / (1.0 + np.exp(-raw_logit[0, 0])))
print(f"Predicted Probability: {probability:.2%}")
