import torch
import torch.nn as nn


# Define a simple model
class SimpleClassifier(nn.Module):

  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 3),  # 3 output classes
    )

  def forward(self, x):
    return self.net(x)


model = SimpleClassifier()
model.eval()  # Switch to evaluation mode

# Create a dummy input matching the expected shape (batch_size=1, features=4)
dummy_input = (torch.randn(1, 4),)

# Export using the Dynamo-based exporter
onnx_program = torch.onnx.export(
    model,
    dummy_input,
    input_names=["input"],
    output_names=["logits"],
    dynamic_shapes={"x": {0: "batch_size"}},  # Allow variable batch sizes
    dynamo=True,
)

# Save to disk
onnx_program.save("classifier.onnx")
print("Saved classifier.onnx")
