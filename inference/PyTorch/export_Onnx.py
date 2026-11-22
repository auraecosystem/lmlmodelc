import torch
import torch.nn as nn


class CNNModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    )

  def forward(self, x):
    return self.net(x)


model = CNNModel().eval()
torch.onnx.export(
    model,
    (torch.randn(1, 3, 224, 224),),
    input_names=["image"],
    output_names=["logits"],
    dynamic_shapes={"x": {0: "batch"}},
    dynamo=True,
).save("cnn.onnx")
