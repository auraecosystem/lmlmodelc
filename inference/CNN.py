import torch
import torch.nn as nn


class ConvNet(nn.Module):

  def __init__(self, num_classes=10):
    super().__init__()
    self.features = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
    )
    self.classifier = nn.Linear(16, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    return self.classifier(x)


model = ConvNet().eval()  # Crucial: freezes BatchNorm & disables Dropout
dummy_input = torch.randn(1, 3, 64, 64)

onnx_program = torch.onnx.export(
    model,
    (dummy_input,),
    input_names=["image"],
    output_names=["logits"],
    dynamic_shapes={"x": {0: "batch_size"}},
    dynamo=True,
)
onnx_program.save("convnet.onnx")
