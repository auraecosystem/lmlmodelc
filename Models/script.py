import torch
import torch.nn as nn


# 1. Residual Block (core building block of modern vision networks)
class ResidualBlock(nn.Module):

  def __init__(self, channels: int):
    super().__init__()
    self.conv_block = nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(channels),
    )
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Add skip connection (input + transformed output)
    return self.relu(x + self.conv_block(x))


# 2. Full Image Classification Network
class ImageClassifier(nn.Module):

  def __init__(self, in_channels: int = 3, num_classes: int = 5):
    super().__init__()
    self.stem = nn.Sequential(
        nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        ),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
    )
    self.stage1 = ResidualBlock(32)
    self.downsample = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
    )
    self.stage2 = ResidualBlock(64)
    self.head = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),  # Works with any input image resolution
        nn.Flatten(),
        nn.Dropout(p=0.2),
        nn.Linear(64, num_classes),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.stem(x)
    x = self.stage1(x)
    x = self.downsample(x)
    x = self.stage2(x)
    return self.head(x)


# 3. Instantiate and Export to ONNX
if __name__ == "__main__":
  model = ImageClassifier(in_channels=3, num_classes=5).eval()
  dummy_image = torch.randn(1, 3, 128, 128)

  onnx_program = torch.onnx.export(
      model,
      (dummy_image,),
      input_names=["image"],
      output_names=["logits"],
      dynamic_shapes={"x": {0: "batch_size"}},
      dynamo=True,
  )
  onnx_program.save("image_classifier.onnx")
  print("Exported image_classifier.onnx successfully!")
