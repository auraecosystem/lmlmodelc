import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

  def __init__(self, in_ch: int, out_ch: int):
    super().__init__()
    self.net = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

  def forward(self, x):
    return self.net(x)


class MiniUNet(nn.Module):

  def __init__(self, in_channels: int = 3, num_classes: int = 2):
    super().__init__()
    self.enc1 = DoubleConv(in_channels, 32)
    self.pool = nn.MaxPool2d(2)
    self.enc2 = DoubleConv(32, 64)
    self.dec1 = DoubleConv(64 + 32, 32)
    self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    e1 = self.enc1(x)
    e2 = self.enc2(self.pool(e1))
    up = F.interpolate(e2, scale_factor=2.0, mode="bilinear", align_corners=False)
    d1 = self.dec1(torch.cat([up, e1], dim=1))
    return self.out_conv(d1)  # Output shape: (batch, num_classes, H, W)


model = MiniUNet().eval()
dummy_img = torch.randn(1, 3, 128, 128)

torch.onnx.export(
    model,
    (dummy_img,),
    input_names=["image"],
    output_names=["mask_logits"],
    dynamic_shapes={"x": {0: "batch_size"}},
    dynamo=True,
).save("mini_unet.onnx")
