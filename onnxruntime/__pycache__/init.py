import torch
import torch.nn as nn


class MultimodalModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.img_fc = nn.Linear(128, 32)
    self.meta_fc = nn.Linear(4, 32)
    self.head = nn.Linear(64, 2)

  def forward(self, image_feat, metadata):
    x1 = torch.relu(self.img_fc(image_feat))
    x2 = torch.relu(self.meta_fc(metadata))
    return self.head(torch.cat([x1, x2], dim=-1))


model = MultimodalModel().eval()

# Dummy inputs must match forward() arguments in exact order
dummy_img = torch.randn(1, 128)
dummy_meta = torch.randn(1, 4)

onnx_program = torch.onnx.export(
    model,
    (dummy_img, dummy_meta),
    input_names=["image_feat", "metadata"],
    output_names=["scores"],
    dynamic_shapes={
        "image_feat": {0: "batch"},
        "metadata": {0: "batch"},
    },
    dynamo=True,
)
onnx_program.save("multimodal.onnx")
