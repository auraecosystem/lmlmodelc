class TabularMLP(nn.Module):

  def __init__(self, in_features=12):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(in_features, 64),
        nn.LayerNorm(64),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
    )

  def forward(self, x):
    return self.net(x)


model = TabularMLP().eval()
torch.onnx.export(
    model,
    (torch.randn(1, 12),),
    input_names=["features"],
    output_names=["output"],
    dynamic_shapes={"x": {0: "batch"}},
    dynamo=True,
).save("mlp.onnx")
