import torch
import torch.nn as nn


class TabularNet(nn.Module):

  def __init__(self, num_features: int, cat_cardinalities: list[int]):
    super().__init__()
    self.embeddings = nn.ModuleList([
        nn.Embedding(num_classes, min(16, (num_classes + 1) // 2))
        for num_classes in cat_cardinalities
    ])
    emb_dim = sum(e.embedding_dim for e in self.embeddings)
    self.num_norm = nn.BatchNorm1d(num_features)

    self.mlp = nn.Sequential(
        nn.Linear(num_features + emb_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
    )

  def forward(
      self, num_x: torch.Tensor, cat_x: torch.Tensor
  ) -> torch.Tensor:
    emb_outs = [emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)]
    x = torch.cat([self.num_norm(num_x)] + emb_outs, dim=1)
    return self.mlp(x)


model = TabularNet(num_features=5, cat_cardinalities=[10, 4, 25]).eval()
dummy_num = torch.randn(2, 5)
dummy_cat = torch.tensor([[1, 0, 12], [4, 2, 7]], dtype=torch.long)

torch.onnx.export(
    model,
    (dummy_num, dummy_cat),
    input_names=["numeric_features", "categorical_ids"],
    output_names=["prediction"],
    dynamic_shapes={
        "num_x": {0: "batch_size"},
        "cat_x": {0: "batch_size"},
    },
    dynamo=True,
).save("tabular_net.onnx")
