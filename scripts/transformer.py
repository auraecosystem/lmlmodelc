import torch
import torch.nn as nn


class TextTransformer(nn.Module):

  def __init__(
      self,
      vocab_size: int = 10000,
      d_model: int = 128,
      nhead: int = 4,
      num_layers: int = 2,
      num_classes: int = 3,
  ):
    super().__init__()
    self.token_emb = nn.Embedding(vocab_size, d_model)
    self.pos_emb = nn.Embedding(512, d_model)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=256,
        dropout=0.1,
        activation="gelu",
        batch_first=True,
    )
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    self.classifier = nn.Linear(d_model, num_classes)

  def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    seq_len = input_ids.size(1)
    positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    x = self.token_emb(input_ids) + self.pos_emb(positions)
    x = self.encoder(x)
    return self.classifier(x.mean(dim=1))  # Mean pooling across tokens


model = TextTransformer().eval()
dummy_ids = torch.randint(0, 10000, (1, 32))

torch.onnx.export(
    model,
    (dummy_ids,),
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_shapes={"input_ids": {0: "batch_size", 1: "seq_len"}},
    dynamo=True,
).save("text_transformer.onnx")
