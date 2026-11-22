class TransformerNet(nn.Module):

  def __init__(self, vocab=1000, d_model=64):
    super().__init__()
    self.emb = nn.Embedding(vocab, d_model)
    layer = nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True)
    self.encoder = nn.TransformerEncoder(layer, num_layers=2)
    self.fc = nn.Linear(d_model, 2)

  def forward(self, input_ids):
    x = self.encoder(self.emb(input_ids))
    return self.fc(x[:, 0, :])  # CLS-style pooling


model = TransformerNet().eval()
dummy_ids = torch.randint(0, 1000, (2, 16))
torch.onnx.export(
    model,
    (dummy_ids,),
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_shapes={"input_ids": {0: "batch", 1: "seq_len"}},
    dynamo=True,
).save("transformer.onnx")
