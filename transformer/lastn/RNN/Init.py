class LSTMNet(nn.Module):

  def __init__(self, input_size=8, hidden_size=32):
    super().__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    self.fc = nn.Linear(hidden_size, 1)

  def forward(self, seq):
    out, _ = self.lstm(seq)
    return self.fc(out[:, -1, :])  # Last time step


model = LSTMNet().eval()
torch.onnx.export(
    model,
    (torch.randn(2, 20, 8),),
    input_names=["seq"],
    output_names=["pred"],
    dynamic_shapes={"seq": {0: "batch", 1: "time_steps"}},
    dynamo=True,
).save("lstm.onnx")
