import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

# --- A. Quick Training Loop ---
model = TextTransformer(vocab_size=10000, num_classes=3)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

model.train()
for step in range(20):  # Replace with your DataLoader loop
  batch_ids = torch.randint(0, 10000, (16, 24))
  labels = torch.randint(0, 3, (16,))

  optimizer.zero_grad()
  loss = criterion(model(batch_ids), labels)
  loss.backward()
  optimizer.step()

# Export after training (using the export code from earlier)
model.eval()
torch.onnx.export(
    model,
    (torch.randint(0, 10000, (1, 24)),),
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_shapes={"input_ids": {0: "batch_size", 1: "seq_len"}},
    dynamo=True,
).save("text_transformer.onnx")

# --- B. ONNX Runtime Inference (Variable Sequence Length) ---
session = ort.InferenceSession(
    "text_transformer.onnx", providers=["CPUExecutionProvider"]
)

# Notice seq_len=10 differs from training length 24—works due to dynamic_shapes
new_tokens = np.array(
    [[42, 105, 889, 12, 4, 99, 201, 7, 33, 1]], dtype=np.int64
)
logits = session.run(["logits"], {"input_ids": new_tokens})
pred_class = int(np.argmax(logits, axis=1))
print("Transformer Predicted Class:", pred_class)
