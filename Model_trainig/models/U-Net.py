import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

# --- A. Quick Training Loop ---
model = MiniUNet(in_channels=3, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()  # Expects target mask of shape (B, H, W)

model.train()
for step in range(20):
  images = torch.randn(4, 3, 128, 128)
  masks = torch.randint(0, 2, (4, 128, 128))  # Pixel class indices (0 or 1)

  optimizer.zero_grad()
  loss = criterion(model(images), masks)
  loss.backward()
  optimizer.step()

model.eval()
torch.onnx.export(
    model,
    (torch.randn(1, 3, 128, 128),),
    input_names=["image"],
    output_names=["mask_logits"],
    dynamic_shapes={"x": {0: "batch_size"}},
    dynamo=True,
).save("mini_unet.onnx")

# --- B. ONNX Runtime Inference ---
session = ort.InferenceSession(
    "mini_unet.onnx", providers=["CPUExecutionProvider"]
)

test_image = np.random.randn(1, 3, 128, 128).astype(np.float32)
mask_logits = session.run(["mask_logits"], {"image": test_image})

# Convert (1, num_classes, H, W) logits -> (H, W) predicted pixel mask
predicted_mask = np.argmax(mask_logits, axis=1)
print("Predicted Mask Shape:", predicted_mask.shape)  # (128, 128)
