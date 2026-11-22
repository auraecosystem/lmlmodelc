For deep neural networks (such as CNNs, Transformers, RNNs, or MLPs), exporting from PyTorch to ONNX works best when you optimize both the graph structure and numerical precision for your target hardware.
Here is how to handle neural networks cleanly in a PyTorch + PyCharm workflow:

1. Exporting Deep Neural Networks (CNN Example)

When exporting vision or multi-layer neural networks, pass a sample tensor with the exact channel/feature dimensions and set dynamic batch axes so you can run single images or batches at inference time:
python
import torch
import torch.nn as nn


class ConvNet(nn.Module):

  def __init__(self, num_classes=10):
    super().__init__()
    self.features = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
    )
    self.classifier = nn.Linear(16, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    return self.classifier(x)


model = ConvNet().eval()  # Crucial: freezes BatchNorm & disables Dropout
dummy_input = torch.randn(1, 3, 64, 64)

onnx_program = torch.onnx.export(
    model,
    (dummy_input,),
    input_names=["image"],
    output_names=["logits"],
    dynamic_shapes={"x": {0: "batch_size"}},
    dynamo=True,
)
onnx_program.save("convnet.onnx")
Use code with caution.

2. Key Optimization Steps for Neural Networks

Deep neural networks have heavy matrix multiplications and convolutions. Apply these optimizations before deploying:

Optimization	What It Does	When to Use
Graph Fusion	Merges adjacent layers (e.g., Conv + BatchNorm + ReLU) into a single fused kernel.	Always — ONNX Runtime applies this automatically at session startup (GraphOptimizationLevel.ORT_ENABLE_ALL).
FP16 (Half Precision)	Converts 32-bit floats to 16-bit floats, cutting memory in half and boosting speed.	GPUs (NVIDIA Tensor Cores, DirectML, CoreML) with minimal accuracy loss.
INT8 Quantization	Compresses weights/activations to 8-bit integers (up to 2–4x faster).	CPUs & Edge/Mobile devices; use Static Quantization with calibration data for CNNs, or Dynamic Quantization for Transformers/RNNs.


3. Common Neural Network Export Pitfalls

• Forgetting model.eval(): Leaving BatchNorm or Dropout in training mode causes mismatched predictions and bloated ONNX graphs.
• In-place operations: Operations like x += 1 or ReLU(inplace=True) occasionally cause graph tracing issues; standard out-of-place ops (x = x + 1) export more reliably.
• Data-dependent shapes: Dynamic slicing or torch.nonzero() can fail during export unless handled with torch.export dynamic shape constraints.
To give you the exact architecture template and quantization script, what kind of neural network are you building:
• Architecture (CNN/vision, Transformer/NLP, RNN/time-series, or custom MLP)?
• Target hardware (CPU, NVIDIA GPU, or mobile/edge)?
