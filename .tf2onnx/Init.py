import numpy as np
import onnxruntime as ort

# 1. Create an inference session (falls back to CPU if CUDA isn't available)
session = ort.InferenceSession(
    "classifier.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# 2. Inspect expected input/output names dynamically
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 3. Prepare NumPy input data (must match expected dtype, usually float32)
sample_data = np.array(
    [[0.5, -1.2, 3.3, 0.1], [1.0, 2.0, -0.5, 0.8]], dtype=np.float32
)

# 4. Run inference
outputs = session.run([output_name], {input_name: sample_data})
logits = outputs[0]

# 5. Convert logits to predicted class indices
predictions = np.argmax(logits, axis=1)
print("Logits:\n", logits)
print("Predicted classes:", predictions)
