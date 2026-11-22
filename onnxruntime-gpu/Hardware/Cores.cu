import onnx
from onnxconverter_common import float16
import onnxruntime as ort

model = onnx.load("cnn.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "cnn_fp16.onnx")

# Run with TensorRT (fastest) -> CUDA -> CPU fallback
session = ort.InferenceSession(
    "cnn_fp16.onnx",
    providers=[
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
)
