from onnxruntime.quantization import QuantType, quantize_dynamic

quantize_dynamic(
    model_input="classifier.onnx",
    model_output="classifier_int8.onnx",
    weight_type=QuantType.QUInt8,
)
