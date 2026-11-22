from onnxruntime.quantization import QuantType, quantize_dynamic

quantize_dynamic("transformer.onnx", "transformer_int8.onnx", weight_type=QuantType.QInt8)
