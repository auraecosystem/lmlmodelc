import onnxruntime as ort

opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = 4  # Match physical CPU cores for CPU inference

session = ort.InferenceSession(
    "cnn.onnx", sess_options=opts, providers=["CPUExecutionProvider"]
)
