"""Optional adapter for an already-exported ONNX model."""
from pathlib import Path


class ONNXPipeline:
    def __init__(self, model_path: str | Path, *, providers: list[str] | None = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install the optional dependency: pip install lmlmodelc[onnx]") from exc
        self.session = ort.InferenceSession(str(model_path), providers=providers)

    def __call__(self, input_ids):
        import numpy as np
        name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {name: np.asarray(input_ids, dtype=np.int64)})
        return outputs[0]
