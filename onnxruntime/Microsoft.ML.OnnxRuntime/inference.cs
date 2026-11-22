using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using var session = new InferenceSession("classifier.onnx");
var tensor = new DenseTensor<float>(new float[] { 0.5f, -1.2f, 3.3f, 0.1f }, new[] { 1, 4 });
using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", tensor) });
var logits = results.First().AsEnumerable<float>().ToArray();
