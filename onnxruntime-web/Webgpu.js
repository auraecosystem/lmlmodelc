import * as ort from 'onnxruntime-web';

async function runInference() {
  const session = await ort.InferenceSession.create('./classifier.onnx', {
    executionProviders: ['webgpu', 'wasm'],
  });
  const tensor = new ort.Tensor('float32', Float32Array.from([0.5, -1.2, 3.3, 0.1]), [1, 4]);
  const results = await session.run({ input: tensor });
  console.log('Output logits:', results.logits.data);
}
runInference();
