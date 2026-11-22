#include <onnxruntime_cxx_api.h>
#include <vector>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXInference");
    Ort::Session session(env, L"classifier.onnx", Ort::SessionOptions{});
    std::vector<float> input_data = {0.5f, -1.2f, 3.3f, 0.1f};
    std::vector<int64_t> input_shape = {1, 4};
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());
    const char* in_names[] = {"input"};
    const char* out_names[] = {"logits"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &input_tensor, 1, out_names, 1);
    return 0;
}
