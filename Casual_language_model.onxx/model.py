from model import NeuroMindConfig, NeuroMindForCausalLM
import torch

config = NeuroMindConfig()
model = NeuroMindForCausalLM(config).eval()

# Optional: load trained PyTorch weights before exporting
# model.load_state_dict(torch.load("neuromind_weights.pt", map_location="cpu"))

dummy_input_ids = torch.randint(
    0, config.vocab_size, (1, 8), dtype=torch.long
)

onnx_program = torch.onnx.export(
    model,
    (dummy_input_ids,),
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_shapes={"input_ids": {0: "batch_size", 1: "seq_len"}},
    dynamo=True,
)

# Saves the ONNX model to disk
onnx_program.save("neuromind_causal_lm.onnx")
print("Saved neuromind_causal_lm.onnx!")
