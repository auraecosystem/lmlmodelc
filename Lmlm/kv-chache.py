from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

model_id = "lmlm/NeuroMindAI"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ORTModelForCausalLM.from_pretrained(model_id, export=True)
