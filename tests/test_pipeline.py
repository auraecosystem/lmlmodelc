from lmlmodel import LMLConfig, LMLModel, LMLPipeline


def test_greedy_generation_length():
    cfg = LMLConfig(vocab_size=24, hidden_size=32, num_hidden_layers=1,
                    num_attention_heads=4, num_key_value_heads=2,
                    intermediate_size=64, max_position_embeddings=8)
    pipe = LMLPipeline(LMLModel(cfg), device="cpu")
    result = pipe.generate([1, 3], max_new_tokens=3)
    assert len(result) == 5
