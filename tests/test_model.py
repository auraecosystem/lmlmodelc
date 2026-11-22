import torch
from lmlmodel import LMLConfig, LMLModel
from lmlmodel.training import causal_lm_loss


def tiny():
    return LMLModel(LMLConfig(vocab_size=32, hidden_size=32, num_hidden_layers=2,
                              num_attention_heads=4, num_key_value_heads=2,
                              intermediate_size=64, max_position_embeddings=16))


def test_forward_shape_and_finite():
    model = tiny()
    logits = model(torch.randint(0, 32, (2, 8)))
    assert logits.shape == (2, 8, 32)
    assert torch.isfinite(logits).all()


def test_causal_loss_and_backward():
    model = tiny()
    loss = causal_lm_loss(model, torch.randint(0, 32, (2, 8)))
    loss.backward()
    assert torch.isfinite(loss)


def test_tied_embeddings():
    model = tiny()
    assert model.lm_head.weight.data_ptr() == model.transformer.embed_tokens.weight.data_ptr()
