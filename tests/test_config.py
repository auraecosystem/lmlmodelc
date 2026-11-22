import pytest
from lmlmodel import LMLConfig


def test_config_round_trip():
    cfg = LMLConfig()
    assert LMLConfig.from_dict(cfg.to_dict()) == cfg


def test_aliases():
    cfg = LMLConfig.from_dict({"num_kv_heads": 2, "n_layers": 2})
    assert cfg.num_key_value_heads == 2
    assert cfg.num_hidden_layers == 2


def test_invalid_attention_geometry():
    with pytest.raises(ValueError):
        LMLConfig(hidden_size=130, num_attention_heads=4)
