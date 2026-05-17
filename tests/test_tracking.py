from qlora_dpo_finance.tracking import flatten


def test_flatten_keeps_scalar_params_only():
    payload = {"model": {"name": "qwen", "layers": 32}, "list_value": [1, 2]}

    assert flatten(payload) == {"model.name": "qwen", "model.layers": 32}
