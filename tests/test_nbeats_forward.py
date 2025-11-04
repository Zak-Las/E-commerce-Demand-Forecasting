import torch
import pytest

from src.models.nbeats_module import NBeatsModule, NBeatsConfig


def test_nbeats_forward_shape():
    cfg = NBeatsConfig(input_length=64, forecast_length=10, num_stacks=2, num_blocks_per_stack=2, n_layers=2, layer_width=128)
    model = NBeatsModule(cfg)
    x = torch.randn(5, cfg.input_length)
    out = model(x)
    assert out.shape == (5, cfg.forecast_length)


def test_nbeats_training_step_runs():
    cfg = NBeatsConfig(input_length=32, forecast_length=8, num_stacks=1, num_blocks_per_stack=2, n_layers=2, layer_width=64)
    model = NBeatsModule(cfg)
    x = torch.randn(4, cfg.input_length)
    y = torch.randn(4, cfg.forecast_length)
    loss = model.training_step((x, y), 0)
    assert loss.item() >= 0.0


@pytest.mark.parametrize("input_len,forecast_len", [(30, 5), (60, 15)])
def test_nbeats_varied_lengths(input_len, forecast_len):
    cfg = NBeatsConfig(input_length=input_len, forecast_length=forecast_len, num_stacks=1, num_blocks_per_stack=1, n_layers=2, layer_width=64)
    model = NBeatsModule(cfg)
    x = torch.randn(3, input_len)
    out = model(x)
    assert out.shape == (3, forecast_len)
