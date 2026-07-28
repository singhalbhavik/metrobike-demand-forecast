"""Offline forward-pass test for DemandLSTM -- shapes and non-negativity, no training."""
import torch

from src.models.lstm import DemandLSTM


def test_forward_pass_shape_and_non_negative():
    batch, seq_len, seq_dim, static_dim, num_stations = 7, 24, 9, 8, 5
    model = DemandLSTM(
        num_stations=num_stations,
        seq_feature_dim=seq_dim,
        static_feature_dim=static_dim,
        embedding_dim=4,
        hidden_size=16,
        num_layers=1,
        fc_hidden=8,
    )

    x_seq = torch.randn(batch, seq_len, seq_dim)
    x_static = torch.randn(batch, static_dim)
    station_idx = torch.randint(0, num_stations, (batch,))

    out = model(x_seq, x_static, station_idx)

    assert out.shape == (batch,)
    assert torch.all(out >= 0)
