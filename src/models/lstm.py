"""Global per-station LSTM demand model.

Predicts log1p(demand) at t+1 from a 24h lookback window plus static features (weekly-scale
lag/rolling signal + the target hour's own calendar attributes) and a learned station embedding.
Softplus output keeps predictions non-negative in log1p-space, so expm1(pred) >= 0 always holds.
"""
import torch
from torch import nn


class DemandLSTM(nn.Module):
    def __init__(
        self,
        num_stations: int,
        seq_feature_dim: int,
        static_feature_dim: int,
        embedding_dim: int = 8,
        hidden_size: int = 64,
        num_layers: int = 1,
        fc_hidden: int = 32,
    ):
        super().__init__()
        self.station_embedding = nn.Embedding(num_stations, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=seq_feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        head_in = hidden_size + embedding_dim + static_feature_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        x_seq: torch.Tensor,       # (batch, seq_len, seq_feature_dim)
        x_static: torch.Tensor,    # (batch, static_feature_dim)
        station_idx: torch.Tensor,  # (batch,)
    ) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x_seq)
        last_hidden = h_n[-1]  # (batch, hidden_size) -- top layer's final hidden state
        emb = self.station_embedding(station_idx)  # (batch, embedding_dim)
        combined = torch.cat([last_hidden, emb, x_static], dim=1)
        return self.head(combined).squeeze(-1)  # (batch,) predicted log1p(demand)
