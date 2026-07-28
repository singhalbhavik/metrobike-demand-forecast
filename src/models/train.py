"""Training loop: Adam + MSE on log1p(demand), early stopping on val RMSE (real units)."""
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.evaluate import compute_rmse_mae, predict


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(model: nn.Module, train_ds, val_ds, config: dict) -> tuple[nn.Module, list[dict]]:
    device = get_device()
    print(f"  Device: {device}")
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    epochs_since_improvement = 0
    history = []

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        train_losses = []
        for x_seq, x_static, station_idx, y in train_loader:
            x_seq = x_seq.to(device)
            x_static = x_static.to(device)
            station_idx = station_idx.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred_log1p = model(x_seq, x_static, station_idx)
            loss = loss_fn(pred_log1p, torch.log1p(y))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_preds, val_actuals = predict(model, val_loader, device)
        val_rmse, val_mae = compute_rmse_mae(val_preds, val_actuals)
        train_loss = float(np.mean(train_losses))
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_rmse": val_rmse, "val_mae": val_mae}
        )
        print(
            f"  Epoch {epoch:>2}/{config['max_epochs']}: "
            f"train_loss={train_loss:.4f}  val_RMSE={val_rmse:.4f}  val_MAE={val_mae:.4f}"
        )

        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= config["early_stopping_patience"]:
                print(
                    f"  Early stopping at epoch {epoch} "
                    f"(no val improvement for {config['early_stopping_patience']} epochs)"
                )
                break

    model.load_state_dict(best_state)
    return model, history
