"""RMSE/MAE computation (real demand units) and the LSTM-vs-baselines comparison table."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.baselines import BASELINES, evaluate_baseline


def compute_rmse_mae(preds: np.ndarray, actuals: np.ndarray) -> tuple[float, float]:
    err = preds - actuals
    return float(np.sqrt(np.mean(err**2))), float(np.mean(np.abs(err)))


def predict(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Runs the model over a loader, returns (predictions, actuals) in real demand units."""
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x_seq, x_static, station_idx, y in loader:
            x_seq, x_static, station_idx = x_seq.to(device), x_static.to(device), station_idx.to(device)
            pred_log1p = model(x_seq, x_static, station_idx)
            preds.append(torch.expm1(pred_log1p).cpu().numpy())
            actuals.append(y.numpy())
    return np.concatenate(preds), np.concatenate(actuals)


def evaluate_all(model, loaders: dict[str, DataLoader], raw_splits: dict, device: torch.device) -> dict:
    """
    loaders: {"val": DataLoader, "test": DataLoader} over StationSequenceDataset windows.
    raw_splits: {"val": val_df, "test": test_df} -- Stage 1 parquet splits, for baseline comparison.
    Returns nested metrics dict: {"LSTM": {"val": {...}, "test": {...}}, "Naive persistence ...": {...}}
    """
    results: dict[str, dict[str, dict[str, float]]] = {"LSTM": {}}
    for split, loader in loaders.items():
        preds, actuals = predict(model, loader, device)
        rmse, mae = compute_rmse_mae(preds, actuals)
        results["LSTM"][split] = {"rmse": rmse, "mae": mae}

    for name, col in BASELINES.items():
        results[name] = {}
        for split, df in raw_splits.items():
            results[name][split] = evaluate_baseline(df, col)

    return results


def print_results_table(results: dict) -> None:
    print(f"{'Model':<32}{'Val RMSE':>10}{'Val MAE':>10}{'Test RMSE':>11}{'Test MAE':>10}")
    for name, splits in results.items():
        val, test = splits["val"], splits["test"]
        print(
            f"{name:<32}{val['rmse']:>10.4f}{val['mae']:>10.4f}"
            f"{test['rmse']:>11.4f}{test['mae']:>10.4f}"
        )
