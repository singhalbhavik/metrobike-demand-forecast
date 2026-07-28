"""
Stage 2 model pipeline: load features -> scale -> window -> train LSTM -> evaluate vs baselines.
Entry point:  python -m src.models.pipeline
              make train
"""
import gc
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.models.dataset import (
    STATIC_FEATURE_COLS,
    SEQ_FEATURE_COLS,
    add_cyclic_features,
    apply_scaler,
    build_station_index,
    build_windows,
    fit_scaler,
)
from src.models.evaluate import evaluate_all, print_results_table
from src.models.lstm import DemandLSTM
from src.models.train import get_device, train_model

_DATA_DIR = Path("data/processed")
_CONFIG_DIR = Path("config")
_MODEL_DIR = Path("models")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    config = yaml.safe_load((_CONFIG_DIR / "model.yaml").read_text())
    _set_seed(config["seed"])

    print("=" * 60)
    print("Stage 2: LSTM demand model")
    print("=" * 60)

    # ── 1. Load Stage 1 outputs ─────────────────────────────────────────────
    print("Loading Stage 1 outputs...")
    features = pd.read_parquet(_DATA_DIR / "features.parquet")
    train_raw = pd.read_parquet(_DATA_DIR / "train.parquet")
    val_raw = pd.read_parquet(_DATA_DIR / "val.parquet")
    test_raw = pd.read_parquet(_DATA_DIR / "test.parquet")
    boundaries = yaml.safe_load((_CONFIG_DIR / "splits.yaml").read_text())
    print(f"  Feature matrix : {len(features):>10,} rows")
    print()

    # ── 2. Cyclic features + train-only scaler ──────────────────────────────
    print("Adding cyclic features and fitting scaler on train split only...")
    features = add_cyclic_features(features)
    scaler = fit_scaler(train_raw)
    features_scaled = apply_scaler(features, scaler)
    station_index = build_station_index(features)
    print(f"  Stations       : {len(station_index):>10,}")
    print()

    # ── 3. Sliding windows ───────────────────────────────────────────────────
    print(
        f"Building sliding windows (seq_len={config['seq_len']}, "
        f"train_stride={config['train_stride']}, eval_stride={config['eval_stride']})..."
    )
    windows = build_windows(
        features_scaled,
        boundaries,
        station_index,
        seq_len=config["seq_len"],
        train_stride=config["train_stride"],
        eval_stride=config["eval_stride"],
    )
    for split in ("train", "val", "test"):
        print(f"  {split.capitalize():<6}: {len(windows[split]):>10,} windows")
    print()

    train_ds, val_ds, test_ds = windows["train"], windows["val"], windows["test"]

    # Windows now hold everything training needs (compact per-station arrays); drop the
    # wide DataFrames -- this machine has 8GB RAM and features_scaled has ~20+ columns.
    del features, features_scaled
    gc.collect()

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print("Training LSTM...")
    model = DemandLSTM(
        num_stations=len(station_index),
        seq_feature_dim=len(SEQ_FEATURE_COLS),
        static_feature_dim=len(STATIC_FEATURE_COLS),
        embedding_dim=config["embedding_dim"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        fc_hidden=config["fc_hidden"],
    )
    model, history = train_model(model, train_ds, val_ds, config)
    print()

    # ── 5. Evaluate vs baselines ────────────────────────────────────────────
    print("Evaluating LSTM vs naive baselines...")
    device = get_device()
    loaders = {
        "val": DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False),
        "test": DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False),
    }
    raw_splits = {"val": val_raw, "test": test_raw}
    results = evaluate_all(model, loaders, raw_splits, device)
    print()
    print_results_table(results)
    print()

    # ── 6. Save artifacts ────────────────────────────────────────────────────
    _MODEL_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), _MODEL_DIR / "lstm_best.pt")

    preprocessing = {"scaler": scaler, "station_index": station_index}
    with open(_CONFIG_DIR / "preprocessing.yaml", "w") as f:
        yaml.dump(preprocessing, f, default_flow_style=False, sort_keys=True)

    with open(_CONFIG_DIR / "metrics.yaml", "w") as f:
        yaml.dump({"results": results, "history": history}, f, default_flow_style=False, sort_keys=False)

    print(f"Saved checkpoint to {_MODEL_DIR / 'lstm_best.pt'}")
    print(f"Saved preprocessing artifacts to {_CONFIG_DIR / 'preprocessing.yaml'}")
    print(f"Saved metrics to {_CONFIG_DIR / 'metrics.yaml'}")
    print("Stage 2 complete.")


if __name__ == "__main__":
    main()
