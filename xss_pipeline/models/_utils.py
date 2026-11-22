"""Shared dataset loading, threshold tuning, and shortcut-audit helpers for Phase 2A."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

SEED: int = 42
PIPELINE_DIR: Path = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_CSV: Path = PIPELINE_DIR / "dataset_final.csv"
ARTIFACTS_DIR: Path = Path(__file__).resolve().parent / "artifacts"

ATTACKER_DOMAIN: str = "attacker.test"
NEUTRAL_DOMAIN: str = "neutral-check.org"


def load_train_val_splits(
    csv_path: Path | str = DEFAULT_DATASET_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ``dataset_final.csv`` and return isolated ``train`` and ``val`` DataFrames.

    Strictly forbids any ``split == "test"`` samples from entering the Phase 2A
    pipeline, asserts zero NaNs on core columns, and verifies zero family overlap
    between ``train`` and ``val``.
    """
    import pandas as pd

    path = Path(csv_path)
    df = pd.read_csv(path)

    required_cols = {"payload_decoded", "label", "family", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    for col in ("payload_decoded", "label", "family", "split"):
        assert not df[col].isna().any(), f"Found NaN values in column '{col}'."

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)

    assert len(train_df) > 0, "Train split is empty."
    assert len(val_df) > 0, "Validation split is empty."
    assert not (train_df["split"] == "test").any(), "Test samples leaked into train_df."
    assert not (val_df["split"] == "test").any(), "Test samples leaked into val_df."
    assert set(train_df["split"].unique()) == {"train"}, "Unexpected split in train_df."
    assert set(val_df["split"].unique()) == {"val"}, "Unexpected split in val_df."

    overlap = set(train_df["family"].tolist()) & set(val_df["family"].tolist())
    assert len(overlap) == 0, f"Family leakage between train and val: {sorted(overlap)[:5]}"

    return train_df, val_df


def compute_binary_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute binary precision, recall, and F1 at ``threshold``."""
    import numpy as np

    y_true_b = np.asarray(y_true, dtype=np.int64) == 1
    y_pred_b = np.asarray(y_proba, dtype=np.float64) >= float(threshold)

    tp = int(np.sum(y_true_b & y_pred_b))
    fp = int(np.sum((~y_true_b) & y_pred_b))
    fn = int(np.sum(y_true_b & (~y_pred_b)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def tune_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    start: float = 0.10,
    stop: float = 0.90,
    step: float = 0.01,
) -> dict[str, float]:
    """Scan thresholds in ``[start, stop]`` to maximize F1 (tie-break closest to 0.50)."""
    n_steps = int(round((stop - start) / step)) + 1
    thresholds = [round(start + i * step, 2) for i in range(n_steps)]

    best_thr = 0.50
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": -1.0}
    best_key = (-1.0, -1.0)

    for thr in thresholds:
        metrics = compute_binary_metrics(y_true, y_proba, thr)
        # Prefer higher F1, then threshold closest to 0.50.
        key = (round(metrics["f1"], 12), -round(abs(thr - 0.50), 12))
        if key > best_key:
            best_key = key
            best_thr = float(thr)
            best_metrics = metrics

    return {
        "threshold": float(best_thr),
        "precision": float(best_metrics["precision"]),
        "recall": float(best_metrics["recall"]),
        "f1": float(best_metrics["f1"]),
    }


def prepare_domain_swap_val(val_df: pd.DataFrame) -> dict[str, Any]:
    """Prepare Layer 2 domain-replacement validation inputs (malicious ``attacker.test``)."""
    import numpy as np

    labels = val_df["label"].to_numpy(dtype=np.int64)
    texts = val_df["payload_decoded"].tolist()
    mal_attacker_mask = (labels == 1) & np.array(
        [ATTACKER_DOMAIN in s for s in texts],
        dtype=bool,
    )
    texts_after = [
        s.replace(ATTACKER_DOMAIN, NEUTRAL_DOMAIN) if int(lbl) == 1 else s
        for s, lbl in zip(texts, labels.tolist())
    ]

    return {
        "y_val": labels,
        "mal_attacker_mask": mal_attacker_mask,
        "subset_count": int(np.sum(mal_attacker_mask)),
        "full_mal_count": int(np.sum(labels == 1)),
        "full_before": texts,
        "full_after": texts_after,
    }


def compute_domain_swap_metrics(
    y_val: np.ndarray,
    mal_attacker_mask: np.ndarray,
    proba_before: np.ndarray,
    proba_after: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute Recall before/after domain replacement on the malicious subset and full val."""
    import numpy as np

    subset_count = int(np.sum(mal_attacker_mask))
    if subset_count > 0:
        subset_before = float(np.mean(proba_before[mal_attacker_mask] >= threshold))
        subset_after = float(np.mean(proba_after[mal_attacker_mask] >= threshold))
        subset_delta = subset_after - subset_before
    else:
        subset_before = 0.0
        subset_after = 0.0
        subset_delta = 0.0

    full_before = compute_binary_metrics(y_val, proba_before, threshold)["recall"]
    full_after = compute_binary_metrics(y_val, proba_after, threshold)["recall"]
    full_delta = full_after - full_before

    return {
        "subset_count": subset_count,
        "subset_recall_before": float(subset_before),
        "subset_recall_after": float(subset_after),
        "subset_recall_delta": float(subset_delta),
        "full_mal_count": int(np.sum(y_val == 1)),
        "full_recall_before": float(full_before),
        "full_recall_after": float(full_after),
        "full_recall_delta": float(full_delta),
    }
