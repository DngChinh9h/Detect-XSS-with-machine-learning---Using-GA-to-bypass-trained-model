"""Phase 2A Char-level CNN+LSTM (Dual Pool v2) training, diagnostics, and domain-swap audit.

Run from ``xss_pipeline/``:
    ./venv/bin/python -m models.train_cnn_lstm
"""

from __future__ import annotations

import json
import random
from collections import Counter
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models._utils import (
    ARTIFACTS_DIR,
    DEFAULT_DATASET_CSV,
    SEED,
    compute_binary_metrics,
    compute_domain_swap_metrics,
    load_train_val_splits,
    prepare_domain_swap_val,
    tune_best_threshold,
)
from models.features import MAX_LEN, PAD_IDX, VOCAB_SIZE, encode_char_sequences

ARCH_NAME: str = "dual_pool_v2"
SEEDS: tuple[int, ...] = (SEED, 102, 202)
POS_WEIGHT: float = 1.0
LABEL_SMOOTHING_EPS: float = 0.05
BATCH_SIZE: int = 64
MAX_EPOCHS: int = 40
EARLY_STOP_PATIENCE: int = 8
LEARNING_RATE: float = 5e-4
WEIGHT_DECAY: float = 1e-2
GRAD_CLIP_MAX_NORM: float = 1.0
LR_SCHEDULER_FACTOR: float = 0.5
LR_SCHEDULER_PATIENCE: int = 3


def set_global_seed(seed: int) -> None:
    """Fix Python, NumPy, and PyTorch RNG seeds for deterministic runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CharCNNLSTM(nn.Module):
    """Character-level CNN + BiLSTM with dual pooling (Max + Avg) and dropout."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 32,
        padding_idx: int = PAD_IDX,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.conv_block = nn.Sequential(
            nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.25),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) -> emb: (B, L, E) -> conv input: (B, E, L)
        emb = self.embedding(x).transpose(1, 2)
        conv_out = self.conv_block(emb)
        # Back to sequence-first for LSTM: (B, T, 64) -> (B, T, 128)
        lstm_in = conv_out.transpose(1, 2)
        lstm_out, _ = self.lstm(lstm_in)
        # Dual pooling over time dimension: concat(MaxPool, AvgPool) -> (B, 256)
        max_pool, _ = torch.max(lstm_out, dim=1)
        avg_pool = torch.mean(lstm_out, dim=1)
        pooled = torch.cat([max_pool, avg_pool], dim=1)
        return self.fc(self.dropout(pooled))


@torch.no_grad()
def predict_proba_sequences(
    model: nn.Module,
    X_seq: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Run batched inference and return positive-class probabilities."""
    model.eval()
    probs: list[np.ndarray] = []
    tensor_x = torch.as_tensor(np.array(X_seq, dtype=np.int64, copy=True), dtype=torch.long)
    for start in range(0, len(tensor_x), batch_size):
        batch_x = tensor_x[start : start + batch_size].to(device)
        logits = model(batch_x).squeeze(1)
        batch_prob = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(batch_prob)
    return np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float64)


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    X_val_seq: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[float, np.ndarray]:
    """Compute hard-label BCE val_loss and positive-class probabilities on validation."""
    model.eval()
    val_criterion = nn.BCEWithLogitsLoss(reduction="sum")
    tensor_x = torch.as_tensor(np.array(X_val_seq, dtype=np.int64, copy=True), dtype=torch.long)
    tensor_y = torch.as_tensor(np.array(y_val, dtype=np.float32, copy=True), dtype=torch.float32)

    total_loss = 0.0
    probs: list[np.ndarray] = []
    n_samples = len(tensor_x)

    for start in range(0, n_samples, batch_size):
        batch_x = tensor_x[start : start + batch_size].to(device)
        batch_y = tensor_y[start : start + batch_size].to(device)
        logits = model(batch_x).squeeze(1)
        total_loss += float(val_criterion(logits, batch_y).item())
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())

    val_loss = total_loss / max(n_samples, 1)
    val_proba = np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float64)
    return float(val_loss), val_proba


def train_single_run(
    seed: int,
    X_train_seq: np.ndarray,
    y_train: np.ndarray,
    X_val_seq: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    """Train CharCNNLSTM for one seed with label smoothing, grad clipping, and val_loss ES."""
    set_global_seed(seed)

    train_dataset = TensorDataset(
        torch.as_tensor(np.array(X_train_seq, dtype=np.int64, copy=True), dtype=torch.long),
        torch.as_tensor(np.array(y_train, dtype=np.float32, copy=True), dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    model = CharCNNLSTM().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
    )
    train_criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_val_f1_050 = 0.0
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0

    print(f"\n--- Training seed={seed} (arch={ARCH_NAME}, pos_weight={POS_WEIGHT:.1f}) ---")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_smooth = batch_y * (1.0 - LABEL_SMOOTHING_EPS) + 0.5 * LABEL_SMOOTHING_EPS

            optimizer.zero_grad()
            logits = model(batch_x).squeeze(1)
            loss = train_criterion(logits, y_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            optimizer.step()

            batch_len = int(batch_x.size(0))
            total_loss += float(loss.item()) * batch_len
            total_samples += batch_len

        avg_train_loss = total_loss / max(total_samples, 1)
        val_loss, val_proba_epoch = evaluate_validation(model, X_val_seq, y_val, device=device)
        val_m_050 = compute_binary_metrics(y_val, val_proba_epoch, threshold=0.50)

        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1_050 = val_m_050["f1"]
            best_epoch = epoch
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"  [seed={seed}] ep={epoch:02d}/{MAX_EPOCHS} "
            f"tr_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} | "
            f"val@0.50 P={val_m_050['precision']:.4f} "
            f"R={val_m_050['recall']:.4f} F1={val_m_050['f1']:.4f} | "
            f"lr={current_lr:.2e} (best_ep={best_epoch}, no_imp={epochs_no_improve}/{EARLY_STOP_PATIENCE})"
        )

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(
                f"  -> Early stopping at epoch {epoch} "
                f"(best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f})."
            )
            break

    assert best_state_dict is not None
    model.load_state_dict(best_state_dict)
    model.to(device)

    restored_val_loss, val_proba_best = evaluate_validation(
        model, X_val_seq, y_val, device=device
    )
    tuned = tune_best_threshold(y_val, val_proba_best)
    val_metrics = {
        "precision": float(tuned["precision"]),
        "recall": float(tuned["recall"]),
        "f1": float(tuned["f1"]),
        "threshold": float(tuned["threshold"]),
        "best_val_loss": float(restored_val_loss),
        "val_f1_at_050": float(best_val_f1_050),
        "best_epoch": int(best_epoch),
    }

    print(
        f"-> Restored epoch {best_epoch} [seed={seed}]: "
        f"val_loss={restored_val_loss:.4f} | Val P={val_metrics['precision']:.4f} "
        f"R={val_metrics['recall']:.4f} F1={val_metrics['f1']:.4f} "
        f"best_threshold={val_metrics['threshold']:.2f}"
    )

    return {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(restored_val_loss),
        "val_metrics": val_metrics,
        "val_proba": val_proba_best,
        "state_dict": best_state_dict,
        "model": model,
    }


def _compute_calibration_quantiles(
    y_val: np.ndarray,
    val_proba: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute p10, p50, p90 predicted probabilities for Benign (0) and Malicious (1)."""
    benign_probs = val_proba[y_val == 0]
    mal_probs = val_proba[y_val == 1]

    def _quantiles(arr: np.ndarray) -> dict[str, float]:
        if len(arr) == 0:
            return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
        p10, p50, p90 = np.percentile(arr, [10, 50, 90])
        return {"p10": float(p10), "p50": float(p50), "p90": float(p90)}

    return {
        "class_0_benign": _quantiles(benign_probs),
        "class_1_malicious": _quantiles(mal_probs),
    }


def _diagnose_val_errors(
    val_df: Any,
    y_val: np.ndarray,
    val_proba: np.ndarray,
    threshold: float,
    top_n: int = 10,
) -> dict[str, Any]:
    """Count FP/FN on val at ``threshold`` and list top offending (group, family) pairs."""
    preds = (val_proba >= threshold).astype(np.int64)
    fp_mask = (y_val == 0) & (preds == 1)
    fn_mask = (y_val == 1) & (preds == 0)

    def _top_groups(mask: np.ndarray) -> list[dict[str, Any]]:
        if not np.any(mask):
            return []
        subset = val_df.loc[mask, ["group", "family"]]
        counts = Counter(zip(subset["group"].tolist(), subset["family"].tolist()))
        return [
            {"group": str(grp), "family": str(fam), "count": int(cnt)}
            for (grp, fam), cnt in counts.most_common(top_n)
        ]

    return {
        "threshold": float(threshold),
        "fp_count": int(np.sum(fp_mask)),
        "fn_count": int(np.sum(fn_mask)),
        "top_fp_group_family": _top_groups(fp_mask),
        "top_fn_group_family": _top_groups(fn_mask),
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Using device: cpu")

    train_df, val_df = load_train_val_splits(DEFAULT_DATASET_CSV)
    y_train = train_df["label"].to_numpy(dtype=np.int64, copy=True)
    y_val = val_df["label"].to_numpy(dtype=np.int64, copy=True)

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    print(
        f"Loaded dataset_final.csv (train={len(train_df)}, val={len(val_df)}; "
        f"train n_pos={n_pos}, n_neg={n_neg}; locked pos_weight={POS_WEIGHT:.1f})"
    )

    X_train_seq = encode_char_sequences(train_df["payload_decoded"].tolist(), max_len=MAX_LEN)
    X_val_seq = encode_char_sequences(val_df["payload_decoded"].tolist(), max_len=MAX_LEN)

    ordered_runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        run = train_single_run(
            seed=seed,
            X_train_seq=X_train_seq,
            y_train=y_train,
            X_val_seq=X_val_seq,
            y_val=y_val,
            device=device,
        )
        ordered_runs.append(run)

    # 3-Seed Summary Table
    print("\n=== CNN+LSTM (dual_pool_v2) 3-Seed Validation Summary ===")
    print(
        f"{'Seed':>6} {'BestEp':>7} {'BestValLoss':>12} "
        f"{'Val P':>8} {'Val R':>8} {'Val F1':>8} {'Best Thr':>9}"
    )
    print("-" * 65)
    for r in ordered_runs:
        m = r["val_metrics"]
        print(
            f"{r['seed']:>6} {r['best_epoch']:>7} {m['best_val_loss']:>12.4f} "
            f"{m['precision']:>8.4f} {m['recall']:>8.4f} "
            f"{m['f1']:>8.4f} {m['threshold']:>9.2f}"
        )

    precisions = [r["val_metrics"]["precision"] for r in ordered_runs]
    recalls = [r["val_metrics"]["recall"] for r in ordered_runs]
    f1s = [r["val_metrics"]["f1"] for r in ordered_runs]
    val_losses = [r["val_metrics"]["best_val_loss"] for r in ordered_runs]

    mean_std = {
        "val_loss_mean": float(np.mean(val_losses)),
        "val_loss_std": float(np.std(val_losses, ddof=1)),
        "precision_mean": float(np.mean(precisions)),
        "precision_std": float(np.std(precisions, ddof=1)),
        "recall_mean": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls, ddof=1)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s, ddof=1)),
    }
    print("-" * 65)
    print(
        f"Mean +/- Std (ddof=1): "
        f"ValLoss={mean_std['val_loss_mean']:.4f} +/- {mean_std['val_loss_std']:.4f} | "
        f"P={mean_std['precision_mean']:.4f} +/- {mean_std['precision_std']:.4f} | "
        f"R={mean_std['recall_mean']:.4f} +/- {mean_std['recall_std']:.4f} | "
        f"F1={mean_std['f1_mean']:.4f} +/- {mean_std['f1_std']:.4f}"
    )

    sorted_by_f1 = sorted(ordered_runs, key=lambda r: (r["val_metrics"]["f1"], r["seed"]))
    representative_seed = int(sorted_by_f1[1]["seed"])
    print(
        f"Representative seed (median Val F1): {representative_seed} "
        f"(Val F1={sorted_by_f1[1]['val_metrics']['f1']:.4f})"
    )

    # Probability Calibration Quantiles on Val
    print("\n=== Validation Probability Calibration Quantiles (p10 / p50 / p90) ===")
    calibration_by_seed: dict[str, Any] = {}
    for r in ordered_runs:
        seed = r["seed"]
        cal = _compute_calibration_quantiles(y_val, r["val_proba"])
        calibration_by_seed[str(seed)] = cal
        c0 = cal["class_0_benign"]
        c1 = cal["class_1_malicious"]
        print(
            f"  seed={seed:<3} | Benign(0): p10={c0['p10']:.4f}, p50={c0['p50']:.4f}, p90={c0['p90']:.4f} | "
            f"Malicious(1): p10={c1['p10']:.4f}, p50={c1['p50']:.4f}, p90={c1['p90']:.4f}"
        )

    # Error Diagnosis on Val
    print("\n=== Validation Error Diagnosis (FP / FN at best_threshold) ===")
    error_diag_by_seed: dict[str, Any] = {}
    for r in ordered_runs:
        seed = r["seed"]
        thr = r["val_metrics"]["threshold"]
        diag = _diagnose_val_errors(val_df, y_val, r["val_proba"], threshold=thr)
        error_diag_by_seed[str(seed)] = diag
        print(
            f"  seed={seed:<3} (thr={thr:.2f}): FP={diag['fp_count']}, FN={diag['fn_count']}"
        )
        if diag["top_fp_group_family"]:
            fp_summary = ", ".join(
                f"({item['group']}, {item['family']}): {item['count']}"
                for item in diag["top_fp_group_family"]
            )
            print(f"    Top FP (group, family): {fp_summary}")
        if diag["top_fn_group_family"]:
            fn_summary = ", ".join(
                f"({item['group']}, {item['family']}): {item['count']}"
                for item in diag["top_fn_group_family"]
            )
            print(f"    Top FN (group, family): {fn_summary}")

    # Layer 2 Audit Gate: Domain swap on val for all 3 seeds
    print("\n=== Shortcut Learning Audit Gate — Layer 2 (Domain Swap on Val) ===")
    swap_data = prepare_domain_swap_val(val_df)
    X_val_after_seq = encode_char_sequences(swap_data["full_after"], max_len=MAX_LEN)

    domain_swap_by_seed: dict[str, Any] = {}
    for r in ordered_runs:
        seed = r["seed"]
        thr = r["val_metrics"]["threshold"]
        proba_before = r["val_proba"]
        proba_after = predict_proba_sequences(r["model"], X_val_after_seq, device=device)
        m = compute_domain_swap_metrics(
            y_val=swap_data["y_val"],
            mal_attacker_mask=swap_data["mal_attacker_mask"],
            proba_before=proba_before,
            proba_after=proba_after,
            threshold=thr,
        )
        domain_swap_by_seed[str(seed)] = m
        print(
            f"  seed={seed:<3} (thr={thr:.2f}) | "
            f"Subset(n={m['subset_count']}) Recall: {m['subset_recall_before']:.4f} -> "
            f"{m['subset_recall_after']:.4f} (delta={m['subset_recall_delta']:+.4f}) | "
            f"Full Val(n_pos={m['full_mal_count']}) Recall: {m['full_recall_before']:.4f} -> "
            f"{m['full_recall_after']:.4f} (delta={m['full_recall_delta']:+.4f})"
        )

    # Save checkpoints and JSON report (overwrite cleanly)
    for r in ordered_runs:
        seed = r["seed"]
        ckpt_path = ARTIFACTS_DIR / f"cnn_lstm_seed{seed}.pt"
        torch.save(
            {
                "state_dict": r["state_dict"],
                "arch": ARCH_NAME,
                "seed": int(seed),
                "pos_weight": float(POS_WEIGHT),
                "best_threshold": float(r["val_metrics"]["threshold"]),
                "best_epoch": int(r["best_epoch"]),
                "val_metrics": r["val_metrics"],
            },
            ckpt_path,
        )

    report = {
        "arch": ARCH_NAME,
        "seeds": list(SEEDS),
        "device_used": str(device),
        "hyperparameters": {
            "vocab_size": VOCAB_SIZE,
            "max_len": MAX_LEN,
            "pos_weight": float(POS_WEIGHT),
            "label_smoothing_eps": float(LABEL_SMOOTHING_EPS),
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
            "scheduler": {
                "type": "ReduceLROnPlateau",
                "mode": "min",
                "factor": LR_SCHEDULER_FACTOR,
                "patience": LR_SCHEDULER_PATIENCE,
            },
        },
        "per_seed_val": {
            str(r["seed"]): {
                "best_epoch": int(r["best_epoch"]),
                "pos_weight": float(POS_WEIGHT),
                "val_metrics": r["val_metrics"],
            }
            for r in ordered_runs
        },
        "summary_3seeds": mean_std,
        "representative_seed": representative_seed,
        "calibration_quantiles_val": calibration_by_seed,
        "error_diagnosis_val": error_diag_by_seed,
        "audit_gate_layer2_domain_swap": domain_swap_by_seed,
    }

    report_path = ARTIFACTS_DIR / "cnn_lstm_val_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nSaved checkpoints and report to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
