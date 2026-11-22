"""Final one-time evaluation on the held-out ``split == "test"`` set (1,191 samples).

Run from ``xss_pipeline/``:
    ./venv/bin/python -m models.evaluate_test
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models._utils import ARTIFACTS_DIR, DEFAULT_DATASET_CSV
from models.predictor import ModelPredictor
from preprocessing.decoder import decode_payload

EXPECTED_TEST_COUNT: int = 1191
CNN_SEEDS: tuple[int, ...] = (42, 102, 202)
CONTEXT_ORDER: tuple[str, ...] = ("html_body", "attribute", "js_string", "url")
SOURCE_GROUP_ORDER: tuple[str, ...] = ("Tainted", "Obfuscated")
BENIGN_GROUP_ORDER: tuple[str, ...] = ("Clean", "Unusual")


def _load_test_split(csv_path: Path = DEFAULT_DATASET_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    assert len(test_df) == EXPECTED_TEST_COUNT, (
        f"Expected {EXPECTED_TEST_COUNT} test rows, found {len(test_df)}."
    )
    for col in ("payload", "payload_decoded", "label", "group", "context", "obfuscation", "family"):
        assert not test_df[col].isna().any(), f"Found NaN in test_df['{col}']."

    decoded_from_raw = [decode_payload(p) for p in test_df["payload"].tolist()]
    assert decoded_from_raw == test_df["payload_decoded"].tolist(), (
        "Mismatch between decode_payload(payload) and payload_decoded on test split!"
    )
    return test_df


def _compute_confusion_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    y_t = y_true == 1
    y_p = y_pred == 1
    tp = int(np.sum(y_t & y_p))
    fp = int(np.sum((~y_t) & y_p))
    tn = int(np.sum((~y_t) & (~y_p)))
    fn = int(np.sum(y_t & (~y_p)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _recall_on_mask(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    n_pos = int(np.sum(mask & (y_true == 1)))
    if n_pos == 0:
        return 0.0
    tp = int(np.sum(mask & (y_true == 1) & (y_pred == 1)))
    return float(tp / n_pos)


def _fpr_on_mask(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    n_neg = int(np.sum(mask & (y_true == 0)))
    fp = int(np.sum(mask & (y_true == 0) & (y_pred == 1)))
    fpr = float(fp / n_neg) if n_neg > 0 else 0.0
    return {"n": n_neg, "fp": fp, "fpr": fpr}


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
    }


def _diagnose_test_errors(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_n: int = 10,
) -> dict[str, Any]:
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)

    def _top_tuples(mask: np.ndarray) -> list[dict[str, Any]]:
        if not np.any(mask):
            return []
        subset = test_df.loc[mask, ["group", "context", "family"]]
        counts = Counter(
            zip(subset["group"].tolist(), subset["context"].tolist(), subset["family"].tolist())
        )
        return [
            {"group": str(g), "context": str(c), "family": str(f), "count": int(cnt)}
            for (g, c, f), cnt in counts.most_common(top_n)
        ]

    return {
        "fp_count": int(np.sum(fp_mask)),
        "fn_count": int(np.sum(fn_mask)),
        "top_fp": _top_tuples(fp_mask),
        "top_fn": _top_tuples(fn_mask),
    }


def main() -> None:
    with open(ARTIFACTS_DIR / "cnn_lstm_val_report.json", encoding="utf-8") as f:
        cnn_val_report = json.load(f)
    rep_seed = int(cnn_val_report["representative_seed"])

    test_df = _load_test_split(DEFAULT_DATASET_CSV)
    raw_payloads = test_df["payload"].tolist()
    y_test = test_df["label"].to_numpy(dtype=np.int64, copy=True)

    print(
        f"Loaded test split: n={len(test_df)} "
        f"(pos={int(np.sum(y_test == 1))}, neg={int(np.sum(y_test == 0))}), "
        f"SSOT decode_payload check PASSED."
    )

    model_specs: list[tuple[str, str, int | None]] = [
        ("rf", "RF", None),
        ("svm", "SVM", None),
        *[
            (
                f"cnn_lstm_seed{s}",
                f"CNN+LSTM (seed={s}{', rep' if s == rep_seed else ''})",
                s,
            )
            for s in CNN_SEEDS
        ],
    ]

    eval_results: dict[str, dict[str, Any]] = {}
    for key, label, seed in model_specs:
        predictor = ModelPredictor(
            model_name="cnn_lstm" if seed is not None else key,
            artifacts_dir=ARTIFACTS_DIR,
            seed=seed,
        )
        probs = predictor.predict_proba(raw_payloads)
        preds = (probs >= predictor.threshold).astype(np.int64)
        metrics = _compute_confusion_metrics(y_test, preds)
        eval_results[key] = {
            "label": label,
            "model_name": predictor.model_name,
            "seed": predictor.seed,
            "threshold": float(predictor.threshold),
            "metrics": metrics,
            "probs": probs,
            "preds": preds,
        }

    cnn_keys = [f"cnn_lstm_seed{s}" for s in CNN_SEEDS]
    rep_key = f"cnn_lstm_seed{rep_seed}"

    cnn_p_ms = _mean_std([eval_results[k]["metrics"]["precision"] for k in cnn_keys])
    cnn_r_ms = _mean_std([eval_results[k]["metrics"]["recall"] for k in cnn_keys])
    cnn_f1_ms = _mean_std([eval_results[k]["metrics"]["f1"] for k in cnn_keys])

    # Table 1: Overall Test Metrics
    print("\n=== Bang 1: Tong quan Test Metrics (split == 'test', n=1191) ===")
    print(
        f"{'Model':<26} {'Thr':>6} {'P':>8} {'R':>8} {'F1':>8} "
        f"{'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}"
    )
    print("-" * 82)
    for key, _, _ in model_specs:
        res = eval_results[key]
        m = res["metrics"]
        print(
            f"{res['label']:<26} {res['threshold']:>6.2f} "
            f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} "
            f"{m['tp']:>5} {m['fp']:>5} {m['tn']:>5} {m['fn']:>5}"
        )
    print("-" * 82)
    print(
        f"{'CNN+LSTM (Mean +/- Std)':<26} {'-':>6} "
        f"{cnn_p_ms['mean']:.4f}+/-{cnn_p_ms['std']:.4f}  "
        f"{cnn_r_ms['mean']:.4f}+/-{cnn_r_ms['std']:.4f}  "
        f"{cnn_f1_ms['mean']:.4f}+/-{cnn_f1_ms['std']:.4f}"
    )

    # Helper to compute grouped Recall rows (RF, SVM, CNN Mean +/- Std, CNN rep)
    def _build_recall_row(mask: np.ndarray) -> dict[str, Any]:
        n_pos = int(np.sum(mask & (y_test == 1)))
        rf_r = _recall_on_mask(y_test, eval_results["rf"]["preds"], mask)
        svm_r = _recall_on_mask(y_test, eval_results["svm"]["preds"], mask)
        cnn_rs = [_recall_on_mask(y_test, eval_results[k]["preds"], mask) for k in cnn_keys]
        cnn_ms = _mean_std(cnn_rs)
        rep_r = _recall_on_mask(y_test, eval_results[rep_key]["preds"], mask)
        return {
            "n": n_pos,
            "rf_recall": rf_r,
            "svm_recall": svm_r,
            "cnn_lstm_per_seed": {str(s): r for s, r in zip(CNN_SEEDS, cnn_rs)},
            "cnn_lstm_mean": cnn_ms["mean"],
            "cnn_lstm_std": cnn_ms["std"],
            "cnn_lstm_rep_recall": rep_r,
        }

    # Table 2: Recall by Context
    print("\n=== Bang 2: Recall theo 4 ngu canh (context, label == 1) ===")
    print(
        f"{'Context':<14} {'n':>5} {'RF':>8} {'SVM':>8} "
        f"{'CNN+LSTM Mean+/-Std':>21} {f'CNN (seed={rep_seed})':>16}"
    )
    print("-" * 76)
    context_breakdown: dict[str, Any] = {}
    for ctx in CONTEXT_ORDER:
        mask = (test_df["label"].to_numpy() == 1) & (test_df["context"].to_numpy() == ctx)
        row = _build_recall_row(mask)
        context_breakdown[ctx] = row
        ms_str = f"{row['cnn_lstm_mean']:.4f} +/- {row['cnn_lstm_std']:.4f}"
        print(
            f"{ctx:<14} {row['n']:>5} {row['rf_recall']:>8.4f} {row['svm_recall']:>8.4f} "
            f"{ms_str:>21} {row['cnn_lstm_rep_recall']:>16.4f}"
        )

    # Table 3: Recall by Group (Tainted vs Obfuscated) & Obfuscation breakdown
    print("\n=== Bang 3: Recall theo nhom nguon (Tainted vs Obfuscated) & Obfuscation ===")
    print(
        f"{'Segment':<22} {'n':>5} {'RF':>8} {'SVM':>8} "
        f"{'CNN+LSTM Mean+/-Std':>21} {f'CNN (seed={rep_seed})':>16}"
    )
    print("-" * 84)
    source_group_breakdown: dict[str, Any] = {}
    for grp in SOURCE_GROUP_ORDER:
        mask = (test_df["label"].to_numpy() == 1) & (test_df["group"].to_numpy() == grp)
        row = _build_recall_row(mask)
        source_group_breakdown[grp] = row
        ms_str = f"{row['cnn_lstm_mean']:.4f} +/- {row['cnn_lstm_std']:.4f}"
        print(
            f"{grp:<22} {row['n']:>5} {row['rf_recall']:>8.4f} {row['svm_recall']:>8.4f} "
            f"{ms_str:>21} {row['cnn_lstm_rep_recall']:>16.4f}"
        )

    print("-" * 84)
    obf_mask_all = (test_df["label"].to_numpy() == 1) & (test_df["group"].to_numpy() == "Obfuscated")
    obf_values = sorted(test_df.loc[obf_mask_all, "obfuscation"].unique().tolist())
    obfuscation_breakdown: dict[str, Any] = {}
    for obf in obf_values:
        mask = obf_mask_all & (test_df["obfuscation"].to_numpy() == obf)
        row = _build_recall_row(mask)
        obfuscation_breakdown[obf] = row
        ms_str = f"{row['cnn_lstm_mean']:.4f} +/- {row['cnn_lstm_std']:.4f}"
        print(
            f"  {obf:<20} {row['n']:>5} {row['rf_recall']:>8.4f} {row['svm_recall']:>8.4f} "
            f"{ms_str:>21} {row['cnn_lstm_rep_recall']:>16.4f}"
        )

    # Table 4: FPR on Benign Groups (Clean, Unusual, All Benign)
    print("\n=== Bang 4: False Positive Rate (FPR) tren tap Benign ===")
    print(
        f"{'Benign Group':<14} {'n':>5} {'RF':>13} {'SVM':>13} "
        f"{'CNN s42':>13} {'CNN s102':>13} {'CNN s202':>13} {'CNN Mean+/-Std':>18}"
    )
    print("-" * 104)
    fpr_breakdown: dict[str, Any] = {}
    benign_segments: list[tuple[str, np.ndarray]] = [
        (grp, (test_df["label"].to_numpy() == 0) & (test_df["group"].to_numpy() == grp))
        for grp in BENIGN_GROUP_ORDER
    ]
    benign_segments.append(("All Benign", test_df["label"].to_numpy() == 0))

    for seg_name, mask in benign_segments:
        n_neg = int(np.sum(mask))
        per_model = {}
        for key, _, _ in model_specs:
            per_model[key] = _fpr_on_mask(y_test, eval_results[key]["preds"], mask)
        cnn_fpr_ms = _mean_std([per_model[k]["fpr"] for k in cnn_keys])
        fpr_breakdown[seg_name] = {
            "n": n_neg,
            "per_model": per_model,
            "cnn_lstm_fpr_mean": cnn_fpr_ms["mean"],
            "cnn_lstm_fpr_std": cnn_fpr_ms["std"],
        }

        def _fmt_fpr(d: dict[str, Any]) -> str:
            return f"{d['fpr']:.4f} ({d['fp']})"

        ms_str = f"{cnn_fpr_ms['mean']:.4f}+/-{cnn_fpr_ms['std']:.4f}"
        print(
            f"{seg_name:<14} {n_neg:>5} "
            f"{_fmt_fpr(per_model['rf']):>13} {_fmt_fpr(per_model['svm']):>13} "
            f"{_fmt_fpr(per_model['cnn_lstm_seed42']):>13} "
            f"{_fmt_fpr(per_model['cnn_lstm_seed102']):>13} "
            f"{_fmt_fpr(per_model['cnn_lstm_seed202']):>13} {ms_str:>18}"
        )

    # Table 5: Error Breakdown (rf, svm, cnn_lstm representative seed)
    print(f"\n=== Bang 5: Phan tich Loi tren Test (RF, SVM, CNN+LSTM seed={rep_seed}) ===")
    error_breakdown: dict[str, Any] = {}
    for key in ("rf", "svm", rep_key):
        res = eval_results[key]
        diag = _diagnose_test_errors(test_df, y_test, res["preds"])
        error_breakdown[key] = diag
        print(f"  {res['label']}: FP={diag['fp_count']}, FN={diag['fn_count']}")
        if diag["top_fp"]:
            fp_str = ", ".join(
                f"({it['group']}, {it['context']}, {it['family']}): {it['count']}"
                for it in diag["top_fp"]
            )
            print(f"    Top FP: {fp_str}")
        if diag["top_fn"]:
            fn_str = ", ".join(
                f"({it['group']}, {it['context']}, {it['family']}): {it['count']}"
                for it in diag["top_fn"]
            )
            print(f"    Top FN: {fn_str}")

    report = {
        "dataset": {
            "split": "test",
            "total_count": int(len(test_df)),
            "pos_count": int(np.sum(y_test == 1)),
            "neg_count": int(np.sum(y_test == 0)),
        },
        "representative_seed": rep_seed,
        "overall_test_metrics": {
            key: {
                "label": eval_results[key]["label"],
                "model_name": eval_results[key]["model_name"],
                "seed": eval_results[key]["seed"],
                "threshold": eval_results[key]["threshold"],
                **eval_results[key]["metrics"],
            }
            for key, _, _ in model_specs
        },
        "cnn_lstm_summary_3seeds": {
            "precision": cnn_p_ms,
            "recall": cnn_r_ms,
            "f1": cnn_f1_ms,
        },
        "recall_by_context": context_breakdown,
        "recall_by_source_group": source_group_breakdown,
        "recall_by_obfuscation": obfuscation_breakdown,
        "fpr_by_benign_group": fpr_breakdown,
        "error_breakdown": error_breakdown,
    }

    report_path = ARTIFACTS_DIR / "test_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nSaved test evaluation report to {report_path}")


if __name__ == "__main__":
    main()
