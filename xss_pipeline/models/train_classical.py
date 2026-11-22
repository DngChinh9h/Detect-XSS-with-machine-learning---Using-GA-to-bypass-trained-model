"""Phase 2A classical model training, ablation, and shortcut-learning audit (SVM & RF).

Run from ``xss_pipeline/``:
    ./venv/bin/python -m models.train_classical
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

from models._utils import (
    ARTIFACTS_DIR,
    ATTACKER_DOMAIN,
    DEFAULT_DATASET_CSV,
    SEED,
    compute_domain_swap_metrics,
    load_train_val_splits,
    prepare_domain_swap_val,
    tune_best_threshold,
)
from models.features import STRUCTURAL_FEATURE_NAMES, XSSFeatureExtractor

SVM_C_GRID: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0)
CLASS_WEIGHT_GRID: tuple[str | None, ...] = (None, "balanced")
RF_MAX_DEPTH_GRID: tuple[int | None, ...] = (20, 40, None)
RF_MIN_SAMPLES_LEAF_GRID: tuple[int, ...] = (1, 2)
ABLATION_MODES: tuple[str, ...] = ("full", "tfidf_only", "structural_only")


def is_attacker_test_ngram(feature_name: str) -> bool:
    """Return True if ``feature_name`` is a >=2-char substring of ``attacker.test``."""
    if feature_name in STRUCTURAL_FEATURE_NAMES:
        return False
    return len(feature_name) >= 2 and feature_name in ATTACKER_DOMAIN


def _build_svm(
    c_val: float,
    class_weight: str | None,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> CalibratedClassifierCV:
    base_svc = LinearSVC(
        C=c_val,
        class_weight=class_weight,
        random_state=SEED,
        max_iter=5000,
    )
    return CalibratedClassifierCV(
        estimator=base_svc,
        method="sigmoid",
        cv=cv_splits,
    )


def _build_rf(
    max_depth: int | None,
    min_samples_leaf: int,
    class_weight: str | None,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=SEED,
        n_jobs=-1,
    )


def _fit_and_eval_val(
    estimator: Any,
    X_train: Any,
    y_train: np.ndarray,
    X_val: Any,
    y_val: np.ndarray,
) -> tuple[Any, dict[str, float]]:
    estimator.fit(X_train, y_train)
    val_proba = estimator.predict_proba(X_val)[:, 1]
    val_metrics = tune_best_threshold(y_val, val_proba)
    return estimator, val_metrics


def _verify_cv_splits(
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    families: np.ndarray,
) -> None:
    for fold_idx, (tr_idx, va_idx) in enumerate(cv_splits):
        overlap = set(families[tr_idx].tolist()) & set(families[va_idx].tolist())
        assert len(overlap) == 0, f"Family leakage inside CV fold {fold_idx}: {overlap}"


def _search_best_svm(
    X_train: Any,
    y_train: np.ndarray,
    X_val: Any,
    y_val: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    print("\n=== Tuning SVM (LinearSVC + StratifiedGroupKFold Platt Calibration) ===")
    best_record: dict[str, Any] | None = None
    best_estimator: Any = None
    all_trials: list[dict[str, Any]] = []

    for c_val in SVM_C_GRID:
        for cw in CLASS_WEIGHT_GRID:
            est = _build_svm(c_val=c_val, class_weight=cw, cv_splits=cv_splits)
            fitted_est, metrics = _fit_and_eval_val(est, X_train, y_train, X_val, y_val)
            params = {"C": c_val, "class_weight": cw}
            trial = {"params": params, "val_metrics": metrics}
            all_trials.append(trial)
            print(
                f"  SVM C={c_val:<4} class_weight={str(cw):<8} | "
                f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                f"F1={metrics['f1']:.4f} thr={metrics['threshold']:.2f}"
            )
            score_key = (round(metrics["f1"], 12), round(metrics["recall"], 12))
            if best_record is None or score_key > best_record["score_key"]:
                best_record = {
                    "params": params,
                    "val_metrics": metrics,
                    "score_key": score_key,
                }
                best_estimator = fitted_est

    assert best_record is not None and best_estimator is not None
    print(
        f"-> Best SVM: {best_record['params']} | "
        f"Val F1={best_record['val_metrics']['f1']:.4f} "
        f"(thr={best_record['val_metrics']['threshold']:.2f})"
    )
    return {
        "estimator": best_estimator,
        "params": best_record["params"],
        "val_metrics": best_record["val_metrics"],
        "trials": all_trials,
    }


def _search_best_rf(
    X_train: Any,
    y_train: np.ndarray,
    X_val: Any,
    y_val: np.ndarray,
) -> dict[str, Any]:
    print("\n=== Tuning Random Forest (n_estimators=300) ===")
    best_record: dict[str, Any] | None = None
    best_estimator: Any = None
    all_trials: list[dict[str, Any]] = []

    for max_depth in RF_MAX_DEPTH_GRID:
        for min_leaf in RF_MIN_SAMPLES_LEAF_GRID:
            for cw in CLASS_WEIGHT_GRID:
                est = _build_rf(
                    max_depth=max_depth,
                    min_samples_leaf=min_leaf,
                    class_weight=cw,
                )
                fitted_est, metrics = _fit_and_eval_val(est, X_train, y_train, X_val, y_val)
                params = {
                    "n_estimators": 300,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_leaf,
                    "class_weight": cw,
                }
                trial = {"params": params, "val_metrics": metrics}
                all_trials.append(trial)
                print(
                    f"  RF depth={str(max_depth):<4} min_leaf={min_leaf} "
                    f"class_weight={str(cw):<8} | "
                    f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                    f"F1={metrics['f1']:.4f} thr={metrics['threshold']:.2f}"
                )
                score_key = (round(metrics["f1"], 12), round(metrics["recall"], 12))
                if best_record is None or score_key > best_record["score_key"]:
                    best_record = {
                        "params": params,
                        "val_metrics": metrics,
                        "score_key": score_key,
                    }
                    best_estimator = fitted_est

    assert best_record is not None and best_estimator is not None
    print(
        f"-> Best RF:  {best_record['params']} | "
        f"Val F1={best_record['val_metrics']['f1']:.4f} "
        f"(thr={best_record['val_metrics']['threshold']:.2f})"
    )
    return {
        "estimator": best_estimator,
        "params": best_record["params"],
        "val_metrics": best_record["val_metrics"],
        "trials": all_trials,
    }


def _run_ablation(
    train_texts: list[str],
    y_train: np.ndarray,
    val_texts: list[str],
    y_val: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    svm_best: dict[str, Any],
    rf_best: dict[str, Any],
) -> list[dict[str, Any]]:
    print("\n=== Ablation Study on Validation Split ===")
    print(f"{'Model':<6} {'Mode':<16} {'Val P':>8} {'Val R':>8} {'Val F1':>8} {'Best Thr':>9}")
    print("-" * 60)

    ablation_rows: list[dict[str, Any]] = []
    svm_params = svm_best["params"]
    rf_params = rf_best["params"]

    for mode in ABLATION_MODES:
        if mode == "full":
            svm_metrics = svm_best["val_metrics"]
            rf_metrics = rf_best["val_metrics"]
        else:
            ext = XSSFeatureExtractor(mode=mode)
            X_tr_m = ext.fit_transform(train_texts)
            X_va_m = ext.transform(val_texts)

            svm_est = _build_svm(
                c_val=svm_params["C"],
                class_weight=svm_params["class_weight"],
                cv_splits=cv_splits,
            )
            _, svm_metrics = _fit_and_eval_val(svm_est, X_tr_m, y_train, X_va_m, y_val)

            rf_est = _build_rf(
                max_depth=rf_params["max_depth"],
                min_samples_leaf=rf_params["min_samples_leaf"],
                class_weight=rf_params["class_weight"],
            )
            _, rf_metrics = _fit_and_eval_val(rf_est, X_tr_m, y_train, X_va_m, y_val)

        for model_name, m in (("SVM", svm_metrics), ("RF", rf_metrics)):
            ablation_rows.append({"model": model_name, "mode": mode, **m})
            print(
                f"{model_name:<6} {mode:<16} "
                f"{m['precision']:>8.4f} {m['recall']:>8.4f} "
                f"{m['f1']:>8.4f} {m['threshold']:>9.2f}"
            )

    return ablation_rows


def _extract_top_features(
    scores: np.ndarray,
    feature_names: list[str],
    top_k: int = 20,
) -> tuple[list[dict[str, Any]], int]:
    top_indices = np.argsort(scores)[::-1][:top_k]
    rows: list[dict[str, Any]] = []
    top10_hits = 0
    for rank, idx in enumerate(top_indices, start=1):
        name = feature_names[int(idx)]
        weight = float(scores[int(idx)])
        is_shortcut = is_attacker_test_ngram(name)
        if rank <= 10 and is_shortcut:
            top10_hits += 1
        rows.append(
            {
                "rank": rank,
                "feature": name,
                "score": weight,
                "is_attacker_test_ngram": is_shortcut,
            }
        )
    return rows, top10_hits


def _print_top_features(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{title}")
    for row in rows:
        flag = " [ATTACKER_TEST]" if row["is_attacker_test_ngram"] else ""
        print(f"  #{row['rank']:>2}: {row['feature']!r:<24} score={row['score']:+.6f}{flag}")


def _audit_layer1_weights(
    best_svm: CalibratedClassifierCV,
    best_rf: RandomForestClassifier,
    feature_names: list[str],
) -> dict[str, Any]:
    print("\n=== Shortcut Learning Audit Gate — Layer 1 (Feature Weights) ===")
    fold_coefs: list[np.ndarray] = []
    for calibrated_clf in best_svm.calibrated_classifiers_:
        inner = getattr(calibrated_clf, "estimator", getattr(calibrated_clf, "base_estimator", None))
        if inner is None or not hasattr(inner, "coef_"):
            raise RuntimeError("Could not extract coef_ from calibrated LinearSVC fold.")
        fold_coefs.append(np.asarray(inner.coef_, dtype=np.float64).ravel())

    mean_svm_coef = np.mean(fold_coefs, axis=0)
    assert mean_svm_coef.shape[0] == len(feature_names)
    assert best_rf.feature_importances_.shape[0] == len(feature_names)

    svm_top20, svm_top10_hits = _extract_top_features(mean_svm_coef, feature_names, top_k=20)
    rf_top20, rf_top10_hits = _extract_top_features(
        best_rf.feature_importances_, feature_names, top_k=20
    )

    _print_top_features("SVM Top 20 Positive Mean Weights (5-fold LinearSVC):", svm_top20)
    _print_top_features("Random Forest Top 20 Feature Importances:", rf_top20)

    status = "FAIL" if (svm_top10_hits >= 2 or rf_top10_hits >= 2) else "PASS"
    print(
        f"\n-> Layer 1 Gate Verdict: {status} "
        f"(SVM Top-10 attacker.test hits={svm_top10_hits}, "
        f"RF Top-10 attacker.test hits={rf_top10_hits})"
    )

    return {
        "status": status,
        "svm_top10_attacker_test_hits": svm_top10_hits,
        "rf_top10_attacker_test_hits": rf_top10_hits,
        "svm_top20": svm_top20,
        "rf_top20": rf_top20,
    }


def _audit_layer2_domain_swap(
    extractor_full: XSSFeatureExtractor,
    best_svm: CalibratedClassifierCV,
    svm_threshold: float,
    best_rf: RandomForestClassifier,
    rf_threshold: float,
    X_val: Any,
    val_df: Any,
) -> dict[str, Any]:
    print("\n=== Shortcut Learning Audit Gate — Layer 2 (Domain Swap on Val) ===")
    swap_data = prepare_domain_swap_val(val_df)
    X_val_after = extractor_full.transform(swap_data["full_after"])

    results: dict[str, Any] = {}
    for model_name, model, thr in (
        ("SVM", best_svm, svm_threshold),
        ("RF", best_rf, rf_threshold),
    ):
        proba_before = model.predict_proba(X_val)[:, 1]
        proba_after = model.predict_proba(X_val_after)[:, 1]
        m = compute_domain_swap_metrics(
            y_val=swap_data["y_val"],
            mal_attacker_mask=swap_data["mal_attacker_mask"],
            proba_before=proba_before,
            proba_after=proba_after,
            threshold=thr,
        )
        results[model_name.lower()] = m
        print(
            f"  {model_name:<3} (thr={thr:.2f}) | "
            f"Subset(n={m['subset_count']}) Recall: {m['subset_recall_before']:.4f} -> "
            f"{m['subset_recall_after']:.4f} (delta={m['subset_recall_delta']:+.4f}) | "
            f"Full Val(n_pos={m['full_mal_count']}) Recall: {m['full_recall_before']:.4f} -> "
            f"{m['full_recall_after']:.4f} (delta={m['full_recall_delta']:+.4f})"
        )

    return results


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_train_val_splits(DEFAULT_DATASET_CSV)
    train_texts = train_df["payload_decoded"].tolist()
    val_texts = val_df["payload_decoded"].tolist()
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    y_val = val_df["label"].to_numpy(dtype=np.int64)
    train_families = train_df["family"].to_numpy()

    print(
        f"Loaded dataset_final.csv (train={len(train_df)}, val={len(val_df)}; "
        f"train pos={int(np.sum(y_train == 1))}, neg={int(np.sum(y_train == 0))}; "
        f"val pos={int(np.sum(y_val == 1))}, neg={int(np.sum(y_val == 0))})"
    )

    extractor_full = XSSFeatureExtractor(mode="full")
    X_train = extractor_full.fit_transform(train_texts)
    X_val = extractor_full.transform(val_texts)
    feature_names = extractor_full.get_feature_names()
    assert X_train.shape[1] == len(feature_names)
    print(f"Extracted full features: X_train={X_train.shape}, X_val={X_val.shape}")

    cv_splits = list(
        StratifiedGroupKFold(n_splits=5).split(X_train, y_train, groups=train_df["family"])
    )
    _verify_cv_splits(cv_splits, train_families)

    svm_best = _search_best_svm(X_train, y_train, X_val, y_val, cv_splits)
    rf_best = _search_best_rf(X_train, y_train, X_val, y_val)

    ablation_rows = _run_ablation(
        train_texts=train_texts,
        y_train=y_train,
        val_texts=val_texts,
        y_val=y_val,
        cv_splits=cv_splits,
        svm_best=svm_best,
        rf_best=rf_best,
    )

    layer1_audit = _audit_layer1_weights(
        best_svm=svm_best["estimator"],
        best_rf=rf_best["estimator"],
        feature_names=feature_names,
    )
    layer2_audit = _audit_layer2_domain_swap(
        extractor_full=extractor_full,
        best_svm=svm_best["estimator"],
        svm_threshold=svm_best["val_metrics"]["threshold"],
        best_rf=rf_best["estimator"],
        rf_threshold=rf_best["val_metrics"]["threshold"],
        X_val=X_val,
        val_df=val_df,
    )

    joblib.dump(extractor_full, ARTIFACTS_DIR / "feature_extractor_full.joblib")
    joblib.dump(svm_best["estimator"], ARTIFACTS_DIR / "svm_best.joblib")
    joblib.dump(rf_best["estimator"], ARTIFACTS_DIR / "rf_best.joblib")

    report = {
        "seed": SEED,
        "dataset": {
            "train_count": int(len(train_df)),
            "val_count": int(len(val_df)),
            "train_pos": int(np.sum(y_train == 1)),
            "train_neg": int(np.sum(y_train == 0)),
            "val_pos": int(np.sum(y_val == 1)),
            "val_neg": int(np.sum(y_val == 0)),
            "num_features_full": int(len(feature_names)),
        },
        "svm": {
            "best_params": svm_best["params"],
            "best_val_metrics": svm_best["val_metrics"],
            "trials": svm_best["trials"],
        },
        "random_forest": {
            "best_params": rf_best["params"],
            "best_val_metrics": rf_best["val_metrics"],
            "trials": rf_best["trials"],
        },
        "ablation_val": ablation_rows,
        "audit_gate": {
            "layer1_feature_weights": layer1_audit,
            "layer2_domain_swap": layer2_audit,
        },
    }

    report_path = ARTIFACTS_DIR / "classical_val_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nSaved artifacts to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
