"""Phan tich ket qua Giai doan 3 (Buoc 3.4): Ma tran 6.2, Error Analysis, Cross-Model Transferability.

Chay tren terminal Ubuntu host (sau khi da hoan tat Buoc 3.3):
    ./venv/bin/python -m ga.analyze_phase3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from models._utils import ARTIFACTS_DIR
from models.features import STRUCTURAL_FEATURE_NAMES, extract_structural_features
from models.predictor import ModelPredictor
from preprocessing.decoder import decode_payload

MODELS: tuple[str, ...] = ("rf", "svm", "cnn_lstm")
MODEL_DISPLAY: dict[str, str] = {
    "rf": "Random Forest (rf)",
    "svm": "Linear SVM (svm)",
    "cnn_lstm": "CNN+LSTM (s202)",
}
CONTEXTS: tuple[str, ...] = ("html_body", "attribute", "js_string", "url")
LARGE_CONTEXTS: tuple[str, ...] = ("html_body", "attribute", "js_string")
MECHANISMS: tuple[str, ...] = ("html_event", "js_protocol", "js_direct")
NA_CELLS: dict[str, frozenset[str]] = {
    "html_body": frozenset(),
    "attribute": frozenset({"js_protocol", "js_direct"}),
    "js_string": frozenset({"js_protocol"}),
    "url": frozenset({"html_event", "js_direct"}),
}
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay file ket qua: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_model_runs(results_dir: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Doc toan bo file GA (5 seed) va Brute Force (url) cho 3 mo hinh; bo qua random_*.json."""
    data: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for model in MODELS:
        data[model] = {}
        for ctx in LARGE_CONTEXTS:
            runs = []
            for seed in SEEDS:
                runs.append(_load_json(results_dir / f"ga_{model}_{ctx}_seed{seed}.json"))
            data[model][ctx] = runs
        data[model]["url"] = [_load_json(results_dir / f"bruteforce_{model}_url_seed0.json")]
    return data


def _pool_unique_by_payload(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hop nhat ca the duy nhat qua cac seed theo payload (uu tien valid=1, bypass=1)."""
    by_payload: dict[str, dict[str, Any]] = {}
    for run in runs:
        for rec in run.get("unique_individuals", []):
            payload = str(rec["payload"])
            prev = by_payload.get(payload)
            if prev is None:
                by_payload[payload] = dict(rec)
                continue
            cur_key = (int(rec["valid"]), int(rec["bypass"]), float(rec.get("fitness", 0.0)))
            prev_key = (int(prev["valid"]), int(prev["bypass"]), float(prev.get("fitness", 0.0)))
            if cur_key > prev_key:
                by_payload[payload] = dict(rec)
    return list(by_payload.values())


def _mean_std_ddof1(values: list[float]) -> tuple[float, float | None]:
    if not values:
        return 0.0, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def _cell_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid_recs = [r for r in records if int(r["valid"]) == 1]
    bypass_recs = [r for r in valid_recs if int(r["bypass"]) == 1]
    n_valid = len(valid_recs)
    n_bypass = len(bypass_recs)
    rate_a = (n_bypass / n_valid) if n_valid > 0 else 0.0
    return {
        "unique_total": len(records),
        "unique_valid": n_valid,
        "unique_bypass": n_bypass,
        "rate_a": round(rate_a, 6),
    }


def build_matrix_6_2(
    model_runs: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, dict[str, Any]]:
    """Tinh Ma tran Bang 6.2 cho ca 3 mo hinh (tap hop nhat + Mean +- Std qua 5 seed)."""
    matrices: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        model_matrix: dict[str, Any] = {}
        for ctx in CONTEXTS:
            runs = model_runs[model][ctx]
            pooled = _pool_unique_by_payload(runs)

            row: dict[str, Any] = {"cells": {}, "total": _cell_stats(pooled)}
            # Mean +- Std theo tung seed cho Tong ngu canh.
            seed_rates = []
            for run in runs:
                s = _cell_stats(run.get("unique_individuals", []))
                if s["unique_valid"] > 0:
                    seed_rates.append(s["rate_a"])
            mean_r, std_r = _mean_std_ddof1(seed_rates)
            row["total"]["seed_rate_a_mean"] = round(mean_r, 6)
            row["total"]["seed_rate_a_std"] = round(std_r, 6) if std_r is not None else None

            for mech in MECHANISMS:
                if mech in NA_CELLS[ctx]:
                    row["cells"][mech] = {"na": True}
                    continue
                mech_pooled = [r for r in pooled if r.get("mechanism") == mech]
                cell = _cell_stats(mech_pooled)
                cell["na"] = False

                mech_seed_rates = []
                for run in runs:
                    sub = [r for r in run.get("unique_individuals", []) if r.get("mechanism") == mech]
                    cs = _cell_stats(sub)
                    if cs["unique_valid"] > 0:
                        mech_seed_rates.append(cs["rate_a"])
                if mech_seed_rates:
                    m_mean, m_std = _mean_std_ddof1(mech_seed_rates)
                    cell["seed_rate_a_mean"] = round(m_mean, 6)
                    cell["seed_rate_a_std"] = round(m_std, 6) if m_std is not None else None
                else:
                    cell["seed_rate_a_mean"] = 0.0
                    cell["seed_rate_a_std"] = None
                row["cells"][mech] = cell

            model_matrix[ctx] = row
        matrices[model] = model_matrix
    return matrices


def build_url_scheme_breakdown(
    model_runs: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, dict[str, Any]]:
    """Tach rieng ngu canh url thanh 2 nhom theo tien to payload: javascript: va data:text/html."""
    breakdown: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        pooled_url = _pool_unique_by_payload(model_runs[model]["url"])
        js_recs = [r for r in pooled_url if str(r["payload"]).startswith("javascript:")]
        data_recs = [r for r in pooled_url if str(r["payload"]).startswith("data:")]
        breakdown[model] = {
            "javascript:": _cell_stats(js_recs),
            "data:text/html": _cell_stats(data_recs),
        }
    return breakdown


def _fmt_cell(cell: dict[str, Any]) -> str:
    if cell.get("na"):
        return "N/A"
    b = int(cell["unique_bypass"])
    v = int(cell["unique_valid"])
    r = float(cell["rate_a"]) * 100.0
    return f"{b}/{v} ({r:.2f}%)"


def print_matrices_6_2(
    matrices: dict[str, dict[str, Any]],
    url_breakdown: dict[str, dict[str, Any]],
) -> None:
    for model in MODELS:
        header = (
            f"{'Ngữ cảnh':11} | {'html_event':20} | {'url_protocol (js:/data:)':24} | "
            f"{'js_direct':20} | {'Tổng (Pooled | Seed Mean±Std)':32}"
        )
        line_len = len(header)
        print("\n" + "=" * line_len)
        print(f"MA TRAN BANG 6.2 — {MODEL_DISPLAY[model]} (unique_bypass / unique_valid, Rate A %)")
        print("=" * line_len)
        print(header)
        print("-" * line_len)
        for ctx in CONTEXTS:
            row = matrices[model][ctx]
            cells = row["cells"]
            tot = row["total"]
            tot_pooled = _fmt_cell(tot)
            if tot["seed_rate_a_std"] is not None:
                tot_seed = f"{tot['seed_rate_a_mean']*100:.2f}%±{tot['seed_rate_a_std']*100:.2f}%"
            else:
                tot_seed = "N/A"
            print(
                f"{ctx:11} | {_fmt_cell(cells['html_event']):20} | "
                f"{_fmt_cell(cells['js_protocol']):24} | "
                f"{_fmt_cell(cells['js_direct']):20} | "
                f"{tot_pooled} | {tot_seed}"
            )
        ub = url_breakdown[model]
        print(
            f"  -> Chi tiet url theo scheme: javascript: = {_fmt_cell(ub['javascript:'])} "
            f"| data:text/html = {_fmt_cell(ub['data:text/html'])}"
        )
        print(
            "  -> Ghi chu: Pooled = gop ca the duy nhat qua 5 seed (phan anh do phu o Ma tran 6.2); "
            "Seed Mean±Std = Bypass Rate chinh tinh trung binh theo 5 seed."
        )


def _extract_svm_linear_weights(svm_model: Any) -> tuple[np.ndarray, float]:
    """Lay trung binh vector coef_ va intercept tu CalibratedClassifierCV(LinearSVC)."""
    if hasattr(svm_model, "calibrated_classifiers_"):
        coefs = [c.estimator.coef_.ravel() for c in svm_model.calibrated_classifiers_]
        intercepts = [float(c.estimator.intercept_.ravel()[0]) for c in svm_model.calibrated_classifiers_]
        return np.mean(coefs, axis=0), float(np.mean(intercepts))
    return svm_model.coef_.ravel(), float(svm_model.intercept_.ravel()[0])


def _structural_comparison(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    detected_payloads = [r["payload"] for r in records if int(r["valid"]) == 1 and int(r["bypass"]) == 0]
    bypassed_payloads = [r["payload"] for r in records if int(r["valid"]) == 1 and int(r["bypass"]) == 1]

    det_feats = extract_structural_features([decode_payload(p) for p in detected_payloads])
    byp_feats = extract_structural_features([decode_payload(p) for p in bypassed_payloads])

    det_means = det_feats.mean(axis=0) if len(det_feats) else np.zeros(len(STRUCTURAL_FEATURE_NAMES))
    byp_means = byp_feats.mean(axis=0) if len(byp_feats) else np.zeros(len(STRUCTURAL_FEATURE_NAMES))

    rows = []
    for idx, fname in enumerate(STRUCTURAL_FEATURE_NAMES):
        d_val = float(det_means[idx])
        b_val = float(byp_means[idx])
        rows.append(
            {
                "feature": fname,
                "detected_mean": round(d_val, 4),
                "bypassed_mean": round(b_val, 4),
                "diff": round(b_val - d_val, 4),
            }
        )
    return {
        "n_detected_valid": len(detected_payloads),
        "n_bypassed_valid": len(bypassed_payloads),
        "features": rows,
    }


def _svm_top_negative_contributors(
    payload: str,
    extractor: Any,
    feature_names: list[str],
    mean_coef: np.ndarray,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    x_csr = extractor.transform([decode_payload(payload)])
    row_dense = np.asarray(x_csr.toarray(), dtype=np.float64).ravel()
    contribs = row_dense * mean_coef
    neg_indices = [i for i in np.argsort(contribs) if contribs[i] < 0 and abs(row_dense[i]) > 1e-12]
    results = []
    for idx in neg_indices[:top_k]:
        results.append(
            {
                "feature": feature_names[idx],
                "value": round(float(row_dense[idx]), 4),
                "coef": round(float(mean_coef[idx]), 4),
                "contribution": round(float(contribs[idx]), 4),
            }
        )
    return results


def _top_counts(records: list[dict[str, Any]], key_fn: Any, top_n: int = 10) -> list[dict[str, Any]]:
    counter = Counter(key_fn(r) for r in records)
    return [{"key": str(k), "count": int(v)} for k, v in counter.most_common(top_n)]


def run_error_analysis(
    model_runs: dict[str, dict[str, list[dict[str, Any]]]],
    artifacts_dir: Path,
) -> dict[str, Any]:
    """Phan tich loi chi tiet cho RF, SVM, CNN+LSTM va giai thich dac trung."""
    extractor = joblib.load(artifacts_dir / "feature_extractor_full.joblib")
    rf_model = joblib.load(artifacts_dir / "rf_best.joblib")
    svm_model = joblib.load(artifacts_dir / "svm_best.joblib")
    feature_names = extractor.get_feature_names()
    svm_coef, svm_intercept = _extract_svm_linear_weights(svm_model)

    # 1. RF (js_string)
    rf_js_pooled = _pool_unique_by_payload(model_runs["rf"]["js_string"])
    rf_js_byp = sorted(
        [r for r in rf_js_pooled if int(r["valid"]) == 1 and int(r["bypass"]) == 1],
        key=lambda r: (float(r["p_malicious"]), str(r["payload"])),
    )

    print("\n" + "=" * 106)
    print(f"PHAN TICH LOI — RF (js_string): {len(rf_js_byp)} payload bypass duy nhat")
    print("=" * 106)
    for i, r in enumerate(rf_js_byp, 1):
        print(
            f"  {i:>2}. p_rf={r['p_malicious']:.4f} | mech={r['mechanism']:10} | "
            f"esc={r['escape']!r:<5} trg={r['trigger']:<8} act={r['action']:<24} "
            f"tag={r['target_tag']:<7} | {r['payload']}"
        )

    # 2. SVM (js_string & url)
    svm_js_pooled = _pool_unique_by_payload(model_runs["svm"]["js_string"])
    svm_js_byp = sorted(
        [r for r in svm_js_pooled if int(r["valid"]) == 1 and int(r["bypass"]) == 1],
        key=lambda r: (float(r["p_malicious"]), str(r["payload"])),
    )
    svm_url_pooled = _pool_unique_by_payload(model_runs["svm"]["url"])
    svm_url_byp = sorted(
        [r for r in svm_url_pooled if int(r["valid"]) == 1 and int(r["bypass"]) == 1],
        key=lambda r: (float(r["p_malicious"]), str(r["payload"])),
    )

    print("\n" + "=" * 106)
    print(
        f"PHAN TICH LOI — SVM: js_string={len(svm_js_byp)} bypass, url={len(svm_url_byp)} bypass"
    )
    print("=" * 106)
    print("  [SVM / js_string]")
    for i, r in enumerate(svm_js_byp, 1):
        print(
            f"    {i:>2}. p_svm={r['p_malicious']:.4f} | esc={r['escape']!r:<5} "
            f"trg={r['trigger']:<8} act={r['action']:<24} tag={r['target_tag']:<7} | {r['payload']}"
        )
    print("  [SVM / url]")
    for i, r in enumerate(svm_url_byp, 1):
        print(
            f"    {i:>2}. p_svm={r['p_malicious']:.4f} | carrier={r['carrier']:<7} "
            f"act={r['action']:<24} | {r['payload']}"
        )

    # 3. CNN+LSTM (s202)
    cnn_stats_by_ctx: dict[str, Any] = {}
    cnn_all_byp: list[dict[str, Any]] = []
    print("\n" + "=" * 106)
    print("PHAN TICH LOI — CNN+LSTM (s202): Thong ke gen gay bypass theo ngu canh")
    print("=" * 106)
    for ctx in CONTEXTS:
        pooled = _pool_unique_by_payload(model_runs["cnn_lstm"][ctx])
        byp = [r for r in pooled if int(r["valid"]) == 1 and int(r["bypass"]) == 1]
        cnn_all_byp.extend(byp)
        ctx_stat = {
            "bypass_count": len(byp),
            "top_triggers": _top_counts(byp, lambda r: r["trigger"], 10),
            "top_actions": _top_counts(byp, lambda r: r["action"], 10),
            "top_target_tags": _top_counts(byp, lambda r: r["target_tag"], 10),
            "top_trigger_tag_pairs": _top_counts(
                byp, lambda r: f"{r['trigger']} + {r['target_tag']}", 10
            ),
        }
        cnn_stats_by_ctx[ctx] = ctx_stat
        print(f"\n  [{ctx}] unique_bypass = {len(byp)}")
        if byp:
            print("    Top trigger        :", ", ".join(f"{x['key']}({x['count']})" for x in ctx_stat["top_triggers"]))
            print("    Top target_tag     :", ", ".join(f"{x['key']}({x['count']})" for x in ctx_stat["top_target_tags"]))
            print("    Top action         :", ", ".join(f"{x['key']}({x['count']})" for x in ctx_stat["top_actions"]))
            print("    Top (trg, tag)     :", ", ".join(f"{x['key']}({x['count']})" for x in ctx_stat["top_trigger_tag_pairs"]))

    cnn_lowest10 = sorted(
        cnn_all_byp, key=lambda r: (float(r["p_malicious"]), str(r["context"]), str(r["payload"]))
    )[:10]
    print("\n  [CNN+LSTM] 10 mau co P_doc thap nhat:")
    for i, r in enumerate(cnn_lowest10, 1):
        print(
            f"    {i:>2}. [{r['context']:9}] p_cnn={r['p_malicious']:.4f} | "
            f"esc={r['escape']!r} trg={r['trigger']} tag={r['target_tag']} | {r['payload']}"
        )

    # 4. Giai thich tu dac trung RF & SVM
    rf_struct_cmp = _structural_comparison(rf_js_pooled)
    svm_url_struct_cmp = _structural_comparison(svm_url_pooled)
    svm_js_struct_cmp = _structural_comparison(svm_js_pooled)

    # Top RF feature importances & struct importances
    rf_imp = rf_model.feature_importances_
    n_tfidf = len(feature_names) - len(STRUCTURAL_FEATURE_NAMES)
    rf_struct_imp = [
        {"feature": STRUCTURAL_FEATURE_NAMES[i], "importance": round(float(rf_imp[n_tfidf + i]), 6)}
        for i in range(len(STRUCTURAL_FEATURE_NAMES))
    ]
    rf_top_overall = [
        {"feature": feature_names[idx], "importance": round(float(rf_imp[idx]), 6)}
        for idx in np.argsort(rf_imp)[::-1][:10]
    ]

    svm_struct_coef = [
        {"feature": STRUCTURAL_FEATURE_NAMES[i], "coef": round(float(svm_coef[n_tfidf + i]), 6)}
        for i in range(len(STRUCTURAL_FEATURE_NAMES))
    ]

    print("\n" + "=" * 106)
    print("GIAI THICH DAC TRUNG CAU TRUC (Detected valid=1 vs Bypassed valid=1)")
    print("=" * 106)
    for title, cmp_data, weight_map, w_label in (
        (
            "RF / js_string",
            rf_struct_cmp,
            {x["feature"]: x["importance"] for x in rf_struct_imp},
            "rf_imp",
        ),
        (
            "SVM / url",
            svm_url_struct_cmp,
            {x["feature"]: x["coef"] for x in svm_struct_coef},
            "svm_coef",
        ),
        (
            "SVM / js_string",
            svm_js_struct_cmp,
            {x["feature"]: x["coef"] for x in svm_struct_coef},
            "svm_coef",
        ),
    ):
        print(
            f"\n  [{title}] detected={cmp_data['n_detected_valid']}, "
            f"bypassed={cmp_data['n_bypassed_valid']}"
        )
        print(f"    {'feature':24} | {'detected':>10} | {'bypassed':>10} | {'diff':>10} | {w_label:>10}")
        print("    " + "-" * 74)
        for row in cmp_data["features"]:
            w = weight_map[row["feature"]]
            print(
                f"    {row['feature']:24} | {row['detected_mean']:>10.4f} | "
                f"{row['bypassed_mean']:>10.4f} | {row['diff']:>+10.4f} | {w:>10.5f}"
            )

    print("\n  [RF Top 10 feature_importances_ toan cuc]:")
    for x in rf_top_overall:
        print(f"    - {x['feature']:24}: {x['importance']:.6f}")

    # Top 5 dong gop am manh nhat cua SVM cho mau tieu bieu o url va js_string
    svm_examples: list[dict[str, Any]] = []
    print(f"\n  [SVM (intercept={svm_intercept:.4f}) — Top 5 dac trung dong gop am cho mau bypass tieu bieu]:")
    for label, sample_list in (("svm/url", svm_url_byp), ("svm/js_string", svm_js_byp)):
        if not sample_list:
            continue
        rep = sample_list[0]
        top_neg = _svm_top_negative_contributors(rep["payload"], extractor, feature_names, svm_coef, top_k=5)
        svm_examples.append(
            {
                "group": label,
                "payload": rep["payload"],
                "p_svm": rep["p_malicious"],
                "top_negative_features": top_neg,
            }
        )
        print(f"    * {label}: p_svm={rep['p_malicious']:.4f} | payload={rep['payload']}")
        for item in top_neg:
            print(
                f"        {item['feature']:24} | val={item['value']:>7.4f} | "
                f"coef={item['coef']:>+8.4f} | contrib={item['contribution']:>+8.4f}"
            )

    return {
        "rf_js_string_bypass": rf_js_byp,
        "svm_js_string_bypass": svm_js_byp,
        "svm_url_bypass": svm_url_byp,
        "cnn_lstm_by_context": cnn_stats_by_ctx,
        "cnn_lstm_lowest10": cnn_lowest10,
        "structural_comparison": {
            "rf_js_string": rf_struct_cmp,
            "svm_url": svm_url_struct_cmp,
            "svm_js_string": svm_js_struct_cmp,
        },
        "rf_structural_importances": rf_struct_imp,
        "rf_top10_importances": rf_top_overall,
        "svm_structural_coefs": svm_struct_coef,
        "svm_intercept": round(svm_intercept, 6),
        "svm_negative_examples": svm_examples,
    }


def export_transferability_and_csv(
    model_runs: dict[str, dict[str, list[dict[str, Any]]]],
    out_csv_path: Path,
) -> dict[str, Any]:
    """Gom toan bo mau valid=1 & bypass=1 tu 3 mo hinh, danh gia Cross-Model va xuat CSV."""
    dedup: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for model in MODELS:
        for ctx in CONTEXTS:
            for run in model_runs[model][ctx]:
                for rec in run.get("unique_individuals", []):
                    if int(rec["valid"]) != 1 or int(rec["bypass"]) != 1:
                        continue
                    key = (
                        str(rec["context"]),
                        str(rec["family"]),
                        str(rec["event"]),
                        str(rec["carrier"]),
                        str(rec["payload"]),
                    )
                    if key not in dedup:
                        dedup[key] = {
                            "context": str(rec["context"]),
                            "mechanism": str(rec["mechanism"]),
                            "family": str(rec["family"]),
                            "event": str(rec["event"]),
                            "carrier": str(rec["carrier"]),
                            "escape": str(rec["escape"]),
                            "trigger": str(rec["trigger"]),
                            "action": str(rec["action"]),
                            "target_tag": str(rec["target_tag"]),
                            "payload": str(rec["payload"]),
                            "valid": 1,
                        }

    records = list(dedup.values())
    payloads = [r["payload"] for r in records]

    predictors = {m: ModelPredictor(model_name=m) for m in MODELS}
    probs = {m: predictors[m].predict_proba(payloads) for m in MODELS}
    thresholds = {m: float(predictors[m].threshold) for m in MODELS}

    enriched: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        p_rf = float(probs["rf"][i])
        p_svm = float(probs["svm"][i])
        p_cnn = float(probs["cnn_lstm"][i])

        b_rf = int(p_rf < thresholds["rf"])
        b_svm = int(p_svm < thresholds["svm"])
        b_cnn = int(p_cnn < thresholds["cnn_lstm"])

        bypassed_models = [
            m for m, flag in (("rf", b_rf), ("svm", b_svm), ("cnn_lstm", b_cnn)) if flag == 1
        ]
        enriched.append(
            {
                **rec,
                "p_rf": round(p_rf, 6),
                "p_svm": round(p_svm, 6),
                "p_cnn_lstm": round(p_cnn, 6),
                "bypass_rf": b_rf,
                "bypass_svm": b_svm,
                "bypass_cnn_lstm": b_cnn,
                "models_bypassed": ";".join(bypassed_models),
                "num_models_bypassed": len(bypassed_models),
            }
        )

    # Sap xep theo (context, -num_models_bypassed, payload) truoc khi danh ID.
    enriched.sort(key=lambda r: (r["context"], -r["num_models_bypassed"], r["payload"]))

    final_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(enriched, 1):
        ordered = {
            "id": f"GA_BYP_{idx:04d}",
            "context": row["context"],
            "mechanism": row["mechanism"],
            "family": row["family"],
            "event": row["event"],
            "carrier": row["carrier"],
            "escape": row["escape"],
            "trigger": row["trigger"],
            "action": row["action"],
            "target_tag": row["target_tag"],
            "payload": row["payload"],
            "valid": row["valid"],
            "p_rf": row["p_rf"],
            "p_svm": row["p_svm"],
            "p_cnn_lstm": row["p_cnn_lstm"],
            "bypass_rf": row["bypass_rf"],
            "bypass_svm": row["bypass_svm"],
            "bypass_cnn_lstm": row["bypass_cnn_lstm"],
            "models_bypassed": row["models_bypassed"],
            "num_models_bypassed": row["num_models_bypassed"],
        }
        final_rows.append(ordered)

    # Assert bat buoc truoc khi xuat.
    assert len(final_rows) > 0, "Khong co payload bypass nao de xuat!"
    for r in final_rows:
        assert r["valid"] == 1, f"Dong khong valid=1: {r['id']}"
        assert r["num_models_bypassed"] >= 1, f"num_models_bypassed < 1 tai {r['id']}"
        assert "__markXSS" not in r["payload"], f"Phat hien __markXSS tai {r['id']}"
        for k, v in r.items():
            assert v is not None, f"Gia tri None tai {r['id']}.{k}"
            if isinstance(v, float):
                assert not math.isnan(v), f"Gia tri NaN tai {r['id']}.{k}"

    fieldnames = list(final_rows[0].keys())
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(final_rows)

    # Thong ke Transferability
    combo_counts = Counter(r["models_bypassed"] for r in final_rows)
    num_counts = Counter(r["num_models_bypassed"] for r in final_rows)
    by_context: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in final_rows:
        by_context[r["context"]]["total"] += 1
        by_context[r["context"]][r["models_bypassed"]] += 1

    pairwise = {
        "rf_and_svm": sum(1 for r in final_rows if r["bypass_rf"] and r["bypass_svm"]),
        "rf_and_cnn_lstm": sum(1 for r in final_rows if r["bypass_rf"] and r["bypass_cnn_lstm"]),
        "svm_and_cnn_lstm": sum(1 for r in final_rows if r["bypass_svm"] and r["bypass_cnn_lstm"]),
        "all_three": sum(
            1 for r in final_rows if r["bypass_rf"] and r["bypass_svm"] and r["bypass_cnn_lstm"]
        ),
    }

    print("\n" + "=" * 106)
    print(f"CROSS-MODEL TRANSFERABILITY (Tong so mau duy nhat xuat CSV: {len(final_rows)})")
    print("=" * 106)
    print(f"  Nguong mo hinh su dung: {thresholds}")
    print(f"  Theo so mo hinh bi bypass: 1={num_counts[1]}, 2={num_counts[2]}, 3={num_counts[3]}")
    print(f"  Giao thoa cap & bo ba   : {pairwise}")
    print("  Phan bo theo to hop models_bypassed:")
    for combo, cnt in combo_counts.most_common():
        print(f"    - {combo:20}: {cnt}")
    print("  Phan bo theo ngu canh:")
    for ctx in CONTEXTS:
        if ctx in by_context:
            print(f"    - {ctx:10}: {dict(by_context[ctx])}")
    print(f"\nDa xuat CSV tai: {out_csv_path}")

    return {
        "total_unique_exported": len(final_rows),
        "thresholds": thresholds,
        "by_num_models_bypassed": {str(k): int(v) for k, v in sorted(num_counts.items())},
        "by_models_bypassed_combo": dict(combo_counts.most_common()),
        "pairwise_and_triple": pairwise,
        "by_context": {ctx: dict(by_context[ctx]) for ctx in CONTEXTS if ctx in by_context},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phan tich ket qua Giai doan 3 (Buoc 3.4)")
    parser.add_argument("--results-dir", default="ga/results_full")
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    parser.add_argument("--out-csv", default="ga_bypassed_payloads.csv")
    parser.add_argument("--out-report", default="ga/results_full/phase3_analysis_report.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    artifacts_dir = Path(args.artifacts_dir)
    out_csv_path = Path(args.out_csv)
    out_report_path = Path(args.out_report)

    model_runs = _load_model_runs(results_dir)
    matrices = build_matrix_6_2(model_runs)
    url_breakdown = build_url_scheme_breakdown(model_runs)
    print_matrices_6_2(matrices, url_breakdown)

    error_analysis = run_error_analysis(model_runs, artifacts_dir)
    transferability = export_transferability_and_csv(model_runs, out_csv_path)

    report = {
        "matrix_6_2": matrices,
        "url_scheme_breakdown": url_breakdown,
        "error_analysis": error_analysis,
        "transferability": transferability,
    }
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    with out_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Da luu bao cao phan tich tai: {out_report_path}")


if __name__ == "__main__":
    main()
