"""Independent read-only Phase 2 audit script (Margin, Action-Swap, OOD Stress Test).

Run from ``xss_pipeline/``:
    ./venv/bin/python -m models.audit_phase2
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd

from models._utils import DEFAULT_DATASET_CSV
from models.predictor import ModelPredictor

OOD_XSS_CASES: tuple[str, ...] = (
    "<details open ontoggle=confirm(1)>",
    "<svg><animate onbegin=prompt(1) attributeName=x>",
    "'-print()-'",
    "javascript:confirm(origin)",
    "<math><mtext><img src=1 onerror=print()>",
    "<body onpageshow=prompt(1)>",
    '<iframe srcdoc="<script>top.print()</script>">',
    '" autofocus onfocus=confirm(1) x="',
    "';window['pro'+'mpt'](1);//",
    "<video><source onerror=confirm(1)>",
    '<a href="javascript:prompt(1)">click</a>',
    "<input onfocus=print() autofocus>",
    "</script><svg onload=confirm(1)>",
    "<img src=x onerror=fetch('/probe')>",
    "javascript:/*comment*/confirm(1)",
)

OOD_BENIGN_CASES: tuple[str, ...] = (
    '<div class="alert-box">Check status</div>',
    "const onError = (e) => console.log(e);",
    "function confirmDelete(id) { return id > 0; }",
    "See https://developer.mozilla.org/en-US/docs/Web/API/Window/alert",
    "SELECT * FROM users WHERE name = 'O''Brien' AND active = 1;",
    r"/^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$/i",
    '<img src="/assets/logo.png" alt="Company logo" width="120">',
    'if (x < 10 && y > 20) { print("ok"); }',
    'curl -s "https://api.example.com/v1/items?a=1&b=2" | jq \'.data\'',
    "Note: avoid using javascript: links in markdown docs",
    "Error: script failed with exit code 1 (see log.txt)",
    "window.addEventListener('load', initApp);",
    '<svg viewBox="0 0 24 24"><path d="M12 2L2 22h20z"/></svg>',
    '// TODO: fix quote escaping for "user\'s input"',
    '{"event": "onload", "status": "ok", "count": 42}',
)


def _display_name(predictor: ModelPredictor) -> str:
    if predictor.model_name == "cnn_lstm":
        return f"CNN+LSTM (s{predictor.seed})"
    return predictor.model_name.upper()


def run_margin_audit(
    df: pd.DataFrame,
    predictors: list[ModelPredictor],
) -> None:
    """Part 1: Probability Margin Audit on val and test splits."""
    print("\n" + "=" * 96)
    print("PART 1: PROBABILITY MARGIN AUDIT (val & test)")
    print("=" * 96)
    print(
        f"{'Model':<16} {'Split':<5} {'Thr':>5} | "
        f"{'Benign p50/p95/max':<23} | "
        f"{'Mal min/p05/p50':<23} | "
        f"{'Uncert[0.2,0.8]':<16} {'MinMargin':>9}"
    )
    print("-" * 110)

    for split_name in ("val", "test"):
        split_df = df[df["split"] == split_name]
        payloads = split_df["payload"].tolist()
        y_true = split_df["label"].to_numpy(dtype=np.int64)
        n_total = len(y_true)

        for predictor in predictors:
            probs = predictor.predict_proba(payloads)
            thr = predictor.threshold

            benign_p = probs[y_true == 0]
            mal_p = probs[y_true == 1]

            b_p50, b_p95 = np.percentile(benign_p, [50, 95])
            b_max = float(np.max(benign_p))

            m_min = float(np.min(mal_p))
            m_p05, m_p50 = np.percentile(mal_p, [5, 50])

            uncert_count = int(np.sum((probs >= 0.20) & (probs <= 0.80)))
            uncert_pct = 100.0 * uncert_count / max(n_total, 1)
            min_margin = float(np.min(np.abs(probs - thr)))

            b_str = f"{b_p50:.3f}/{b_p95:.3f}/{b_max:.3f}"
            m_str = f"{m_min:.3f}/{m_p05:.3f}/{m_p50:.3f}"
            u_str = f"{uncert_count:>3}/{n_total} ({uncert_pct:4.1f}%)"

            print(
                f"{_display_name(predictor):<16} {split_name:<5} {thr:>5.2f} | "
                f"{b_str:<23} | {m_str:<23} | {u_str:<16} {min_margin:>9.4f}"
            )


def _swap_alert_to(target_fn: str) -> Callable[[str], str]:
    pattern = re.compile(r"alert\(", flags=re.IGNORECASE)
    return lambda s: pattern.sub(f"{target_fn}(", s)


def _swap_fingerprint(s: str) -> str:
    out = re.sub(r"attacker\.test", "cdn-static.example.org", s, flags=re.IGNORECASE)
    out = re.sub(r"document\.cookie", "window.name", out, flags=re.IGNORECASE)
    out = re.sub(r"//\s*$", "", out)
    return out


def run_action_swap_audit(
    df: pd.DataFrame,
    predictors: list[ModelPredictor],
) -> None:
    """Part 2: Action-Swap Robustness on malicious val samples (n=195)."""
    print("\n" + "=" * 96)
    print("PART 2: ACTION-SWAP ROBUSTNESS (split == 'val', label == 1)")
    print("=" * 96)

    val_mal_df = df[(df["split"] == "val") & (df["label"] == 1)]
    orig_payloads = val_mal_df["payload"].tolist()
    print(f"Malicious val samples: n={len(orig_payloads)}")

    orig_preds_by_model = {
        _display_name(p): p.predict(orig_payloads) for p in predictors
    }

    transforms: list[tuple[str, Callable[[str], str]]] = [
        ("1. alert( -> confirm(", _swap_alert_to("confirm")),
        ("2. alert( -> prompt(", _swap_alert_to("prompt")),
        ("3. alert( -> print(", _swap_alert_to("print")),
        ("4. domain/cookie/trailing-//", _swap_fingerprint),
    ]

    print(
        f"{'Transform':<29} {'n_aff':>5} | "
        f"{'Model':<16} {'R_before':>9} {'R_after':>9} {'Delta':>9}"
    )
    print("-" * 88)

    for tx_name, tx_fn in transforms:
        swapped_payloads = [tx_fn(s) for s in orig_payloads]
        affected_mask = np.array(
            [sw != orig for sw, orig in zip(swapped_payloads, orig_payloads)],
            dtype=bool,
        )
        n_affected = int(np.sum(affected_mask))
        affected_swapped = [
            sw for sw, is_aff in zip(swapped_payloads, affected_mask.tolist()) if is_aff
        ]

        for idx, predictor in enumerate(predictors):
            m_name = _display_name(predictor)
            if n_affected == 0:
                r_before = 0.0
                r_after = 0.0
            else:
                r_before = float(np.mean(orig_preds_by_model[m_name][affected_mask]))
                r_after = float(np.mean(predictor.predict(affected_swapped)))
            delta = r_after - r_before

            tx_col = tx_name if idx == 0 else ""
            n_col = f"{n_affected}" if idx == 0 else ""
            print(
                f"{tx_col:<29} {n_col:>5} | "
                f"{m_name:<16} {r_before:>9.4f} {r_after:>9.4f} {delta:>+9.4f}"
            )
        print("-" * 88)


def run_ood_stress_test(predictors: list[ModelPredictor]) -> None:
    """Part 3: Out-of-Distribution Stress Test on 30 hand-crafted cases (15 XSS + 15 Benign)."""
    print("\n" + "=" * 96)
    print("PART 3: OUT-OF-DISTRIBUTION (OOD) STRESS TEST (15 XSS + 15 Benign)")
    print("=" * 96)

    assert len(OOD_XSS_CASES) == 15
    assert len(OOD_BENIGN_CASES) == 15

    all_cases = list(OOD_XSS_CASES) + list(OOD_BENIGN_CASES)
    y_true = np.array([1] * 15 + [0] * 15, dtype=np.int64)

    print(
        f"{'Model':<16} {'Thr':>5} | "
        f"{'OOD XSS Recall':<18} | {'OOD Benign FPR':<18}"
    )
    print("-" * 66)

    per_model_outputs: list[tuple[ModelPredictor, np.ndarray, np.ndarray]] = []
    for predictor in predictors:
        probs = predictor.predict_proba(all_cases)
        preds = (probs >= predictor.threshold).astype(np.int64)
        per_model_outputs.append((predictor, probs, preds))

        tp = int(np.sum((y_true == 1) & (preds == 1)))
        fp = int(np.sum((y_true == 0) & (preds == 1)))
        recall_str = f"{tp}/15 ({tp / 15.0:.4f})"
        fpr_str = f"{fp}/15 ({fp / 15.0:.4f})"

        print(
            f"{_display_name(predictor):<16} {predictor.threshold:>5.2f} | "
            f"{recall_str:<18} | {fpr_str:<18}"
        )

    print("\n--- Detailed OOD Misclassifications (FN / FP) ---")
    for predictor, probs, preds in per_model_outputs:
        m_name = _display_name(predictor)
        errors = [
            (i, int(y_true[i]), int(preds[i]), float(probs[i]), all_cases[i])
            for i in range(len(all_cases))
            if preds[i] != y_true[i]
        ]
        print(f"\n[{m_name}] (thr={predictor.threshold:.2f}) — Total errors: {len(errors)}/30")
        if not errors:
            print("  (None — 30/30 correct)")
            continue
        for idx, yt, yp, prob, payload in errors:
            err_type = "FN (missed XSS)" if (yt == 1 and yp == 0) else "FP (benign flagged)"
            case_num = (idx + 1) if yt == 1 else (idx - 15 + 1)
            print(
                f"  - {err_type:<20} | case #{case_num:02d} | "
                f"p_malicious={prob:.4f} | payload={payload!r}"
            )


def main() -> None:
    df = pd.read_csv(DEFAULT_DATASET_CSV)
    for col in ("payload", "label", "split"):
        assert not df[col].isna().any(), f"Unexpected NaN in column '{col}'."

    predictors = [
        ModelPredictor("rf"),
        ModelPredictor("svm"),
        ModelPredictor("cnn_lstm"),
    ]

    run_margin_audit(df, predictors)
    run_action_swap_audit(df, predictors)
    run_ood_stress_test(predictors)


if __name__ == "__main__":
    main()
