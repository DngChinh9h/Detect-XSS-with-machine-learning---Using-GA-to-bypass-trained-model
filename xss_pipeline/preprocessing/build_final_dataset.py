"""Merge malicious + benign datasets and build stratified group splits for Project T02.

Workflow:
1. Load ``oracle_results_2.csv``, filter ``valid == 1``, and deduplicate via
   :func:`preprocessing.decoder.deduplicate_malicious_df` (preferring ``Tainted``
   ahead of ``Obfuscated``).
2. Load ``benign_dataset_4000.csv``, recompute/verify ``payload_decoded`` via
   :func:`preprocessing.decoder.decode_payload`, and standardize IDs
   (``BEN_CLN_0001``..., ``BEN_UNU_0001``...).
3. Merge both sources into the unified 10-column schema:
   ``['id', 'payload', 'payload_decoded', 'label', 'group', 'context',
      'family', 'split', 'obfuscation', 'source']``
4. Drop any existing ``split`` column and re-assign ``split`` via
   **Constrained Multi-trial Group Split** (``N_TRIALS = 5000``, ``SEED = 42``,
   targets ``train=0.70``, ``val=0.10``, ``test=0.20``):
   - Strata: ``attribute``, ``html_body``, ``js_string``, ``url`` (malicious)
     and ``Clean``, ``Unusual`` (benign).
   - Hard constraints per stratum with ``F`` families:
     ``val_families >= max(3, ceil(0.10 * F))``,
     ``test_families >= max(4, ceil(0.20 * F))``,
     and no single family > 50% of ``val`` samples (when feasible).
   - Objective: minimize
     ``abs(r_train - 0.70) + 1.5 * abs(r_val - 0.10) + 1.2 * abs(r_test - 0.20)``.
5. Run mandatory assertions (zero family overlap across splits, minimum family
   diversity per stratum, zero decoded duplicates globally, zero NaNs in core
   columns), save ``dataset_final.csv``, and print the full 6-part Audit Report.

Run from ``xss_pipeline/``:
    ./venv/bin/python -m preprocessing.build_final_dataset
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import pandas as pd

from preprocessing.build_benign_dataset import (
    DEFAULT_ORACLE_CSV,
    DEFAULT_OUTPUT_CSV as DEFAULT_BENIGN_CSV,
    PIPELINE_DIR,
    SEED,
    compute_length_percentiles,
    load_dedup_malicious,
)
from preprocessing.decoder import decode_payload

DEFAULT_FINAL_CSV = PIPELINE_DIR / "dataset_final.csv"

FINAL_SCHEMA = [
    "id",
    "payload",
    "payload_decoded",
    "label",
    "group",
    "context",
    "family",
    "split",
    "obfuscation",
    "source",
]

CORE_NON_NULL_COLUMNS = [
    "id",
    "payload",
    "payload_decoded",
    "label",
    "group",
    "context",
    "family",
    "split",
    "obfuscation",
    "source",
]

SPLIT_ORDER = ("train", "val", "test")
TARGET_RATIOS: dict[str, float] = {
    "train": 0.70,
    "val": 0.10,
    "test": 0.20,
}
N_TRIALS = 5000

MALICIOUS_STRATA = ("attribute", "html_body", "js_string", "url")
BENIGN_STRATA = ("Clean", "Unusual")
ALL_STRATA = MALICIOUS_STRATA + BENIGN_STRATA

SENSITIVE_KEYWORDS = [
    "alert",
    "confirm",
    "prompt",
    "onerror",
    "onload",
    "onclick",
    "onfocus",
    "document.cookie",
    "javascript:",
    "script",
    "fetch",
    "iframe",
    "svg",
    "attacker.test",
]


def _assign_stratum_key(df: pd.DataFrame) -> pd.Series:
    """Return stratification key: ``context`` for label==1, ``group`` for label==0."""
    strata: list[str] = []
    for label, group, context in zip(
        df["label"].tolist(), df["group"].tolist(), df["context"].tolist()
    ):
        if int(label) == 1:
            strata.append(str(context))
        elif int(label) == 0:
            strata.append(str(group))
        else:
            raise ValueError(f"Unexpected label value: {label!r}")
    return pd.Series(strata, index=df.index, dtype="object")


def prepare_merged_dataframe(
    oracle_csv: Path,
    benign_csv: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load, deduplicate, align schema, and concatenate malicious and benign sets."""
    mal_df, mal_stats = load_dedup_malicious(oracle_csv)
    ben_df = pd.read_csv(benign_csv, keep_default_na=False)

    # Recompute payload_decoded for benign via Single Source of Truth decoder.
    ben_df["payload_decoded"] = [decode_payload(val) for val in ben_df["payload"].tolist()]

    # Standardize benign IDs if missing or ensure expected prefixes.
    clean_mask = ben_df["group"] == "Clean"
    unusual_mask = ben_df["group"] == "Unusual"
    if "id" not in ben_df.columns or ben_df["id"].astype(str).str.strip().eq("").any():
        ben_df["id"] = ""
        ben_df.loc[clean_mask, "id"] = [
            f"BEN_CLN_{i:04d}" for i in range(1, int(clean_mask.sum()) + 1)
        ]
        ben_df.loc[unusual_mask, "id"] = [
            f"BEN_UNU_{i:04d}" for i in range(1, int(unusual_mask.sum()) + 1)
        ]

    # Ensure label types and drop any pre-existing split column before re-splitting.
    mal_df = mal_df.copy()
    ben_df = ben_df.copy()
    mal_df["label"] = 1
    ben_df["label"] = 0

    if "split" in mal_df.columns:
        mal_df = mal_df.drop(columns=["split"])
    if "split" in ben_df.columns:
        ben_df = ben_df.drop(columns=["split"])

    base_cols = [c for c in FINAL_SCHEMA if c != "split"]
    missing_mal = [c for c in base_cols if c not in mal_df.columns]
    missing_ben = [c for c in base_cols if c not in ben_df.columns]
    if missing_mal:
        raise ValueError(f"Malicious DataFrame missing columns: {missing_mal}")
    if missing_ben:
        raise ValueError(f"Benign DataFrame missing columns: {missing_ben}")

    merged = pd.concat(
        [mal_df[base_cols], ben_df[base_cols]],
        ignore_index=True,
    )
    merged["label"] = merged["label"].astype(int)
    for col in base_cols:
        if col != "label":
            merged[col] = merged[col].astype(str)

    return merged, mal_stats


def _solve_stratum_multitrial(
    fam_counts: dict[str, int],
    target_ratios: dict[str, float],
    seed: int,
    n_trials: int = N_TRIALS,
) -> tuple[dict[str, str], dict[str, float]]:
    """Find the best family->split assignment for one stratum via constrained multi-trial search."""
    items_base: list[tuple[str, int]] = sorted(fam_counts.items(), key=lambda x: x[0])
    num_fams = len(items_base)
    total_n = sum(sz for _, sz in items_base)
    if total_n == 0 or num_fams == 0:
        return {}, {"strict_val_cap": 1.0, "max_val_share": 0.0, "loss": 0.0}

    r_tr_target = target_ratios["train"]
    r_va_target = target_ratios["val"]
    r_te_target = target_ratios["test"]
    t_val = r_va_target * total_n
    t_test = r_te_target * total_n

    min_val_fams = max(3, math.ceil(r_va_target * num_fams))
    min_test_fams = max(4, math.ceil(r_te_target * num_fams))
    if min_val_fams + min_test_fams >= num_fams:
        raise ValueError(
            f"Stratum has too few families ({num_fams}) to satisfy "
            f"min_val_fams={min_val_fams} and min_test_fams={min_test_fams}."
        )

    val_single_cap = max(1, int(math.floor(0.50 * t_val * 1.1)))
    test_single_cap = max(1, int(math.floor(0.55 * t_test)))

    rng = random.Random(seed)
    best_strict: tuple[
        tuple[float, float, float, int],
        float,
        float,
        list[tuple[str, int]],
        list[tuple[str, int]],
        list[tuple[str, int]],
    ] | None = None
    best_relaxed: tuple[
        tuple[float, float, float, int],
        float,
        float,
        list[tuple[str, int]],
        list[tuple[str, int]],
        list[tuple[str, int]],
    ] | None = None

    for trial in range(n_trials):
        mode = trial % 5
        if mode in (0, 1, 2):
            order = list(items_base)
            rng.shuffle(order)
        elif mode == 3:
            order = sorted(items_base, key=lambda x: x[1] * rng.uniform(0.5, 2.5))
        else:
            order = sorted(items_base, key=lambda x: x[1] * rng.uniform(0.25, 1.8))

        # 1. Assign small/medium families to val until min_val_fams and close to t_val
        val_set: list[tuple[str, int]] = []
        val_sum = 0
        rem1: list[tuple[str, int]] = []
        for item in order:
            _fam, sz = item
            if len(val_set) < min_val_fams:
                if sz <= val_single_cap and (val_sum + sz <= t_val * 1.25 or not val_set):
                    val_set.append(item)
                    val_sum += sz
                else:
                    rem1.append(item)
            else:
                if sz <= val_single_cap and abs((val_sum + sz) - t_val) < abs(val_sum - t_val):
                    val_set.append(item)
                    val_sum += sz
                else:
                    rem1.append(item)

        if len(val_set) < min_val_fams:
            rem1_sorted = sorted(rem1, key=lambda x: (x[1], rng.random()))
            need = min_val_fams - len(val_set)
            val_set.extend(rem1_sorted[:need])
            val_sum += sum(x[1] for x in rem1_sorted[:need])
            rem1 = rem1_sorted[need:]

        # 2. Assign small/medium families from remaining pool to test
        if mode in (0, 1, 2):
            rem1_ord = list(rem1)
            rng.shuffle(rem1_ord)
        else:
            rem1_ord = sorted(rem1, key=lambda x: x[1] * rng.uniform(0.4, 2.2))

        test_set: list[tuple[str, int]] = []
        test_sum = 0
        train_set: list[tuple[str, int]] = []
        for item in rem1_ord:
            _fam, sz = item
            if len(test_set) < min_test_fams:
                if sz <= test_single_cap and (test_sum + sz <= t_test * 1.25 or not test_set):
                    test_set.append(item)
                    test_sum += sz
                else:
                    train_set.append(item)
            else:
                if abs((test_sum + sz) - t_test) < abs(test_sum - t_test):
                    test_set.append(item)
                    test_sum += sz
                else:
                    train_set.append(item)

        if len(test_set) < min_test_fams:
            tr_sorted = sorted(train_set, key=lambda x: (x[1], rng.random()))
            need = min_test_fams - len(test_set)
            test_set.extend(tr_sorted[:need])
            test_sum += sum(x[1] for x in tr_sorted[:need])
            train_set = tr_sorted[need:]

        # 3. Remaining families go to train
        train_sum = total_n - val_sum - test_sum
        if not train_set or len(val_set) < min_val_fams or len(test_set) < min_test_fams:
            continue

        r_tr = train_sum / total_n
        r_va = val_sum / total_n
        r_te = test_sum / total_n
        loss = (
            abs(r_tr - r_tr_target)
            + 1.5 * abs(r_va - r_va_target)
            + 1.2 * abs(r_te - r_te_target)
        )
        max_val_share = max(x[1] for x in val_set) / val_sum if val_sum else 1.0
        fam_err = (
            abs(len(train_set) / num_fams - r_tr_target)
            + abs(len(val_set) / num_fams - r_va_target)
            + abs(len(test_set) / num_fams - r_te_target)
        )
        score_key = (round(loss, 12), round(fam_err, 12), round(max_val_share, 12), trial)
        cand = (score_key, loss, max_val_share, train_set, val_set, test_set)

        if max_val_share <= 0.50 + 1e-12:
            if best_strict is None or cand[0] < best_strict[0]:
                best_strict = cand
        if best_relaxed is None or cand[0] < best_relaxed[0]:
            best_relaxed = cand

    chosen = best_strict if best_strict is not None else best_relaxed
    if chosen is None:
        raise RuntimeError("Constrained multi-trial split failed to find a valid partition.")

    _, best_loss, best_max_val_share, tr_list, va_list, te_list = chosen
    mapping: dict[str, str] = {}
    for fam, _sz in tr_list:
        mapping[fam] = "train"
    for fam, _sz in va_list:
        mapping[fam] = "val"
    for fam, _sz in te_list:
        mapping[fam] = "test"

    meta = {
        "strict_val_cap": 1.0 if best_strict is not None else 0.0,
        "max_val_share": best_max_val_share,
        "loss": best_loss,
    }
    return mapping, meta


def constrained_multitrial_group_split(
    df: pd.DataFrame,
    target_ratios: dict[str, float] = TARGET_RATIOS,
    seed: int = SEED,
    n_trials: int = N_TRIALS,
) -> pd.DataFrame:
    """Assign train/val/test splits using Constrained Multi-trial Group Split.

    Within each stratum (malicious ``context`` or benign ``group``):
    - Runs ``n_trials`` (default 5,000) deterministic seeded trials.
    - Enforces minimum family diversity:
      ``val_families >= max(3, ceil(0.10 * F))``,
      ``test_families >= max(4, ceil(0.20 * F))``.
    - Prevents any single family from exceeding 50% of ``val`` samples when feasible.
    - Minimizes ``abs(r_train - 0.70) + 1.5 * abs(r_val - 0.10) + 1.2 * abs(r_test - 0.20)``.
    """
    work = df.copy()
    if "split" in work.columns:
        work = work.drop(columns=["split"])

    stratum_series = _assign_stratum_key(work)
    observed_strata = set(stratum_series.unique().tolist())
    unexpected = observed_strata - set(ALL_STRATA)
    if unexpected:
        raise ValueError(f"Unexpected strata found: {sorted(unexpected)}")

    # Ensure no family spans multiple strata before grouping.
    fam_stratum_counts = (
        pd.DataFrame({"family": work["family"], "stratum": stratum_series})
        .groupby("family")["stratum"]
        .nunique()
    )
    cross_stratum = fam_stratum_counts[fam_stratum_counts > 1]
    assert cross_stratum.empty, (
        f"Found families spanning multiple strata: {cross_stratum.to_dict()}"
    )

    family_to_split: dict[str, str] = {}
    for stratum_idx, stratum in enumerate(ALL_STRATA):
        mask = stratum_series == stratum
        sub = work.loc[mask]
        if len(sub) == 0:
            continue
        fam_counts = {
            str(fam): int(cnt) for fam, cnt in sub["family"].value_counts().items()
        }
        stratum_seed = seed + stratum_idx * 10007
        stratum_map, _meta = _solve_stratum_multitrial(
            fam_counts=fam_counts,
            target_ratios=target_ratios,
            seed=stratum_seed,
            n_trials=n_trials,
        )
        family_to_split.update(stratum_map)

    work["split"] = work["family"].map(family_to_split)
    return work[FINAL_SCHEMA].reset_index(drop=True)


def run_mandatory_assertions(df: pd.DataFrame) -> None:
    """Enforce strict data-integrity and zero-leakage invariants before export."""
    # 1. Schema check
    assert list(df.columns) == FINAL_SCHEMA, (
        f"Schema mismatch: expected {FINAL_SCHEMA}, got {list(df.columns)}"
    )

    # 2. Zero NaN & zero blank strings in core columns
    for col in CORE_NON_NULL_COLUMNS:
        na_cnt = int(df[col].isna().sum())
        assert na_cnt == 0, f"Found {na_cnt} NaN values in column '{col}'!"
        if col != "label":
            empty_cnt = int(df[col].astype(str).str.strip().eq("").sum())
            assert empty_cnt == 0, f"Found {empty_cnt} empty strings in column '{col}'!"

    # 3. Valid labels, unique IDs, valid splits
    assert set(df["label"].unique()).issubset({0, 1}), (
        f"Invalid label values: {df['label'].unique().tolist()}"
    )
    assert df["id"].nunique() == len(df), (
        f"Duplicate IDs found: {len(df) - df['id'].nunique()}"
    )
    assert set(df["split"].unique()) == set(SPLIT_ORDER), (
        f"Missing or invalid split values: {df['split'].unique().tolist()}"
    )

    # 4. Zero payload_decoded duplicates globally
    n_unique_decoded = int(df["payload_decoded"].nunique())
    assert n_unique_decoded == len(df), (
        f"Found {len(df) - n_unique_decoded} duplicate payload_decoded rows in dataset_final.csv!"
    )

    # 5. Zero family overlap across train / val / test
    train_fams = set(df.loc[df["split"] == "train", "family"])
    val_fams = set(df.loc[df["split"] == "val", "family"])
    test_fams = set(df.loc[df["split"] == "test", "family"])

    tv_overlap = train_fams & val_fams
    tt_overlap = train_fams & test_fams
    vt_overlap = val_fams & test_fams
    assert not tv_overlap, f"Family overlap between train and val: {sorted(tv_overlap)}"
    assert not tt_overlap, f"Family overlap between train and test: {sorted(tt_overlap)}"
    assert not vt_overlap, f"Family overlap between val and test: {sorted(vt_overlap)}"

    # Confirm every family maps to exactly one split
    fam_split_nunique = df.groupby("family")["split"].nunique()
    split_families = fam_split_nunique[fam_split_nunique > 1]
    assert split_families.empty, (
        f"Families split across multiple partitions: {split_families.to_dict()}"
    )

    # 6. Minimum family diversity constraints per stratum
    stratum_col = _assign_stratum_key(df)
    for stratum in ALL_STRATA:
        sub = df.loc[stratum_col == stratum]
        if len(sub) == 0:
            continue
        f_total = int(sub["family"].nunique())
        min_val_f = max(3, math.ceil(TARGET_RATIOS["val"] * f_total))
        min_test_f = max(4, math.ceil(TARGET_RATIOS["test"] * f_total))
        val_f = int(sub.loc[sub["split"] == "val", "family"].nunique())
        test_f = int(sub.loc[sub["split"] == "test", "family"].nunique())
        assert val_f >= min_val_f, (
            f"Stratum '{stratum}' val_families={val_f} < required min {min_val_f} (F={f_total})"
        )
        assert test_f >= min_test_f, (
            f"Stratum '{stratum}' test_families={test_f} < required min {min_test_f} (F={f_total})"
        )


def print_audit_report(
    df: pd.DataFrame,
    mal_stats: dict[str, object],
    output_csv: Path,
    seed: int = SEED,
) -> None:
    """Print the 5-section Audit Report required for dataset sign-off."""
    total_n = len(df)
    mal_df = df.loc[df["label"] == 1]
    ben_df = df.loc[df["label"] == 0]
    clean_df = df.loc[df["group"] == "Clean"]
    unusual_df = df.loc[df["group"] == "Unusual"]

    n_mal = len(mal_df)
    n_ben = len(ben_df)
    ratio_ben_to_mal = (n_ben / n_mal) if n_mal else 0.0

    # ------------------------------------------------------------------
    # SECTION 1: Overview by label, group, context & malicious:benign ratio
    # ------------------------------------------------------------------
    print("=" * 98)
    print("BÁO CÁO KIỂM TOÁN BỘ DỮ LIỆU HUẤN LUYỆN CUỐI CÙNG (dataset_final.csv)")
    print("=" * 98)
    print("\n[1] TỔNG QUAN SỐ MẪU THEO LABEL, GROUP, CONTEXT & TỈ LỆ ĐỘC:SẠCH")
    print("-" * 98)
    print(f"  Nguồn độc hại (oracle_results_2.csv) : {mal_stats['raw_total']} dòng gốc -> {mal_stats['valid_count']} valid==1 -> {mal_stats['rows_after']} duy nhất (loại {mal_stats['duplicates_removed']} trùng)")
    print(f"  Tổng số mẫu dataset_final.csv        : {total_n}")
    print(f"  Tỉ lệ Độc : Sạch thực tế             : {n_mal} : {n_ben}  (1 : {ratio_ben_to_mal:.2f} | {100.0 * n_mal / total_n:.2f}% : {100.0 * n_ben / total_n:.2f}%)")
    print()

    print("  Theo label:")
    for lbl, name in [(1, "Malicious (1)"), (0, "Benign    (0)")]:
        cnt = int((df["label"] == lbl).sum())
        print(f"    - {name:<15}: {cnt:>5} ({100.0 * cnt / total_n:6.2f}%)")

    print("\n  Theo group:")
    for grp in ["Tainted", "Obfuscated", "Clean", "Unusual"]:
        cnt = int((df["group"] == grp).sum())
        fams = int(df.loc[df["group"] == grp, "family"].nunique())
        print(f"    - {grp:<15}: {cnt:>5} ({100.0 * cnt / total_n:6.2f}%) | {fams:>3} family")

    print("\n  Theo context:")
    for ctx in ["attribute", "html_body", "js_string", "url", "benign"]:
        cnt = int((df["context"] == ctx).sum())
        fams = int(df.loc[df["context"] == ctx, "family"].nunique())
        print(f"    - {ctx:<15}: {cnt:>5} ({100.0 * cnt / total_n:6.2f}%) | {fams:>3} family")

    # ------------------------------------------------------------------
    # SECTION 2: Split distribution per stratum (count, %, family count)
    # ------------------------------------------------------------------
    print("\n[2] PHÂN BỐ SPLIT THEO TỪNG TẦNG (SỐ MẪU | TỈ LỆ % | SỐ FAMILY)")
    print("-" * 98)
    stratum_col = _assign_stratum_key(df)
    hdr2 = (
        f"  {'Tầng (Stratum)':<14} | {'Tổng':>5} {'(Fam)':>6} | "
        f"{'Train':>20} | {'Val':>20} | {'Test':>20}"
    )
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))

    def _fmt_cell(sub_df: pd.DataFrame, denom: int) -> str:
        n = len(sub_df)
        pct = 100.0 * n / denom if denom else 0.0
        nf = int(sub_df["family"].nunique())
        return f"{n:>4} ({pct:5.1f}%, {nf:>2}f)"

    for stratum in ALL_STRATA:
        sub = df.loc[stratum_col == stratum]
        s_total = len(sub)
        s_fams = int(sub["family"].nunique())
        tr = sub.loc[sub["split"] == "train"]
        va = sub.loc[sub["split"] == "val"]
        te = sub.loc[sub["split"] == "test"]
        print(
            f"  {stratum:<14} | {s_total:>5} ({s_fams:>3}f) | "
            f"{_fmt_cell(tr, s_total):>20} | {_fmt_cell(va, s_total):>20} | {_fmt_cell(te, s_total):>20}"
        )

    print("  " + "-" * (len(hdr2) - 2))
    tot_fams = int(df["family"].nunique())
    tr_all = df.loc[df["split"] == "train"]
    va_all = df.loc[df["split"] == "val"]
    te_all = df.loc[df["split"] == "test"]
    print(
        f"  {'Tổng':<14} | {total_n:>5} ({tot_fams:>3}f) | "
        f"{_fmt_cell(tr_all, total_n):>20} | {_fmt_cell(va_all, total_n):>20} | {_fmt_cell(te_all, total_n):>20}"
    )

    # ------------------------------------------------------------------
    # SECTION 3: Length distribution comparison (Malicious vs Clean vs Unusual)
    # ------------------------------------------------------------------
    print("\n[3] SO SÁNH PHÂN BỐ ĐỘ DÀI payload_decoded (Malicious vs Clean vs Unusual)")
    print("-" * 98)
    hdr3 = (
        f"  {'Nhóm':<10} {'N':>5} {'min':>5} {'p10':>6} {'p25':>6} {'p50':>6} "
        f"{'p75':>6} {'p90':>6} {'p95':>6} {'p99':>6} {'max':>5} {'>200 (%)':>10}"
    )
    print(hdr3)
    print("  " + "-" * (len(hdr3) - 2))

    for name, sub in [("Malicious", mal_df), ("Clean", clean_df), ("Unusual", unusual_df)]:
        lens = [len(s) for s in sub["payload_decoded"].astype(str)]
        st = compute_length_percentiles(lens)
        print(
            f"  {name:<10} {int(st['count']):>5} {st['min']:>5.0f} {st['p10']:>6.1f} "
            f"{st['p25']:>6.1f} {st['p50']:>6.1f} {st['p75']:>6.1f} {st['p90']:>6.1f} "
            f"{st['p95']:>6.1f} {st['p99']:>6.1f} {st['max']:>5.0f} {st['gt200_pct']:>9.2f}%"
        )

    # ------------------------------------------------------------------
    # SECTION 4: attacker.test distribution by group and by split
    # ------------------------------------------------------------------
    print("\n[4] THỐNG KÊ MẪU CHỨA 'attacker.test' THEO NHÓM VÀ THEO SPLIT")
    print("-" * 98)
    atk_series = df["payload_decoded"].astype(str).str.contains("attacker.test", regex=False)

    print("  4a. Theo nhóm (Group):")
    print(f"  {'Nhóm':<12} | {'Tổng mẫu':>8} | {'Có attacker.test':>16} | {'Tỉ lệ (%)':>10} | {'Số family có':>12}")
    print("  " + "-" * 72)
    for name, mask in [
        ("Malicious", df["label"] == 1),
        ("Clean", df["group"] == "Clean"),
        ("Unusual", df["group"] == "Unusual"),
        ("Tổng", pd.Series(True, index=df.index)),
    ]:
        n_grp = int(mask.sum())
        n_atk = int((mask & atk_series).sum())
        n_fam = int(df.loc[mask & atk_series, "family"].nunique())
        pct = 100.0 * n_atk / n_grp if n_grp else 0.0
        print(f"  {name:<12} | {n_grp:>8} | {n_atk:>16} | {pct:>9.2f}% | {n_fam:>12}")

    print("\n  4b. Theo split (và chi tiết Malicious / Unusual trong từng split):")
    print(
        f"  {'Split':<8} | {'Tổng':>6} | {'attacker.test':>13} {'(%)':>7} | "
        f"{'Malicious atk':>18} | {'Unusual atk':>18}"
    )
    print("  " + "-" * 86)
    for sp in SPLIT_ORDER:
        sp_mask = df["split"] == sp
        n_sp = int(sp_mask.sum())
        n_sp_atk = int((sp_mask & atk_series).sum())
        pct_sp = 100.0 * n_sp_atk / n_sp if n_sp else 0.0

        m_mask = sp_mask & (df["label"] == 1)
        m_n = int(m_mask.sum())
        m_atk = int((m_mask & atk_series).sum())
        m_pct = 100.0 * m_atk / m_n if m_n else 0.0

        u_mask = sp_mask & (df["group"] == "Unusual")
        u_n = int(u_mask.sum())
        u_atk = int((u_mask & atk_series).sum())
        u_pct = 100.0 * u_atk / u_n if u_n else 0.0

        print(
            f"  {sp:<8} | {n_sp:>6} | {n_sp_atk:>13} {pct_sp:>6.2f}% | "
            f"{m_atk:>4}/{m_n:<4} ({m_pct:5.1f}%) | {u_atk:>4}/{u_n:<4} ({u_pct:5.1f}%)"
        )

    # ------------------------------------------------------------------
    # SECTION 5: Random qualitative inspection (10 Clean + 15 Unusual >= 5 attacker.test)
    # ------------------------------------------------------------------
    print("\n[5] MẪU NGẪU NHIÊN ĐỂ KIỂM DUYỆT BẰNG MẮT THƯỜNG (SEED = 42)")
    print("-" * 98)
    rng = random.Random(seed)

    clean_indices = clean_df.index.tolist()
    sample_clean_idx = rng.sample(clean_indices, min(10, len(clean_indices)))

    unu_atk_df = unusual_df.loc[
        unusual_df["payload_decoded"].astype(str).str.contains("attacker.test", regex=False)
    ]
    unu_non_df = unusual_df.loc[
        ~unusual_df["payload_decoded"].astype(str).str.contains("attacker.test", regex=False)
    ]
    n_atk_show = min(6, len(unu_atk_df))  # Guarantee >= 5 attacker.test samples
    n_non_show = min(15 - n_atk_show, len(unu_non_df))
    sample_unusual_idx = rng.sample(unu_atk_df.index.tolist(), n_atk_show) + rng.sample(
        unu_non_df.index.tolist(), n_non_show
    )
    rng.shuffle(sample_unusual_idx)

    print("  --- 10 mẫu Clean ngẫu nhiên ---")
    for rank, idx in enumerate(sample_clean_idx, start=1):
        row = df.loc[idx]
        dlen = len(str(row["payload_decoded"]))
        print(
            f"  {rank:2d}. [{row['id']}] ({row['split']:<5} | {row['family']:<26} | len={dlen:3d}) | "
            f"{row['payload']}"
        )

    print(f"\n  --- 15 mẫu Unusual ngẫu nhiên (gồm {n_atk_show} mẫu chứa attacker.test) ---")
    for rank, idx in enumerate(sample_unusual_idx, start=1):
        row = df.loc[idx]
        dlen = len(str(row["payload_decoded"]))
        print(
            f"  {rank:2d}. [{row['id']}] ({row['split']:<5} | {row['family']:<33} | len={dlen:3d}) | "
            f"{row['payload']}"
        )

    # ------------------------------------------------------------------
    # SECTION 6: Sensitive keyword prevalence across Malicious / Unusual / Clean
    # ------------------------------------------------------------------
    print("\n[6] THỐNG KÊ TẦN SUẤT TỪ KHÓA NHẠY CẢM TRÊN payload_decoded (Malicious vs Unusual vs Clean)")
    print("-" * 116)
    hdr6 = (
        f"  {'Từ khóa':<16} | {'Malicious (N=' + str(len(mal_df)) + ')':>18} | "
        f"{'Unusual (N=' + str(len(unusual_df)) + ')':>18} | "
        f"{'Clean (N=' + str(len(clean_df)) + ')':>18} | "
        f"{'Unu_Fam':>7} | {'Unu (tr/va/te)':>15}"
    )
    print(hdr6)
    print("  " + "-" * (len(hdr6) - 2))

    mal_dec = mal_df["payload_decoded"].astype(str)
    unu_dec = unusual_df["payload_decoded"].astype(str)
    cln_dec = clean_df["payload_decoded"].astype(str)

    for kw in SENSITIVE_KEYWORDS:
        m_cnt = int(mal_dec.str.contains(kw, regex=False).sum())
        u_mask = unu_dec.str.contains(kw, regex=False)
        u_cnt = int(u_mask.sum())
        c_cnt = int(cln_dec.str.contains(kw, regex=False).sum())

        m_pct = 100.0 * m_cnt / len(mal_df) if len(mal_df) else 0.0
        u_pct = 100.0 * u_cnt / len(unusual_df) if len(unusual_df) else 0.0
        c_pct = 100.0 * c_cnt / len(clean_df) if len(clean_df) else 0.0

        u_sub = unusual_df.loc[u_mask]
        u_fams = int(u_sub["family"].nunique())
        u_tr = int((u_sub["split"] == "train").sum())
        u_va = int((u_sub["split"] == "val").sum())
        u_te = int((u_sub["split"] == "test").sum())

        m_cell = f"{m_cnt:>5} ({m_pct:5.2f}%)"
        u_cell = f"{u_cnt:>5} ({u_pct:5.2f}%)"
        c_cell = f"{c_cnt:>5} ({c_pct:5.2f}%)"
        split_cell = f"{u_tr}/{u_va}/{u_te}"
        print(
            f"  {kw:<16} | {m_cell:>18} | {u_cell:>18} | {c_cell:>18} | "
            f"{u_fams:>7} | {split_cell:>15}"
        )

    print("\n" + "=" * 98)
    print(
        "ASSERTIONS PASSED: zero family overlap | min family diversity | "
        "zero payload_decoded duplicate | zero NaN"
    )
    print(f"OUTPUT SAVED     : {output_csv}")
    print("=" * 98)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Merge malicious + benign datasets and build constrained multi-trial group splits."
    )
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE_CSV, help="Path to oracle_results_2.csv")
    parser.add_argument("--benign", type=Path, default=DEFAULT_BENIGN_CSV, help="Path to benign_dataset_4000.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_FINAL_CSV, help="Path to dataset_final.csv")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS, help="Number of trials per stratum (default: 5000)")
    args = parser.parse_args(argv)

    merged_df, mal_stats = prepare_merged_dataframe(args.oracle, args.benign)
    final_df = constrained_multitrial_group_split(
        merged_df,
        target_ratios=TARGET_RATIOS,
        seed=args.seed,
        n_trials=args.n_trials,
    )
    run_mandatory_assertions(final_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output, index=False, encoding="utf-8")

    print_audit_report(final_df, mal_stats, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
