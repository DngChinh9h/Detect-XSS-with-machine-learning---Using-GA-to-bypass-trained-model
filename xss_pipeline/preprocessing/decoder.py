"""Single Source of Truth for XSS payload decoding and malicious dataset deduplication.

Used across dataset preparation, training, and inference pipelines in Project T02
to guarantee identical payload normalization.
"""

from __future__ import annotations

import html
import re
import urllib.parse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

# JavaScript hex (\xHH), 4-digit unicode (\uHHHH), and ES6 braced unicode (\u{H+}).
_JS_HEX_RE = re.compile(r"\\x([0-9a-fA-F]{2})")
_JS_UNICODE4_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
_JS_UNICODE_BRACE_RE = re.compile(r"\\u\{([0-9a-fA-F]+)\}")

# Lower value means higher priority when resolving duplicate decoded payloads.
_GROUP_PRIORITY: dict[str, int] = {
    "Tainted": 0,
    "Obfuscated": 1,
}


def _replace_hex_or_unicode4(match: re.Match[str]) -> str:
    return chr(int(match.group(1), 16))


def _replace_unicode_brace(match: re.Match[str]) -> str:
    try:
        return chr(int(match.group(1), 16))
    except (ValueError, OverflowError):
        return match.group(0)


def decode_payload(raw: Any, max_rounds: int = 3) -> str:
    """Iteratively decode URL, HTML entity, and JS hex/unicode escapes in a payload.

    Args:
        raw: Raw payload value. ``None``, ``NaN``, or any non-string returns ``""``.
        max_rounds: Maximum number of decoding passes (stops early on convergence).

    Returns:
        Normalized, decoded, lowercased, and stripped payload string.
    """
    if not isinstance(raw, str):
        return ""

    s = str(raw)
    for _ in range(max(0, max_rounds)):
        prev = s
        # 1. URL decode
        s = urllib.parse.unquote(s)
        # 2. HTML entity decode
        s = html.unescape(s)
        # 3. JavaScript hex & unicode escapes
        s = _JS_HEX_RE.sub(_replace_hex_or_unicode4, s)
        s = _JS_UNICODE4_RE.sub(_replace_hex_or_unicode4, s)
        s = _JS_UNICODE_BRACE_RE.sub(_replace_unicode_brace, s)
        if s == prev:
            break

    return s.lower().strip()


def deduplicate_malicious_df(
    df: pd.DataFrame,
    max_rounds: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Decode payloads and deduplicate malicious rows with deterministic priority.

    Priority rules when multiple rows share the same ``payload_decoded``:
    1. Keep ``group == "Tainted"`` ahead of ``group == "Obfuscated"`` (and other groups).
    2. Within the same group, keep the first row ordered ascending by ``id``.

    Args:
        df: Input pandas DataFrame containing at least a ``payload`` column.
        max_rounds: Maximum decoding rounds passed to :func:`decode_payload`.

    Returns:
        A tuple ``(dedup_df, stats)`` where ``dedup_df`` is the deduplicated copy
        with ``payload_decoded`` populated and ``stats`` summarizes row counts and
        removed duplicates per ``context``.
    """
    import pandas as pd

    if "payload" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'payload' column.")

    work = df.copy()
    work["payload_decoded"] = [
        decode_payload(val, max_rounds=max_rounds)
        for val in work["payload"].tolist()
    ]

    rows_before = int(len(work))
    has_context = "context" in work.columns

    def _normalize_context(val: Any) -> str:
        if pd.isna(val):
            return "unknown"
        text = str(val).strip()
        return text if text else "unknown"

    if rows_before == 0:
        stats: dict[str, Any] = {
            "rows_before": 0,
            "rows_after": 0,
            "duplicates_removed": 0,
            "duplicates_by_context": {},
        }
        return work.reset_index(drop=True), stats

    orig_pos_col = "__decoder_orig_pos__"
    group_prio_col = "__decoder_group_priority__"
    work[orig_pos_col] = list(range(rows_before))

    if "group" in work.columns:
        work[group_prio_col] = [
            _GROUP_PRIORITY.get(val, 2) if isinstance(val, str) else 2
            for val in work["group"].tolist()
        ]
    else:
        work[group_prio_col] = 2

    sort_cols = [group_prio_col]
    if "id" in work.columns:
        sort_cols.append("id")
    sort_cols.append(orig_pos_col)

    prioritized = work.sort_values(
        by=sort_cols,
        ascending=True,
        kind="mergesort",
        na_position="last",
    )

    duplicated_mask = prioritized.duplicated(subset=["payload_decoded"], keep="first")
    dropped = prioritized.loc[duplicated_mask]
    kept = (
        prioritized.loc[~duplicated_mask]
        .sort_values(by=orig_pos_col, ascending=True, kind="mergesort")
        .drop(columns=[group_prio_col, orig_pos_col])
        .reset_index(drop=True)
    )

    duplicates_by_context: dict[str, int] = {}
    if has_context:
        all_contexts = sorted({_normalize_context(v) for v in work["context"].tolist()})
        duplicates_by_context = {ctx: 0 for ctx in all_contexts}
        for ctx_val in dropped["context"].tolist():
            ctx = _normalize_context(ctx_val)
            duplicates_by_context[ctx] = duplicates_by_context.get(ctx, 0) + 1

    rows_after = int(len(kept))
    stats = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicates_removed": rows_before - rows_after,
        "duplicates_by_context": duplicates_by_context,
    }
    return kept, stats
