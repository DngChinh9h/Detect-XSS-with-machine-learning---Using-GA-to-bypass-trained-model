"""Kiem thu hoi quy cho 4 hang muc sua loi Oracle pipeline + GA.

Chay app.py o terminal khac, sau do:
  python test_regressions.py ../malicious_payloads_mixed_3060_v8.csv
"""

import argparse
import asyncio
import csv
import json
from pathlib import Path

from ga.build import build_payload
from ga.ga_runner import family_for_individual
from oracle import check_batch


TARGET_IDS = [
    # 1. attribute onfocusin + autofocus + document.location
    "SYN0008",
    # 2. js_string new Image().src & document.location -> attacker.test
    "SYN0024",
    "SYN0039",
    # 3. html_body iframe_srcdoc document.location -> attacker.test
    "SYN0485",
    "SYN0070",
    # 4. html_body anchor_js_href_* & svg_xlink_* (event=none)
    "SYN0032",
    "SYN0091",
    "SYN0101",
    "SYN3009",
]


def load_csv_by_id(csv_path: str) -> dict[str, dict]:
    with open(csv_path, encoding="utf-8", newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def build_ga_cases() -> list[dict]:
    # Kiem tra ca anh xa family cho cac loai escape o attribute & js_string.
    assert family_for_individual("attribute", {"escape": "'"}) == "attr_break_squote_ga"
    assert family_for_individual("attribute", {"escape": '"'}) == "attr_break_dquote_ga"
    assert family_for_individual("js_string", {"escape": "';"}) == "js_break_squote_ga"
    assert family_for_individual("js_string", {"escape": '";'}) == "js_break_dquote_ga"
    assert family_for_individual("js_string", {"escape": "`"}) == "js_break_template_ga"
    assert family_for_individual("js_string", {"escape": "${"}) == "js_break_template_ga"
    assert family_for_individual("js_string", {"escape": "*/"}) == "js_break_comment_ga"
    assert family_for_individual("js_string", {"escape": ""}) == "js_direct_code_no_quote_ga"
    assert family_for_individual("url", {"escape": "", "target_tag": "a"}) == "url_js_ga"
    assert family_for_individual("url", {"escape": "", "target_tag": "iframe"}) == "iframe_srcdoc_ga"
    assert family_for_individual("url", {"escape": "", "target_tag": "object"}) == "object_data_ga"

    ga_specs = [
        (
            "ga-reg-attr-squote",
            "attribute",
            {"escape": "'", "trigger": "onfocus", "action": "alert(1)", "target_tag": "input"},
            "attr_break_squote_ga",
        ),
        (
            "ga-reg-js-squote",
            "js_string",
            {"escape": "';", "trigger": "direct", "action": "window.__markXSS()", "target_tag": "script"},
            "js_break_squote_ga",
        ),
    ]
    cases = []
    for case_id, context, ind, expected_family in ga_specs:
        family = family_for_individual(context, ind)
        assert family == expected_family, f"{case_id}: expected {expected_family}, got {family}"
        event = ind["trigger"] if ind["trigger"].startswith("on") else "none"
        payload = build_payload(context, **ind)
        cases.append({
            "id": case_id,
            "payload": payload,
            "context": context,
            "event": event,
            "family": family,
            "target_tag": ind["target_tag"],
        })
    return cases


async def run_tests(csv_path: str, log_path: Path) -> None:
    if log_path.exists():
        log_path.unlink()
    rows = load_csv_by_id(csv_path)
    cases = [rows[cid] for cid in TARGET_IDS] + build_ga_cases()
    flags = await check_batch(cases, concurrency=4, log_path=log_path)

    records = {
        rec["id"]: rec
        for rec in (json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip())
    }

    failures = []
    print(f"{'id':22} {'ctx':10} {'event':10} {'action':11} {'valid':5} {'ms':6} {'error'}")
    for case, flag in zip(cases, flags):
        rec = records[case["id"]]
        print(
            f"{rec['id']:22} {rec['context']:10} {rec['event']:10} "
            f"{rec['action']:11} {rec['valid']:<5} {rec['elapsed_ms']:<6} {rec['error']}"
        )
        if rec["valid"] != 1:
            failures.append(f"{rec['id']}: valid={rec['valid']} error={rec['error']!r}")
        if rec["id"] == "SYN0008" and rec["error"] and "Execution context was destroyed" in rec["error"]:
            failures.append(f"SYN0008 still hit Execution context was destroyed: {rec['error']!r}")
        if rec["id"] in {"SYN0024", "SYN0039"} and rec["elapsed_ms"] >= 1000:
            failures.append(f"{rec['id']} too slow: {rec['elapsed_ms']}ms (>= 1000ms)")

    if failures:
        raise RuntimeError("Regression failures:\n  - " + "\n  - ".join(failures))
    print(f"\nALL {len(cases)} REGRESSION CASES PASSED.")


def main():
    parser = argparse.ArgumentParser(description="Chay bo kiem thu hoi quy cho Oracle + GA")
    parser.add_argument("input_csv", nargs="?", default="../malicious_payloads_mixed_3060_v8.csv")
    parser.add_argument("--log", default="verification_artifacts/regression_log.jsonl")
    args = parser.parse_args()
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run_tests(args.input_csv, log_path))


if __name__ == "__main__":
    main()
