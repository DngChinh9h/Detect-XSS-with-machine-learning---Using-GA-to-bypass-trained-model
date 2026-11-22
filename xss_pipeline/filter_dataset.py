"""Loc CSV bang Oracle, checkpoint/merge theo khoa id cua dataset."""

import argparse
import asyncio
import csv
import json
from pathlib import Path

from event_actions import browser_compatibility_note, get_event_plan
from oracle import check_batch, load_done_ids

REQUIRED_COLUMNS = {"id", "payload", "group", "context", "event", "family"}


def read_dataset(path: str) -> tuple[list[dict], list[str]]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV khong co header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV thieu cot bat buoc: {sorted(missing)}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    ids = [row["id"] for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Cot id phai khong rong va duy nhat de checkpoint/merge an toan")
    return rows, fieldnames


def read_results_by_id(log_path: str) -> dict[str, int]:
    values = {}
    path = Path(log_path)
    if not path.exists():
        return values
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                if "id" in record:
                    values[str(record["id"])] = int(record["valid"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return values


def write_results(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    names = fieldnames if "valid" in fieldnames else fieldnames + ["valid"]
    if "oracle_status" not in names:
        names.append("oracle_status")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Loc dataset bang Playwright Oracle")
    parser.add_argument("input_csv", help="CSV co id,payload,group,context,event,family")
    parser.add_argument("--context", choices=["html_body", "attribute", "js_string", "url"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--log")
    parser.add_argument("--concurrency", type=int, default=15)
    args = parser.parse_args()

    label = args.context or "all"
    output = args.output or f"oracle_results_{label}_{args.seed}.csv"
    log = args.log or f"oracle_log_{label}_{args.seed}.jsonl"
    rows, fieldnames = read_dataset(args.input_csv)
    target = [r.copy() for r in rows if r["group"] in {"Tainted", "Obfuscated"}]
    if args.context:
        target = [r for r in target if r["context"] == args.context]

    for row in target:
        get_event_plan(row["event"])

    done = load_done_ids(log)
    runnable, excluded = [], []
    for row in target:
        plan = get_event_plan(row["event"])
        reason = browser_compatibility_note(row["payload"], row["context"], row["family"])
        if not plan.supported:
            reason = f"unsupported_browser: {plan.note}"
        if reason:
            row["valid"] = 0
            row["oracle_status"] = reason
            excluded.append(row)
        else:
            runnable.append(row)
    target_ids = {row["id"] for row in runnable}
    todo = [row for row in runnable if row["id"] not in done]
    already = len(target_ids & done)
    print(f"Tong {len(target)} dong; excluded/co flag {len(excluded)}; da xong dung id {already}; con lai {len(todo)}.")
    if todo:
        asyncio.run(check_batch(todo, concurrency=args.concurrency, log_path=log))

    result_by_id = read_results_by_id(log)
    missing = [row["id"] for row in runnable if row["id"] not in result_by_id]
    if missing:
        raise RuntimeError(f"Log thieu {len(missing)} id sau oracle, vi du: {missing[:5]}")
    for row in runnable:
        row["valid"] = result_by_id[row["id"]]
        row["oracle_status"] = "tested"
    write_results(output, target, fieldnames)
    valid = sum(int(row["valid"]) for row in target)
    pct = valid / len(target) * 100 if target else 0
    print(f"Da ghi {output}: valid=1 {valid}/{len(target)} ({pct:.1f}%).")
    print(f"Checkpoint log: {log}")


if __name__ == "__main__":
    main()
