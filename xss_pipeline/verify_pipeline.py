"""Kiem tra bat buoc truoc khi ket luan pipeline OK.

Chay app.py o terminal khac, sau do:
  python verify_pipeline.py ../malicious_payloads_mixed_3060_v8.csv
Script in du 7 muc: mapping, mau theo event, known +/- , raw HTML,
concurrency, resume va tong hop.
"""

import argparse
import asyncio
import csv
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from event_actions import EVENT_ACTIONS, action_for, browser_compatibility_note
from oracle import check_batch, load_done_ids, rendered_html


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "payload", "context", "event", "family"}
        if reader.fieldnames is None or required - set(reader.fieldnames):
            raise ValueError(f"CSV thieu cot bat buoc: {sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    ids = [row["id"] for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Cot id phai khong rong va duy nhat")
    return rows


def samples_by_event(rows, limit=5):
    grouped = defaultdict(list)
    for row in rows:
        if len(grouped[row["event"]]) < limit:
            grouped[row["event"]].append(row)
    return grouped


def print_failed_records(log_path: Path, heading: str) -> None:
    """In record JSONL that bai, khong mat payload hay diagnostic."""
    print(f"\n{heading} ({log_path})")
    if not log_path.exists():
        print("Khong co log")
        return
    with log_path.open(encoding="utf-8") as f:
        failed = [json.loads(line) for line in f if line.strip() and not json.loads(line)["valid"]]
    if not failed:
        print("Khong co valid=0")
        return
    for record in failed:
        print("FAIL " + json.dumps({
            key: record.get(key) for key in
            ("id", "event", "action", "payload", "context", "family", "error", "attempts")
        }, ensure_ascii=False))


# page.content() la HTML serialization cua DOM, khong phai ban sao byte-for-byte
# cua response. Vi vay tag/attribute lowercase va boolean attr="" la hop le.
# Chi coi la escape sai neu serializer xuat hien THEM entity cho <, >, " hoac &
# ma payload goc khong co entity tuong ung.
_ESCAPE_TOKENS = {
    "<": re.compile(r"(?:&lt;|&#0*60;|&#x0*3c;)", re.IGNORECASE),
    ">": re.compile(r"(?:&gt;|&#0*62;|&#x0*3e;)", re.IGNORECASE),
    '"': re.compile(r"(?:&quot;|&#0*34;|&#x0*22;)", re.IGNORECASE),
    "&": re.compile(r"(?:&amp;|&#0*38;|&#x0*26;)", re.IGNORECASE),
}


def introduced_html_escapes(payload: str, rendered_html: str) -> list[str]:
    """Tra ve entity moi do sink escape, bo qua entity von co trong payload."""
    introduced = []
    for character, pattern in _ESCAPE_TOKENS.items():
        source_count = len(pattern.findall(payload))
        rendered_count = len(pattern.findall(rendered_html))
        if rendered_count > source_count:
            introduced.append(f"{character}: +{rendered_count - source_count}")
    return introduced


async def main_async(args):
    rows = read_rows(args.input_csv)
    grouped = samples_by_event(rows, args.per_event)
    work = (Path(args.workdir) if args.workdir else
            Path("verification_artifacts") /
            datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ"))
    if work.exists():
        if not args.clean_workdir:
            raise FileExistsError(
                f"Thu muc artifact da ton tai: {work}. Dung --clean-workdir de xoa co chu dich "
                "hoac chon --workdir khac."
            )
        shutil.rmtree(work)
    work.mkdir(parents=True)
    print(f"Verification artifacts: {work.resolve()}")
    print(f"Sample detail log: {(work / 'samples.jsonl').resolve()}")
    print(f"Known-case detail log: {(work / 'known.jsonl').resolve()}")

    print("1. EVENT -> ORACLE ACTION")
    for event in sorted(grouped):
        plan = EVENT_ACTIONS[event]
        event_rows = [row for row in rows if row["event"] == event]
        contexts = sorted({row["context"] for row in event_rows})
        sinks = []
        for context in contexts:
            if context == "attribute":
                sink = ("input#payload-target" if event in
                        {"onkeyup", "oninput", "onchange", "onfocus", "onfocusin"}
                        else "img#payload-target (404 that)")
            elif context == "html_body":
                sink = "phan tu do payload tao (body-root cho body load/pageshow)"
            elif context == "js_string":
                sink = "script string raw + #payload-target"
            else:
                sink = "a/iframe/object carrier theo family"
            sinks.append(f"{context}: {sink}")
        sample = event_rows[0]
        print(f"{event:38} rows={sum(r['event'] == event for r in rows):4} "
              f"action={action_for(event, sample['context'], sample['family']):12} "
              f"sink={' | '.join(sinks)}; supported={plan.supported}; note={plan.note}")

    sampled = [row for event in sorted(grouped) if EVENT_ACTIONS[event].supported
               for row in grouped[event]
               if not browser_compatibility_note(row["payload"], row["context"], row["family"])]
    sample_log = work / "samples.jsonl"
    flags = await check_batch(sampled, concurrency=args.concurrency, log_path=sample_log)
    summary = {}
    cursor = 0
    print("\n2. MAU THUC TE THEO EVENT")
    print("event | so dong test | valid=1 | valid=0")
    for event in sorted(grouped):
        event_rows = grouped[event]
        runnable_rows = [row for row in event_rows if EVENT_ACTIONS[event].supported
                         and not browser_compatibility_note(row["payload"], row["context"], row["family"])]
        event_flags = flags[cursor:cursor + len(runnable_rows)]
        cursor += len(runnable_rows)
        summary[event] = (len(runnable_rows), sum(event_flags), len(event_flags) - sum(event_flags))
        print(f"{event} | {summary[event][0]} | {summary[event][1]} | {summary[event][2]}")

    if args.verbose_failures:
        print_failed_records(sample_log, "2b. CHI TIET TAT CA FAIL MAU THUC TE")

    # Mot event chi co 1-2 sample khong du bang chung de ket luan Oracle
    # hong. Van in canh bao de nguoi van hanh xem lai, nhung chi fail verify
    # khi co it nhat 3 sample da test ma tat ca deu valid=0.
    sparse_all_zero = [event for event, (tested, yes, _) in summary.items()
                       if 0 < tested < 3 and yes == 0 and EVENT_ACTIONS[event].supported]
    all_zero = [event for event, (tested, yes, _) in summary.items()
                if tested >= 3 and yes == 0 and EVENT_ACTIONS[event].supported]
    if sparse_all_zero:
        print("CANH BAO: event co <3 mau va 100% valid=0 (khong fail verify): "
              + ", ".join(sparse_all_zero))

    known = [
        {"id": "known-positive-error", "payload": "<img src=x onerror=alert(1)>",
         "context": "html_body", "event": "onerror", "family": "manual"},
        {"id": "known-positive-hover", "payload": "<div onmouseover=alert(1)>x</div>",
         "context": "html_body", "event": "onmouseover", "family": "manual"},
        {"id": "known-positive-focus", "payload": "<input onfocus=alert(1)>",
         "context": "html_body", "event": "onfocus", "family": "manual"},
        {"id": "known-negative-syntax", "payload": "<img src=x onerror=>",
         "context": "html_body", "event": "onerror", "family": "manual"},
        {"id": "known-negative-text", "payload": "alert(1)",
         "context": "html_body", "event": "none", "family": "manual"},
        {"id": "known-negative-unknown-url", "payload": "https://example.invalid/",
         "context": "url", "event": "none", "family": "unknown"},
    ]
    known_flags = await check_batch(known, concurrency=3, log_path=work / "known.jsonl")
    expected = [1, 1, 1, 0, 0, 0]
    print("\n3. KNOWN POSITIVE/NEGATIVE", list(zip([x["id"] for x in known], known_flags)))
    if args.verbose_failures:
        print_failed_records(work / "known.jsonl", "3b. CHI TIET FAIL KNOWN CASES")
    if known_flags != expected:
        raise RuntimeError(f"Known-case oracle failure: expected {expected}, got {known_flags}")

    print("\n4. RAW HTML")
    # Lay payload nguyen van tu CSV o ca bon context, khong dung payload tu tao.
    raw_cases = []
    for context in ("html_body", "attribute", "js_string", "url"):
        raw_cases.append(next(row for row in rows if row["context"] == context))
    for case in raw_cases:
        html = await rendered_html(case)
        print(case["id"], case["context"], html)
        escapes = introduced_html_escapes(case["payload"], html)
        if escapes:
            raise RuntimeError(
                f"Payload bi escape/encode: {case['id']} ({', '.join(escapes)})"
            )

    concurrent = rows[:args.concurrent_size]
    once = await check_batch(concurrent, concurrency=args.concurrency, log_path=work / "concurrent.jsonl")
    again = await check_batch(concurrent, concurrency=1, log_path=work / "serial.jsonl")
    print(f"\n5. CONCURRENCY rows={len(concurrent)} same_as_serial={once == again}")
    if once != again:
        raise RuntimeError("Concurrent result lech serial result")

    resume_cases = rows[:min(10, len(rows))]
    resume_log = work / "resume.jsonl"
    # Gia lap bi dung giua chung: dot 1 chi ghi mot nua, dot 2 doc id trong
    # checkpoint va chi xu ly phan con lai.
    split = max(1, len(resume_cases) // 2)
    await check_batch(resume_cases[:split], concurrency=args.concurrency, log_path=resume_log)
    before = resume_log.read_text(encoding="utf-8").count("\n")
    done = load_done_ids(resume_log)
    todo = [case for case in resume_cases if case["id"] not in done]
    if todo:
        await check_batch(todo, concurrency=args.concurrency, log_path=resume_log)
    after = resume_log.read_text(encoding="utf-8").count("\n")
    print(f"6. RESUME done_dot_1={len(done)}/{len(resume_cases)} xu_ly_tiep={len(todo)} "
          f"log_lines_before={before} after={after}")
    final_done = load_done_ids(resume_log)
    if len(final_done) != len(resume_cases) or after != len(resume_cases):
        raise RuntimeError("Resume bi trung hoac bo sot dong")

    print("\n7. TONG HOP EVENT DA TEST")
    for event, (tested, yes, no) in sorted(summary.items()):
        rate = f"{yes/tested*100:.1f}%" if tested else "N/A (unsupported/flagged)"
        print(f"{event}: tested={tested}, valid={yes}/{tested} ({rate}), action="
              f"{action_for(event, grouped[event][0]['context'], grouped[event][0]['family'])}")
    unsupported = [event for event in sorted(grouped) if not EVENT_ACTIONS[event].supported]
    patched = sum(bool(browser_compatibility_note(row["payload"], row["context"], row["family"]))
                  for row in rows)
    print("Chua xu ly/ngoai pham vi Chromium:", ", ".join(unsupported) or "khong")
    print(f"Vector known_browser_patched (encoded-whitespace javascript scheme): {patched}")
    if all_zero:
        raise RuntimeError(
            "Event co plan supported nhung mau that 100% valid=0; can sua truoc khi ket luan: "
            + ", ".join(all_zero)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--per-event", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--concurrent-size", type=int, default=60)
    parser.add_argument("--workdir",
                        help="Thu muc artifact; mac dinh tao verification_artifacts/run-<UTC timestamp>")
    parser.add_argument("--clean-workdir", action="store_true",
                        help="Cho phep xoa --workdir neu no da ton tai")
    parser.add_argument("--verbose-failures", action="store_true",
                        help="In payload, event, action va error cho moi valid=0 trong JSONL")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
