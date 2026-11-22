"""Kiem ke khong gian tim kiem GA cho ca 4 ngu canh (thuan Python).

Chay:
    python3 -m ga.inspect_space
"""

from __future__ import annotations

import itertools

from event_actions import EVENT_ACTIONS
from ga.build import build_payload, event_for_individual, mechanism_for_individual
from ga.genes import GENE_POOLS

CONTEXT_ORDER = ("html_body", "attribute", "js_string", "url")
MECHANISMS = ("html_event", "js_protocol", "js_direct")
EXPECTED_NA = {
    "html_body": frozenset(),
    "attribute": frozenset({"js_protocol", "js_direct"}),
    "js_string": frozenset({"js_protocol"}),
    "url": frozenset({"html_event", "js_direct"}),
}


def inspect_context(context: str) -> dict:
    pool = GENE_POOLS[context]
    genes = ("escape", "trigger", "action", "target_tag")
    counts = {g: len(pool[g]) for g in genes}

    # Kiem tra rang buoc gen dau vao.
    for g in genes:
        for val in pool[g]:
            assert "__markXSS" not in val, f"Phat hien __markXSS trong {context}.{g}: {val!r}"
    for act in pool["action"]:
        assert '"' not in act, f"action chua nhay kep '\"' trong {context}: {act!r}"
    for trig in pool["trigger"]:
        ev = event_for_individual(context, trig)
        if ev != "none":
            assert ev in EVENT_ACTIONS, f"Event {ev!r} khong co trong EVENT_ACTIONS"
            assert EVENT_ACTIONS[ev].supported is True, f"Event {ev!r} khong supported=True"

    cartesian_total = 0
    unique_payloads: set[str] = set()
    mech_cartesian = {m: 0 for m in MECHANISMS}
    mech_unique: dict[str, set[str]] = {m: set() for m in MECHANISMS}

    for escape, trigger, action, target_tag in itertools.product(
        pool["escape"], pool["trigger"], pool["action"], pool["target_tag"]
    ):
        cartesian_total += 1
        payload = build_payload(context, escape, trigger, action, target_tag)
        assert "__markXSS" not in payload, f"Phat hien __markXSS trong payload: {payload!r}"
        mech = mechanism_for_individual(context, trigger)
        mech_cartesian[mech] += 1
        mech_unique[mech].add(payload)
        unique_payloads.add(payload)

    for na_mech in EXPECTED_NA[context]:
        assert mech_cartesian[na_mech] == 0, f"{context}.{na_mech} phai la 0 (N/A), got {mech_cartesian[na_mech]}"
        assert len(mech_unique[na_mech]) == 0

    return {
        "context": context,
        "counts": counts,
        "cartesian_total": cartesian_total,
        "unique_total": len(unique_payloads),
        "mech_cartesian": mech_cartesian,
        "mech_unique": {m: len(s) for m, s in mech_unique.items()},
    }


def main() -> None:
    rows = [inspect_context(ctx) for ctx in CONTEXT_ORDER]
    header = (
        f"{'context':10} | {'esc':>3} {'trg':>3} {'act':>3} {'tag':>3} | "
        f"{'cartesian':>9} | {'|S| (uniq)':>10} | "
        f"{'html_event':>10} {'js_protocol':>11} {'js_direct':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        c = r["counts"]
        mu = r["mech_unique"]
        print(
            f"{r['context']:10} | {c['escape']:>3} {c['trigger']:>3} {c['action']:>3} {c['target_tag']:>3} | "
            f"{r['cartesian_total']:>9} | {r['unique_total']:>10} | "
            f"{mu['html_event']:>10} {mu['js_protocol']:>11} {mu['js_direct']:>9}"
        )

    print("\nPhan bo to hop Descartes theo mechanism (kiem chung Ma tran 6.2 - o N/A = 0):")
    for r in rows:
        mc = r["mech_cartesian"]
        print(
            f"  {r['context']:10}: html_event={mc['html_event']}, "
            f"js_protocol={mc['js_protocol']}, js_direct={mc['js_direct']}"
        )
    print("\nALL SPACE ASSERTIONS PASSED.")


if __name__ == "__main__":
    main()
