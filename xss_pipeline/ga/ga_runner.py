"""GA runner + Random/Bruteforce baselines voi ModelPredictor that va OracleCache.

Ho tro:
- 3 che do: --mode {ga,random,bruteforce}
- 3 mo hinh: --model {rf,svm,cnn_lstm}
- OracleCache dung chung (ga/oracle_cache.jsonl) khoa (context, family, event, carrier, payload)
- Chong trung lap 2 tang (Tang 1 trong tung the he, Tang 2 theo payload duy nhat cuoi run)
- Tinh Bypass Rate Cach A (unique_bypass / unique_valid) & Cach B (unique_bypass / unique_total)
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
import time
from pathlib import Path
from typing import Any, Callable

from ga.build import build_payload, event_for_individual, mechanism_for_individual
from ga.genes import GENE_POOLS

MAX_DEDUP_RETRIES = 15


def random_individual(context: str, rng: random.Random | None = None) -> dict[str, str]:
    chooser = rng if rng is not None else random
    pool = GENE_POOLS[context]
    return {gene: chooser.choice(values) for gene, values in pool.items()}


def mutate(ind: dict[str, str], context: str, rng: random.Random | None = None) -> dict[str, str]:
    chooser = rng if rng is not None else random
    pool = GENE_POOLS[context]
    gene = chooser.choice(list(ind.keys()))
    values = pool[gene]
    new_ind = dict(ind)
    if len(values) > 1:
        candidates = [v for v in values if v != ind[gene]]
        new_ind[gene] = chooser.choice(candidates)
    else:
        new_ind[gene] = values[0]
    return new_ind


def crossover(
    a: dict[str, str],
    b: dict[str, str],
    rng: random.Random | None = None,
) -> dict[str, str]:
    chooser = rng if rng is not None else random
    return {gene: chooser.choice([a[gene], b[gene]]) for gene in a}


def family_for_individual(context: str, individual: dict[str, Any]) -> str:
    """Anh xa gene cua ca the GA sang family khop cac nhanh sink trong app.py."""
    escape = str(individual.get("escape", ""))
    trigger = str(individual.get("trigger", ""))
    if context == "html_body":
        if trigger == "javascript:":
            return "anchor_js_ga"
        return ""
    if context == "url":
        target_tag = str(individual.get("target_tag", "a"))
        if target_tag == "a":
            return "url_js_ga"
        if target_tag == "iframe":
            return "iframe_srcdoc_ga"
        return "object_data_ga"
    if context == "attribute":
        return "attr_break_squote_ga" if "'" in escape else "attr_break_dquote_ga"
    if context == "js_string":
        stripped = escape.lstrip()
        if not stripped:
            return "js_direct_code_no_quote_ga"
        if stripped.startswith("*/"):
            return "js_break_comment_ga"
        if stripped.startswith(("`", "${")):
            return "js_break_template_ga"
        if stripped.startswith('"'):
            return "js_break_dquote_ga"
        if stripped.startswith("'"):
            return "js_break_squote_ga"
        return "js_direct_code_no_quote_ga"
    return ""


class OracleCache:
    """Cache JSONL dung chung khoa bang (context, family, event, carrier, payload)."""

    def __init__(self, cache_path: str | Path = "ga/oracle_cache.jsonl") -> None:
        self.path = Path(cache_path)
        self._store: dict[tuple[str, str, str, str, str], int] = {}
        self._load()

    @staticmethod
    def make_key(
        context: str,
        family: str,
        event: str,
        carrier: str,
        payload: str,
    ) -> tuple[str, str, str, str, str]:
        return (str(context), str(family), str(event), str(carrier), str(payload))

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rec = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                if not {"context", "family", "event", "carrier", "payload", "valid"}.issubset(rec.keys()):
                    continue
                key = self.make_key(
                    rec["context"],
                    rec["family"],
                    rec["event"],
                    rec["carrier"],
                    rec["payload"],
                )
                self._store[key] = 1 if int(rec["valid"]) else 0

    def get(self, key: tuple[str, str, str, str, str]) -> int | None:
        return self._store.get(key)

    def put_many(self, entries: list[tuple[tuple[str, str, str, str, str], int]]) -> None:
        if not entries:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for key, valid in entries:
                norm_valid = 1 if int(valid) else 0
                self._store[key] = norm_valid
                record = {
                    "context": key[0],
                    "family": key[1],
                    "event": key[2],
                    "carrier": key[3],
                    "payload": key[4],
                    "valid": norm_valid,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    def __len__(self) -> int:
        return len(self._store)


def compute_bypass_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Tang 2: khu trung lap theo payload duy nhat de tinh Bypass Rate Cach A & B."""
    dedup_by_payload: dict[str, dict[str, Any]] = {}
    for rec in records:
        payload = str(rec["payload"])
        existing = dedup_by_payload.get(payload)
        if existing is None:
            dedup_by_payload[payload] = dict(rec)
            continue
        # Neu cung payload xuat hien voi nhieu carrier/lan danh gia, uu tien ban co valid=1 va fitness cao hon.
        cur_score = (int(rec.get("valid", 0)), float(rec.get("fitness", 0.0)))
        old_score = (int(existing.get("valid", 0)), float(existing.get("fitness", 0.0)))
        if cur_score > old_score:
            dedup_by_payload[payload] = dict(rec)

    unique_records = list(dedup_by_payload.values())
    unique_total = len(unique_records)
    unique_valid = sum(1 for r in unique_records if int(r["valid"]) == 1)
    unique_bypass = sum(
        1
        for r in unique_records
        if int(r["valid"]) == 1 and float(r["p_malicious"]) < float(r["threshold"])
    )
    bypass_rate_a = (unique_bypass / unique_valid) if unique_valid > 0 else 0.0
    bypass_rate_b = (unique_bypass / unique_total) if unique_total > 0 else 0.0
    return {
        "unique_records": unique_records,
        "unique_total": unique_total,
        "unique_valid": unique_valid,
        "unique_bypass": unique_bypass,
        "bypass_rate_a": bypass_rate_a,
        "bypass_rate_b": bypass_rate_b,
    }


def _deduplicate_individual(
    ind: dict[str, str],
    context: str,
    seen_payloads: set[str],
    rng: random.Random,
    max_retries: int = MAX_DEDUP_RETRIES,
) -> tuple[dict[str, str], str]:
    payload = build_payload(context, **ind)
    retries = 0
    while payload in seen_payloads and retries < max_retries:
        ind = mutate(ind, context, rng)
        payload = build_payload(context, **ind)
        retries += 1
    seen_payloads.add(payload)
    return ind, payload


def _tournament_select(
    population: list[dict[str, str]],
    fitnesses: list[float],
    k: int,
    rng: random.Random,
) -> dict[str, str]:
    sample_size = max(1, min(k, len(population)))
    indices = rng.sample(range(len(population)), sample_size)
    best_idx = max(indices, key=lambda idx: fitnesses[idx])
    return population[best_idx]


def _enumerate_cartesian(context: str) -> list[dict[str, str]]:
    pool = GENE_POOLS[context]
    genes = list(pool.keys())
    return [dict(zip(genes, combo)) for combo in itertools.product(*(pool[g] for g in genes))]


async def evaluate_candidates(
    individuals: list[dict[str, str]],
    context: str,
    predictor: Any,
    oracle_cache: OracleCache,
    check_batch_fn: Callable[..., Any],
    concurrency: int,
    oracle_log_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Danh gia lo ca the qua OracleCache va ModelPredictor."""
    prepared: list[dict[str, Any]] = []
    uncached_cases: list[dict[str, Any]] = []
    uncached_keys: list[tuple[str, str, str, str, str]] = []
    uncached_seen: set[tuple[str, str, str, str, str]] = set()
    cache_hits = 0

    for idx, ind in enumerate(individuals):
        payload = build_payload(context, **ind)
        mechanism = mechanism_for_individual(context, ind["trigger"])
        event = event_for_individual(context, ind["trigger"])
        family = family_for_individual(context, ind)
        carrier = ind["target_tag"] if context == "url" else ""
        key = OracleCache.make_key(context, family, event, carrier, payload)
        cached_valid = oracle_cache.get(key)

        if cached_valid is not None:
            cache_hits += 1
        elif key not in uncached_seen:
            uncached_seen.add(key)
            uncached_keys.append(key)
            uncached_cases.append(
                {
                    "id": f"ga-{context}-{idx}-{abs(hash(key))}",
                    "payload": payload,
                    "context": context,
                    "event": event,
                    "family": family,
                    "target_tag": ind["target_tag"],
                }
            )

        prepared.append(
            {
                "ind": dict(ind),
                "key": key,
                "payload": payload,
                "mechanism": mechanism,
                "family": family,
                "event": event,
                "carrier": carrier,
            }
        )

    oracle_calls = len(uncached_cases)
    if uncached_cases:
        valid_flags = await check_batch_fn(
            uncached_cases,
            concurrency=concurrency,
            log_path=str(oracle_log_path) if oracle_log_path else None,
        )
        oracle_cache.put_many(list(zip(uncached_keys, valid_flags)))

    payloads = [item["payload"] for item in prepared]
    probs = predictor.predict_proba(payloads) if payloads else []
    threshold = float(predictor.threshold)

    records: list[dict[str, Any]] = []
    for item, prob in zip(prepared, probs):
        valid = oracle_cache.get(item["key"])
        valid_int = 1 if valid else 0
        p_malicious = float(prob)
        bypass = int(valid_int == 1 and p_malicious < threshold)
        fitness = float(valid_int) * (1.0 - p_malicious)
        ind = item["ind"]
        records.append(
            {
                "context": context,
                "mechanism": item["mechanism"],
                "family": item["family"],
                "event": item["event"],
                "carrier": item["carrier"],
                "escape": ind["escape"],
                "trigger": ind["trigger"],
                "action": ind["action"],
                "target_tag": ind["target_tag"],
                "payload": item["payload"],
                "valid": valid_int,
                "p_malicious": p_malicious,
                "threshold": threshold,
                "bypass": bypass,
                "fitness": fitness,
            }
        )

    return records, cache_hits, oracle_calls


def _summarize_generation(
    gen_num: int,
    records: list[dict[str, Any]],
    elapsed_s: float,
    cache_hits: int,
    oracle_calls: int,
) -> dict[str, Any]:
    n = len(records)
    fitnesses = [float(r["fitness"]) for r in records]
    valid_count = sum(int(r["valid"]) for r in records)
    return {
        "gen": gen_num,
        "population_size": n,
        "valid_rate": (valid_count / n) if n else 0.0,
        "best_fitness": max(fitnesses) if fitnesses else 0.0,
        "mean_fitness": (sum(fitnesses) / n) if n else 0.0,
        "elapsed_s": round(elapsed_s, 3),
        "cache_hits": cache_hits,
        "oracle_calls": oracle_calls,
    }


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


async def _run_ga(
    args: argparse.Namespace,
    predictor: Any,
    oracle_cache: OracleCache,
    check_batch_fn: Callable[..., Any],
    out_dir: Path,
    oracle_log_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(args.seed)
    ckpt_path = out_dir / f"checkpoint_{args.mode}_{args.model}_{args.context}_seed{args.seed}.json"
    ckpt = _load_json(ckpt_path)

    if ckpt and not ckpt.get("completed", False):
        start_gen = int(ckpt["next_gen"])
        population = ckpt["population"]
        history = ckpt["history"]
        all_records = ckpt["all_records"]
        best_overall = float(ckpt["best_overall"])
        no_improve = int(ckpt["no_improve"])
        rng_state = ckpt.get("rng_state")
        if rng_state is not None:
            # Khoi phuc tuple cho random.setstate tu JSON list.
            rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))
        print(f"Resume GA tu checkpoint: tiep tuc the he {start_gen + 1}/{args.generations}")
    else:
        start_gen = 0
        history = []
        all_records = []
        best_overall = -1.0
        no_improve = 0
        seen_init: set[str] = set()
        population = []
        for _ in range(args.population):
            ind = random_individual(args.context, rng)
            ind, _ = _deduplicate_individual(ind, args.context, seen_init, rng)
            population.append(ind)

    for gen in range(start_gen, args.generations):
        t0 = time.perf_counter()
        records, cache_hits, oracle_calls = await evaluate_candidates(
            population,
            args.context,
            predictor,
            oracle_cache,
            check_batch_fn,
            args.concurrency,
            oracle_log_path,
        )
        elapsed_s = time.perf_counter() - t0
        gen_stats = _summarize_generation(gen + 1, records, elapsed_s, cache_hits, oracle_calls)
        history.append(gen_stats)
        all_records.extend(records)

        best_gen = gen_stats["best_fitness"]
        if best_gen > best_overall + 1e-12:
            best_overall = best_gen
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"[GA {args.model} {args.context} seed={args.seed}] Gen {gen + 1}/{args.generations}: "
            f"valid={gen_stats['valid_rate']:.3f}, best={best_gen:.4f}, "
            f"mean={gen_stats['mean_fitness']:.4f}, hits={cache_hits}, calls={oracle_calls}, "
            f"t={elapsed_s:.2f}s"
        )

        stop_early = args.patience > 0 and no_improve >= args.patience
        is_last = (gen == args.generations - 1) or stop_early

        if not is_last:
            fitnesses = [float(r["fitness"]) for r in records]
            ranked = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
            elite_count = max(1, round(args.elitism_ratio * len(population))) if args.elitism_ratio > 0 else 0
            elite_count = min(elite_count, len(population))

            next_population: list[dict[str, str]] = [dict(ind) for ind, _ in ranked[:elite_count]]
            seen_next: set[str] = {build_payload(args.context, **ind) for ind in next_population}

            while len(next_population) < args.population:
                p1 = _tournament_select(population, fitnesses, args.tournament_k, rng)
                p2 = _tournament_select(population, fitnesses, args.tournament_k, rng)
                child = crossover(p1, p2, rng) if rng.random() < args.crossover_rate else dict(p1)
                if rng.random() < args.mutation_rate:
                    child = mutate(child, args.context, rng)
                child, _ = _deduplicate_individual(child, args.context, seen_next, rng)
                next_population.append(child)

            population = next_population

        state = rng.getstate()
        _save_json(
            ckpt_path,
            {
                "completed": is_last,
                "next_gen": gen + 1,
                "population": population,
                "history": history,
                "all_records": all_records,
                "best_overall": best_overall,
                "no_improve": no_improve,
                "rng_state": [state[0], list(state[1]), state[2]],
            },
        )

        if stop_early:
            print(f"Early stop tai Gen {gen + 1}: khong cai thien qua {args.patience} the he.")
            break

    return history, all_records


async def _run_batch_baseline(
    args: argparse.Namespace,
    candidates: list[dict[str, str]],
    predictor: Any,
    oracle_cache: OracleCache,
    check_batch_fn: Callable[..., Any],
    oracle_log_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunk_size = max(1, args.population)
    history: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []

    total_chunks = (len(candidates) + chunk_size - 1) // chunk_size if candidates else 0
    for chunk_idx in range(total_chunks):
        batch = candidates[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
        t0 = time.perf_counter()
        records, cache_hits, oracle_calls = await evaluate_candidates(
            batch,
            args.context,
            predictor,
            oracle_cache,
            check_batch_fn,
            args.concurrency,
            oracle_log_path,
        )
        elapsed_s = time.perf_counter() - t0
        gen_stats = _summarize_generation(chunk_idx + 1, records, elapsed_s, cache_hits, oracle_calls)
        history.append(gen_stats)
        all_records.extend(records)
        print(
            f"[{args.mode.upper()} {args.model} {args.context} seed={args.seed}] "
            f"Batch {chunk_idx + 1}/{total_chunks} ({len(batch)}): "
            f"valid={gen_stats['valid_rate']:.3f}, best={gen_stats['best_fitness']:.4f}, "
            f"hits={cache_hits}, calls={oracle_calls}, t={elapsed_s:.2f}s"
        )

    return history, all_records


def select_random_candidates(context: str, seed: int, budget: int) -> list[dict[str, str]]:
    """Chon ngau nhien toi da ``budget`` ca the co chuoi payload duy nhat."""
    rng = random.Random(seed)
    all_cartesian = _enumerate_cartesian(context)
    rng.shuffle(all_cartesian)
    seen_payloads: set[str] = set()
    selected: list[dict[str, str]] = []
    for ind in all_cartesian:
        payload = build_payload(context, **ind)
        if payload in seen_payloads:
            continue
        seen_payloads.add(payload)
        selected.append(ind)
        if len(selected) >= budget:
            break
    return selected


async def run(
    args: argparse.Namespace,
    *,
    predictor: Any | None = None,
    oracle_cache: OracleCache | None = None,
    check_batch_fn: Callable[..., Any] | None = None,
) -> Path:
    if predictor is None:
        from models.predictor import ModelPredictor

        predictor = ModelPredictor(model_name=args.model)
    if check_batch_fn is None:
        from oracle import check_batch

        check_batch_fn = check_batch
    if oracle_cache is None:
        oracle_cache = OracleCache(args.cache_path)

    t_start = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle_log_path = out_dir / f"oracle_log_{args.mode}_{args.model}_{args.context}_seed{args.seed}.jsonl"

    budget_arg = getattr(args, "budget", None)
    resolved_budget: int | None = None

    if args.mode == "ga":
        history, all_records = await _run_ga(
            args, predictor, oracle_cache, check_batch_fn, out_dir, oracle_log_path
        )
    elif args.mode == "random":
        resolved_budget = (
            int(budget_arg)
            if (budget_arg is not None and int(budget_arg) > 0)
            else int(args.population) * int(args.generations)
        )
        selected = select_random_candidates(args.context, args.seed, resolved_budget)
        history, all_records = await _run_batch_baseline(
            args, selected, predictor, oracle_cache, check_batch_fn, oracle_log_path
        )
    elif args.mode == "bruteforce":
        all_cartesian = _enumerate_cartesian(args.context)
        seen_keys: set[tuple[str, str, str, str]] = set()
        selected = []
        for ind in all_cartesian:
            payload = build_payload(args.context, **ind)
            family = family_for_individual(args.context, ind)
            event = event_for_individual(args.context, ind["trigger"])
            carrier = ind["target_tag"] if args.context == "url" else ""
            dedup_key = (family, event, carrier, payload)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            selected.append(ind)
        history, all_records = await _run_batch_baseline(
            args, selected, predictor, oracle_cache, check_batch_fn, oracle_log_path
        )
    else:
        raise ValueError(f"Che do khong hop le: {args.mode}")

    metrics = compute_bypass_metrics(all_records)
    total_elapsed = time.perf_counter() - t_start
    total_cache_hits = sum(int(h["cache_hits"]) for h in history)
    total_oracle_calls = sum(int(h["oracle_calls"]) for h in history)

    output_path = out_dir / f"{args.mode}_{args.model}_{args.context}_seed{args.seed}.json"
    result_payload: dict[str, Any] = {
        "config": {
            "mode": args.mode,
            "model": args.model,
            "context": args.context,
            "seed": args.seed,
            "threshold": float(predictor.threshold),
            "population": args.population,
            "generations": args.generations,
            "budget": resolved_budget if resolved_budget is not None else budget_arg,
            "crossover_rate": args.crossover_rate,
            "mutation_rate": args.mutation_rate,
            "tournament_k": args.tournament_k,
            "elitism_ratio": args.elitism_ratio,
            "patience": args.patience,
            "concurrency": args.concurrency,
            "cache_path": str(args.cache_path),
        },
        "summary": {
            "unique_total": metrics["unique_total"],
            "unique_valid": metrics["unique_valid"],
            "unique_bypass": metrics["unique_bypass"],
            "bypass_rate_a": round(metrics["bypass_rate_a"], 6),
            "bypass_rate_b": round(metrics["bypass_rate_b"], 6),
            "total_evaluated": len(all_records),
            "total_cache_hits": total_cache_hits,
            "total_oracle_calls": total_oracle_calls,
            "generations_completed": len(history),
            "total_elapsed_s": round(total_elapsed, 3),
        },
        "history": history,
        "unique_individuals": metrics["unique_records"],
    }
    if budget_arg is not None:
        result_payload["budget"] = int(budget_arg)
    _save_json(output_path, result_payload)
    print(
        f"Hoan tat {output_path}: unique_total={metrics['unique_total']}, "
        f"unique_valid={metrics['unique_valid']}, unique_bypass={metrics['unique_bypass']}, "
        f"RateA={metrics['bypass_rate_a']:.4f}, RateB={metrics['bypass_rate_b']:.4f}"
    )
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GA runner cho sinh payload XSS")
    parser.add_argument(
        "--context",
        required=True,
        choices=["html_body", "attribute", "js_string", "url"],
    )
    parser.add_argument(
        "--model",
        choices=["rf", "svm", "cnn_lstm"],
        default="rf",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["ga", "random", "bruteforce"],
        default="ga",
    )
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--crossover-rate", type=float, default=0.75)
    parser.add_argument("--mutation-rate", type=float, default=0.10)
    parser.add_argument("--tournament-k", type=int, default=3)
    parser.add_argument("--elitism-ratio", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--cache-path", default="ga/oracle_cache.jsonl")
    parser.add_argument("--out-dir", default="ga/results")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
