"""Script dieu phoi Full Run Giai doan 3 (3 mo hinh x 4 ngu canh x 5 seed + Baseline).

Chay tren terminal host (sau khi da mo app.py o terminal khac):
    ./venv/bin/python -m ga.run_phase3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

from ga.ga_runner import OracleCache, run as run_ga_task

LARGE_CONTEXTS = ("html_body", "attribute", "js_string")
METRICS = (
    "unique_total",
    "unique_valid",
    "unique_bypass",
    "bypass_rate_a",
    "bypass_rate_b",
)


def _try_load_existing_result(path: Path) -> dict[str, Any] | None:
    """Doc file ket qua JSON neu da ton tai va hop le de ho tro resume."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    summary = data.get("summary")
    if not isinstance(summary, dict) or "unique_total" not in summary:
        return None
    return data


def _mean_std_ddof1(values: list[float]) -> tuple[float, float | None]:
    if not values:
        return 0.0, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def summarize_phase3_groups(
    grouped_runs: dict[tuple[str, str, str], list[tuple[int, dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """Tinh Mean +- Std (ddof=1) cho tung nhom (model, context, mode)."""
    summaries: list[dict[str, Any]] = []
    for (model, context, mode), items in grouped_runs.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        seeds = [seed for seed, _ in items_sorted]
        metric_stats: dict[str, dict[str, float | None]] = {}
        for metric in METRICS:
            vals = [float(res["summary"][metric]) for _, res in items_sorted]
            mean, std = _mean_std_ddof1(vals)
            metric_stats[metric] = {
                "mean": round(mean, 6),
                "std": round(std, 6) if std is not None else None,
            }
        summaries.append(
            {
                "model": model,
                "context": context,
                "mode": mode,
                "n_runs": len(items_sorted),
                "seeds": seeds,
                "metrics": metric_stats,
                "per_seed": [
                    {"seed": seed, **res["summary"]} for seed, res in items_sorted
                ],
            }
        )
    return summaries


def _format_stat(stat: dict[str, float | None], *, is_rate: bool = False) -> str:
    mean = float(stat["mean"] or 0.0)
    std = stat["std"]
    if is_rate:
        if std is None:
            return f"{mean * 100:6.2f}% (N/A)"
        return f"{mean * 100:6.2f}% ± {std * 100:5.2f}%"
    if std is None:
        return f"{mean:7.1f} (N/A)"
    return f"{mean:7.1f} ± {std:5.1f}"


def _print_summary_table(
    group_summaries: list[dict[str, Any]],
    cache_size: int,
    total_elapsed_s: float,
) -> None:
    print("\n" + "=" * 124)
    print("TONG HOP KET QUA GIAI DOAN 3 (Mean ± Std, ddof=1)")
    print("=" * 124)
    header = (
        f"{'model':8} | {'context':10} | {'mode':10} | {'n':>2} | "
        f"{'unique_total':15} | {'unique_valid':15} | {'unique_bypass':15} | "
        f"{'Rate A (primary)':17} | {'Rate B (secondary)':17}"
    )
    print(header)
    print("-" * len(header))
    for g in group_summaries:
        m = g["metrics"]
        print(
            f"{g['model']:8} | {g['context']:10} | {g['mode']:10} | {g['n_runs']:>2} | "
            f"{_format_stat(m['unique_total']):15} | "
            f"{_format_stat(m['unique_valid']):15} | "
            f"{_format_stat(m['unique_bypass']):15} | "
            f"{_format_stat(m['bypass_rate_a'], is_rate=True):17} | "
            f"{_format_stat(m['bypass_rate_b'], is_rate=True):17}"
        )
    print("=" * 124)
    print(
        f"OracleCache: {cache_size} muc da luu | Tong thoi gian phien: {total_elapsed_s:.2f}s"
    )


def _make_runner_args(
    args: argparse.Namespace,
    *,
    model: str,
    context: str,
    mode: str,
    seed: int,
    budget: int | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        context=context,
        model=model,
        seed=seed,
        mode=mode,
        population=args.population,
        generations=args.generations,
        budget=budget,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        tournament_k=args.tournament_k,
        elitism_ratio=args.elitism_ratio,
        patience=args.patience,
        concurrency=args.concurrency,
        cache_path=args.cache_path,
        out_dir=args.out_dir,
    )


async def run_phase3(args: argparse.Namespace) -> Path:
    t_session_start = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oracle_cache = OracleCache(args.cache_path)
    print(f"Khoi dong Phase 3 Full Run | OracleCache hien co {len(oracle_cache)} muc tai {args.cache_path}")

    grouped_runs: dict[tuple[str, str, str], list[tuple[int, dict[str, Any]]]] = {}

    for model_name in args.models:
        predictor: Any | None = None

        def get_predictor() -> Any:
            nonlocal predictor
            if predictor is None:
                from models.predictor import ModelPredictor

                print(f"\n>>> Load ModelPredictor(model_name={model_name!r}) ...")
                predictor = ModelPredictor(model_name=model_name)
            return predictor

        async def run_or_resume(
            *,
            context: str,
            mode: str,
            seed: int,
            budget: int | None = None,
        ) -> dict[str, Any]:
            result_path = out_dir / f"{mode}_{model_name}_{context}_seed{seed}.json"
            existing = _try_load_existing_result(result_path)
            if existing is not None:
                s = existing["summary"]
                print(
                    f"[RESUME] Bo qua {result_path.name} "
                    f"(unique_total={s['unique_total']}, valid={s['unique_valid']}, "
                    f"bypass={s['unique_bypass']}, RateA={s['bypass_rate_a']:.4f})"
                )
                return existing

            runner_args = _make_runner_args(
                args,
                model=model_name,
                context=context,
                mode=mode,
                seed=seed,
                budget=budget,
            )
            await run_ga_task(
                runner_args,
                predictor=get_predictor(),
                oracle_cache=oracle_cache,
            )
            loaded = _try_load_existing_result(result_path)
            if loaded is None:
                raise RuntimeError(f"Khong doc duoc file ket qua sau khi chay: {result_path}")
            return loaded

        for context in args.contexts:
            if context in LARGE_CONTEXTS:
                ga_items: list[tuple[int, dict[str, Any]]] = []
                random_items: list[tuple[int, dict[str, Any]]] = []
                for seed in args.seeds:
                    ga_res = await run_or_resume(context=context, mode="ga", seed=seed)
                    ga_items.append((seed, ga_res))

                    ga_unique_total = int(ga_res["summary"]["unique_total"])
                    rand_res = await run_or_resume(
                        context=context,
                        mode="random",
                        seed=seed,
                        budget=ga_unique_total,
                    )
                    random_items.append((seed, rand_res))

                grouped_runs[(model_name, context, "ga")] = ga_items
                grouped_runs[(model_name, context, "random")] = random_items
            elif context == "url":
                bf_res = await run_or_resume(context="url", mode="bruteforce", seed=0)
                grouped_runs[(model_name, "url", "bruteforce")] = [(0, bf_res)]

    total_elapsed_s = time.perf_counter() - t_session_start
    group_summaries = summarize_phase3_groups(grouped_runs)
    _print_summary_table(group_summaries, len(oracle_cache), total_elapsed_s)

    summary_path = out_dir / "phase3_summary.json"
    summary_payload = {
        "config": {
            "models": list(args.models),
            "contexts": list(args.contexts),
            "seeds": list(args.seeds),
            "population": args.population,
            "generations": args.generations,
            "patience": args.patience,
            "crossover_rate": args.crossover_rate,
            "mutation_rate": args.mutation_rate,
            "tournament_k": args.tournament_k,
            "elitism_ratio": args.elitism_ratio,
            "concurrency": args.concurrency,
            "cache_path": str(args.cache_path),
            "out_dir": str(args.out_dir),
        },
        "cache_entries_total": len(oracle_cache),
        "total_elapsed_s": round(total_elapsed_s, 3),
        "groups": group_summaries,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)
    print(f"Da luu tong hop tai: {summary_path}")
    return summary_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dieu phoi Full Run Giai doan 3 (GA + Random/Bruteforce Baseline)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["rf", "svm", "cnn_lstm"],
        default=["rf", "svm", "cnn_lstm"],
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        choices=["html_body", "attribute", "js_string", "url"],
        default=["html_body", "attribute", "js_string", "url"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--population", type=int, default=60)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--crossover-rate", type=float, default=0.75)
    parser.add_argument("--mutation-rate", type=float, default=0.10)
    parser.add_argument("--tournament-k", type=int, default=3)
    parser.add_argument("--elitism-ratio", type=float, default=0.05)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--cache-path", default="ga/oracle_cache.jsonl")
    parser.add_argument("--out-dir", default="ga/results_full")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(run_phase3(args))


if __name__ == "__main__":
    main()
