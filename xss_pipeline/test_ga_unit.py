"""Unit tests thuan Python cho kho gen, build payload, OracleCache va GA runner."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from event_actions import EVENT_ACTIONS
from ga.build import build_payload, event_for_individual, mechanism_for_individual
from ga.ga_runner import (
    OracleCache,
    build_arg_parser,
    compute_bypass_metrics,
    evaluate_candidates,
    family_for_individual,
    run as run_ga,
)
from ga.genes import GENE_POOLS
from ga.run_phase3 import _try_load_existing_result, summarize_phase3_groups


class _MockPredictor:
    def __init__(self, probs_by_payload: dict[str, float] | None = None, threshold: float = 0.5) -> None:
        self._probs = probs_by_payload or {}
        self.threshold = threshold

    def predict_proba(self, payloads: list[str]) -> list[float]:
        return [self._probs.get(p, 0.25) for p in payloads]


class TestGenePools(unittest.TestCase):
    def test_no_markxss_and_safe_actions(self) -> None:
        for ctx, pool in GENE_POOLS.items():
            self.assertEqual(set(pool.keys()), {"escape", "trigger", "action", "target_tag"})
            for gene_name, values in pool.items():
                self.assertTrue(values, f"{ctx}.{gene_name} must not be empty")
                for val in values:
                    self.assertNotIn("__markXSS", val, f"Found __markXSS in {ctx}.{gene_name}")
            for act in pool["action"]:
                self.assertNotIn('"', act, f"Action contains double quote in {ctx}: {act!r}")
            for trig in pool["trigger"]:
                ev = event_for_individual(ctx, trig)
                if ev != "none":
                    self.assertIn(ev, EVENT_ACTIONS)
                    self.assertTrue(EVENT_ACTIONS[ev].supported, f"{ev} must be supported")


class TestBuildAndMappings(unittest.TestCase):
    def test_mechanism_and_event_helpers(self) -> None:
        self.assertEqual(mechanism_for_individual("html_body", "onerror"), "html_event")
        self.assertEqual(mechanism_for_individual("html_body", "javascript:"), "js_protocol")
        self.assertEqual(mechanism_for_individual("js_string", "direct"), "js_direct")
        self.assertEqual(event_for_individual("html_body", "javascript:"), "none")
        self.assertEqual(event_for_individual("js_string", "direct"), "none")
        self.assertEqual(event_for_individual("attribute", "onfocus"), "onfocus")

    def test_html_body_variants(self) -> None:
        # javascript: -> anchor_js_ga
        self.assertEqual(
            family_for_individual("html_body", {"escape": "", "trigger": "javascript:"}),
            "anchor_js_ga",
        )
        self.assertEqual(
            build_payload("html_body", "", "javascript:", "alert(1)", "a"),
            '<a href="javascript:alert(1)">click</a>',
        )
        self.assertEqual(
            build_payload("html_body", "", "javascript:", "alert(1)", "iframe"),
            '<iframe src="javascript:alert(1)"></iframe>',
        )
        # direct
        self.assertEqual(
            build_payload("html_body", "", "direct", "alert(1)", "script"),
            "<script>alert(1)</script>",
        )
        self.assertEqual(
            build_payload("html_body", "", "direct", "alert(1)", "img"),
            "<img><script>alert(1)</script></img>",
        )
        # on* special rules
        p_err = build_payload("html_body", "", "onerror", "fetch('/probe')", "img")
        self.assertIn('src="/__oracle_missing_resource__"', p_err)
        self.assertIn('onerror="fetch(\'/probe\')"', p_err)

        p_load = build_payload("html_body", "", "onload", "alert(1)", "img")
        self.assertIn('src="/validimage.gif"', p_load)

        p_toggle = build_payload("html_body", "", "ontoggle", "alert(1)", "details")
        self.assertIn(" open ", p_toggle)

        p_focus_div = build_payload("html_body", "", "onfocus", "alert(1)", "svg")
        self.assertIn('tabindex="0"', p_focus_div)
        p_focus_inp = build_payload("html_body", "", "onfocus", "alert(1)", "input")
        self.assertNotIn("tabindex", p_focus_inp)

        p_begin = build_payload("html_body", "", "onbegin", "alert(1)", "svg")
        self.assertEqual(p_begin, '<svg><animate onbegin="alert(1)" attributeName="x" dur="1s"></svg>')

    def test_attribute_variants(self) -> None:
        self.assertEqual(family_for_individual("attribute", {"escape": "'"}), "attr_break_squote_ga")
        self.assertEqual(family_for_individual("attribute", {"escape": "' "}), "attr_break_squote_ga")
        self.assertEqual(family_for_individual("attribute", {"escape": '"'}), "attr_break_dquote_ga")
        self.assertEqual(family_for_individual("attribute", {"escape": '">'}), "attr_break_dquote_ga")

        self.assertEqual(
            build_payload("attribute", "'", "onfocus", "alert(1)", "input"),
            '\'><input onfocus="alert(1)">',
        )
        self.assertEqual(
            build_payload("attribute", '">', "onerror", "alert(1)", "img"),
            '"><img src="/__oracle_missing_resource__" onerror="alert(1)">',
        )
        self.assertEqual(
            build_payload("attribute", '" ', "onfocus", "alert(1)", "input"),
            '" onfocus="alert(1)"',
        )

    def test_js_string_and_url_variants(self) -> None:
        self.assertEqual(family_for_individual("js_string", {"escape": "';"}), "js_break_squote_ga")
        self.assertEqual(family_for_individual("js_string", {"escape": '";'}), "js_break_dquote_ga")
        self.assertEqual(family_for_individual("js_string", {"escape": "`;"}), "js_break_template_ga")
        self.assertEqual(family_for_individual("js_string", {"escape": "${"}), "js_break_template_ga")
        self.assertEqual(family_for_individual("js_string", {"escape": "*/;"}), "js_break_comment_ga")
        self.assertEqual(family_for_individual("js_string", {"escape": ""}), "js_direct_code_no_quote_ga")

        self.assertEqual(
            build_payload("js_string", "';", "direct", "alert(1)", "script"),
            "';alert(1)//",
        )
        self.assertEqual(
            build_payload("js_string", '";', "onerror", "alert(1)", "img"),
            '";</script><img src="/__oracle_missing_resource__" onerror="alert(1)">',
        )

        self.assertEqual(family_for_individual("url", {"escape": "", "target_tag": "a"}), "url_js_ga")
        self.assertEqual(family_for_individual("url", {"escape": "", "target_tag": "iframe"}), "iframe_srcdoc_ga")
        self.assertEqual(family_for_individual("url", {"escape": "", "target_tag": "object"}), "object_data_ga")
        self.assertEqual(
            build_payload("url", "", "javascript:", "alert(1)", "a"),
            "javascript:alert(1)",
        )
        self.assertEqual(
            build_payload("url", "", "javascript:", "alert(1)", "iframe"),
            "data:text/html,<script>alert(1)</script>",
        )


class TestOracleCacheAndMetrics(unittest.TestCase):
    def test_oracle_cache_save_load_and_corrupt_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "oracle_cache.jsonl"
            cache = OracleCache(cache_file)
            k1 = OracleCache.make_key("html_body", "", "onerror", "", "<img src=x onerror=alert(1)>")
            k2 = OracleCache.make_key("url", "iframe_srcdoc_ga", "none", "iframe", "data:text/html,<script>alert(1)</script>")
            k3 = OracleCache.make_key("url", "object_data_ga", "none", "object", "data:text/html,<script>alert(1)</script>")

            cache.put_many([(k1, 1), (k2, 1), (k3, 0)])
            # Gia lap dong JSON bi cat ngang hoac thieu truong.
            with cache_file.open("a", encoding="utf-8") as f:
                f.write('{"context": "html_body", "truncated"\n')
                f.write('{"context": "html_body", "payload": "missing_fields"}\n')
                f.write("\n")

            reloaded = OracleCache(cache_file)
            self.assertEqual(len(reloaded), 3)
            self.assertEqual(reloaded.get(k1), 1)
            self.assertEqual(reloaded.get(k2), 1)
            self.assertEqual(reloaded.get(k3), 0)
            self.assertIsNone(
                reloaded.get(OracleCache.make_key("attribute", "attr_break_dquote_ga", "onfocus", "", "x"))
            )

    def test_compute_bypass_metrics_dedup_and_rates(self) -> None:
        records = [
            # payload p1: valid=1, bypass=1 (xuat hien 2 lan -> tinh 1)
            {"payload": "p1", "valid": 1, "p_malicious": 0.2, "threshold": 0.5, "fitness": 0.8},
            {"payload": "p1", "valid": 1, "p_malicious": 0.2, "threshold": 0.5, "fitness": 0.8},
            # payload p2: valid=1, bypass=0 (p >= threshold)
            {"payload": "p2", "valid": 1, "p_malicious": 0.8, "threshold": 0.5, "fitness": 0.2},
            # payload p3: valid=0, p < threshold -> khong tinh bypass
            {"payload": "p3", "valid": 0, "p_malicious": 0.1, "threshold": 0.5, "fitness": 0.0},
            # payload p4: valid=1, bypass=1
            {"payload": "p4", "valid": 1, "p_malicious": 0.49, "threshold": 0.5, "fitness": 0.51},
        ]
        m = compute_bypass_metrics(records)
        self.assertEqual(m["unique_total"], 4)
        self.assertEqual(m["unique_valid"], 3)
        self.assertEqual(m["unique_bypass"], 2)
        self.assertAlmostEqual(m["bypass_rate_a"], 2 / 3)
        self.assertAlmostEqual(m["bypass_rate_b"], 2 / 4)

        empty_m = compute_bypass_metrics([])
        self.assertEqual(empty_m["bypass_rate_a"], 0.0)
        self.assertEqual(empty_m["bypass_rate_b"], 0.0)

    def test_evaluate_candidates_uses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OracleCache(Path(tmpdir) / "cache.jsonl")
            ind1 = {"escape": "", "trigger": "onerror", "action": "alert(1)", "target_tag": "img"}
            ind2 = {"escape": "", "trigger": "onload", "action": "alert(1)", "target_tag": "img"}
            p1 = build_payload("html_body", **ind1)
            p2 = build_payload("html_body", **ind2)

            predictor = _MockPredictor({p1: 0.3, p2: 0.7}, threshold=0.5)
            oracle_batches: list[list[dict]] = []

            async def fake_check_batch(cases, concurrency=8, log_path=None):
                del concurrency, log_path
                oracle_batches.append(cases)
                return [1 for _ in cases]

            # Lan 1: ca 2 chua co trong cache -> 2 oracle calls, 0 cache hits.
            recs1, hits1, calls1 = asyncio.run(
                evaluate_candidates([ind1, ind2], "html_body", predictor, cache, fake_check_batch, 4)
            )
            self.assertEqual(hits1, 0)
            self.assertEqual(calls1, 2)
            self.assertEqual(len(oracle_batches), 1)
            self.assertEqual(recs1[0]["bypass"], 1)
            self.assertEqual(recs1[1]["bypass"], 0)

            # Lan 2: ca 2 da co trong cache -> 2 cache hits, 0 oracle calls.
            recs2, hits2, calls2 = asyncio.run(
                evaluate_candidates([ind1, ind2], "html_body", predictor, cache, fake_check_batch, 4)
            )
            self.assertEqual(hits2, 2)
            self.assertEqual(calls2, 0)
            self.assertEqual(len(oracle_batches), 1)
            self.assertEqual(recs2[0]["valid"], 1)

    def test_random_mode_respects_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cache = OracleCache(tmp / "cache.jsonl")
            predictor = _MockPredictor(threshold=0.5)

            async def fake_check_batch(cases, concurrency=8, log_path=None):
                del concurrency, log_path
                return [1 for _ in cases]

            # 1. Explicit --budget nho hon |S| va khac population * generations.
            args = build_arg_parser().parse_args(
                [
                    "--context",
                    "html_body",
                    "--model",
                    "rf",
                    "--mode",
                    "random",
                    "--seed",
                    "3",
                    "--population",
                    "10",
                    "--generations",
                    "5",
                    "--budget",
                    "17",
                    "--cache-path",
                    str(tmp / "cache.jsonl"),
                    "--out-dir",
                    str(tmp / "out"),
                ]
            )
            out_path = asyncio.run(
                run_ga(args, predictor=predictor, oracle_cache=cache, check_batch_fn=fake_check_batch)
            )
            data = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(data["summary"]["unique_total"], 17)
            self.assertEqual(data["budget"], 17)
            self.assertEqual(data["config"]["budget"], 17)
            self.assertEqual(data["config"]["population"], 10)
            self.assertEqual(data["config"]["generations"], 5)
            self.assertIn("patience", data["config"])

            # 2. --budget lon hon |S| cua url (26) -> dung tai 26 payload duy nhat.
            args_url = build_arg_parser().parse_args(
                [
                    "--context",
                    "url",
                    "--model",
                    "rf",
                    "--mode",
                    "random",
                    "--seed",
                    "0",
                    "--budget",
                    "100",
                    "--cache-path",
                    str(tmp / "cache.jsonl"),
                    "--out-dir",
                    str(tmp / "out"),
                ]
            )
            out_url = asyncio.run(
                run_ga(args_url, predictor=predictor, oracle_cache=cache, check_batch_fn=fake_check_batch)
            )
            data_url = json.loads(out_url.read_text(encoding="utf-8"))
            self.assertEqual(data_url["summary"]["unique_total"], 26)

            # 3. Khong truyen --budget -> mac dinh population * generations = 6 * 2 = 12.
            args_default = build_arg_parser().parse_args(
                [
                    "--context",
                    "attribute",
                    "--model",
                    "svm",
                    "--mode",
                    "random",
                    "--seed",
                    "1",
                    "--population",
                    "6",
                    "--generations",
                    "2",
                    "--cache-path",
                    str(tmp / "cache.jsonl"),
                    "--out-dir",
                    str(tmp / "out"),
                ]
            )
            out_default = asyncio.run(
                run_ga(args_default, predictor=predictor, oracle_cache=cache, check_batch_fn=fake_check_batch)
            )
            data_default = json.loads(out_default.read_text(encoding="utf-8"))
            self.assertEqual(data_default["summary"]["unique_total"], 12)
            self.assertNotIn("budget", data_default)
            self.assertEqual(data_default["config"]["budget"], 12)


class TestPhase3Helpers(unittest.TestCase):
    def test_summarize_phase3_groups_and_resume_loader(self) -> None:
        grouped = {
            ("rf", "html_body", "ga"): [
                (0, {"summary": {"unique_total": 100, "unique_valid": 80, "unique_bypass": 20, "bypass_rate_a": 0.25, "bypass_rate_b": 0.20}}),
                (1, {"summary": {"unique_total": 120, "unique_valid": 100, "unique_bypass": 30, "bypass_rate_a": 0.30, "bypass_rate_b": 0.25}}),
            ],
            ("rf", "url", "bruteforce"): [
                (0, {"summary": {"unique_total": 26, "unique_valid": 13, "unique_bypass": 5, "bypass_rate_a": 5 / 13, "bypass_rate_b": 5 / 26}}),
            ],
        }
        summaries = summarize_phase3_groups(grouped)
        self.assertEqual(len(summaries), 2)
        ga_stat = summaries[0]["metrics"]
        self.assertAlmostEqual(ga_stat["unique_total"]["mean"], 110.0)
        self.assertAlmostEqual(ga_stat["unique_total"]["std"], 14.142136, places=5)
        bf_stat = summaries[1]["metrics"]
        self.assertEqual(bf_stat["unique_total"]["mean"], 26.0)
        self.assertIsNone(bf_stat["unique_total"]["std"])

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "res.json"
            self.assertIsNone(_try_load_existing_result(p))
            p.write_text("{bad json", encoding="utf-8")
            self.assertIsNone(_try_load_existing_result(p))
            p.write_text(json.dumps({"summary": {"unique_total": 42}}), encoding="utf-8")
            loaded = _try_load_existing_result(p)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["summary"]["unique_total"], 42)


if __name__ == "__main__":
    unittest.main()
