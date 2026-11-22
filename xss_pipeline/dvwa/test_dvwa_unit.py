"""Pure-Python unit tests for DVWA evaluation helpers (no Playwright/Docker)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import unittest
from urllib.parse import parse_qs, urlparse

DVWA_DIR = Path(__file__).resolve().parent
XSS_PIPELINE_DIR = DVWA_DIR.parent
if str(DVWA_DIR) not in sys.path:
    sys.path.insert(0, str(DVWA_DIR))

from dvwa_eval import (
    build_t02_url,
    build_xss_r_url,
    format_report_tables,
    is_external_url,
    load_csv_cases,
    summarize_results,
)


class TestOriginGuard(unittest.TestCase):
    BASE = "http://127.0.0.1:8080"

    def test_same_origin_is_not_external(self):
        self.assertFalse(is_external_url("http://127.0.0.1:8080/", self.BASE))
        self.assertFalse(
            is_external_url("http://127.0.0.1:8080/vulnerabilities/xss_t02/index.php?csp=1", self.BASE)
        )
        self.assertFalse(is_external_url("http://127.0.0.1:8080/probe", self.BASE))
        self.assertFalse(is_external_url("http://127.0.0.1:80/test", "http://127.0.0.1"))

    def test_external_domains_and_ports_are_external(self):
        self.assertTrue(is_external_url("http://attacker.test/collect", self.BASE))
        self.assertTrue(is_external_url("https://attacker.test/", self.BASE))
        self.assertTrue(is_external_url("http://127.0.0.1:5000/html_body", self.BASE))
        self.assertTrue(is_external_url("http://127.0.0.1:8080.attacker.test/", self.BASE))
        self.assertTrue(is_external_url("http://127.0.0.1.attacker.test:8080/", self.BASE))

    def test_non_network_schemes_are_not_external(self):
        self.assertFalse(is_external_url("data:text/html,<script>alert(1)</script>", self.BASE))
        self.assertFalse(is_external_url("DATA:text/html,ok", self.BASE))
        self.assertFalse(is_external_url("javascript:confirm(origin)", self.BASE))
        self.assertFalse(is_external_url("about:blank", self.BASE))
        self.assertFalse(is_external_url("about:srcdoc", self.BASE))
        self.assertFalse(is_external_url("blob:http://127.0.0.1:8080/uuid", self.BASE))
        self.assertFalse(is_external_url("", self.BASE))


class TestCsvLoadingAndUrlBuilding(unittest.TestCase):
    def test_load_frozen_ga_bypassed_csv(self):
        csv_path = XSS_PIPELINE_DIR / "ga_bypassed_payloads.csv"
        cases = load_csv_cases(csv_path)
        self.assertEqual(len(cases), 380)

        ctx_counts = Counter(c["context"] for c in cases)
        self.assertEqual(
            dict(ctx_counts),
            {"html_body": 60, "attribute": 125, "js_string": 182, "url": 13},
        )

        model_counts = Counter(c["models_bypassed"] for c in cases)
        self.assertEqual(model_counts["rf"], 16)
        self.assertEqual(model_counts["svm"], 15)
        self.assertEqual(model_counts["cnn_lstm"], 349)

    def test_build_urls(self):
        case = {
            "id": "GA_BYP_0368",
            "context": "url",
            "family": "iframe_srcdoc_ga",
            "event": "none",
            "carrier": "iframe",
            "payload": "data:text/html,<script>alert(1)</script>",
        }
        url_c4 = build_t02_url("http://127.0.0.1:8080", case, "C4")
        parsed = urlparse(url_c4)
        self.assertEqual(parsed.path, "/vulnerabilities/xss_t02/index.php")
        qs = parse_qs(parsed.query)
        self.assertEqual(qs["ctx"], ["url"])
        self.assertEqual(qs["carrier"], ["iframe"])
        self.assertEqual(qs["csp"], ["1"])
        self.assertEqual(qs["enc"], ["1"])
        self.assertEqual(qs["payload"], ["data:text/html,<script>alert(1)</script>"])

        xss_r = build_xss_r_url("http://127.0.0.1:8080", case)
        self.assertEqual(urlparse(xss_r).path, "/vulnerabilities/xss_r/")
        self.assertEqual(parse_qs(urlparse(xss_r).query)["name"], [case["payload"]])


class TestSummarizeResults(unittest.TestCase):
    def test_block_rates_and_audit_sections(self):
        synthetic_cases = [
            {
                "id": "T1",
                "context": "html_body",
                "family": "",
                "event": "onclick",
                "carrier": "",
                "models_bypassed": "cnn_lstm",
                "bypass_rf": "0",
                "bypass_svm": "0",
                "bypass_cnn_lstm": "1",
                "payload": "<details onclick=alert(1)>x</details>",
            },
            {
                "id": "T2",
                "context": "attribute",
                "family": "attr_break_dquote_ga",
                "event": "onclick",
                "carrier": "",
                "models_bypassed": "rf",
                "bypass_rf": "1",
                "bypass_svm": "0",
                "bypass_cnn_lstm": "0",
                "payload": '"><details onclick=alert(1)>x</details>',
            },
            {
                "id": "T3",
                "context": "js_string",
                "family": "js_break_squote_ga",
                "event": "none",
                "carrier": "",
                "models_bypassed": "svm",
                "bypass_rf": "0",
                "bypass_svm": "1",
                "bypass_cnn_lstm": "0",
                "payload": "';alert(1);//",
            },
            {
                "id": "T4",
                "context": "url",
                "family": "iframe_srcdoc_ga",
                "event": "none",
                "carrier": "iframe",
                "models_bypassed": "cnn_lstm",
                "bypass_rf": "0",
                "bypass_svm": "0",
                "bypass_cnn_lstm": "1",
                "payload": "data:text/html,<script>alert(1)</script>",
            },
        ]

        # C1: all 4 execute
        # C2 (CSP only): T1 blocked, T2 blocked, T3 executes (non-inline <script> in js_string? simulated), T4 executes (data: iframe)
        # C3 (Enc only): all 4 blocked
        # C4 (CSP+Enc): all 4 blocked
        exec_matrix = {
            "C1": {"T1": 1, "T2": 1, "T3": 1, "T4": 1},
            "C2": {"T1": 0, "T2": 0, "T3": 1, "T4": 1},
            "C3": {"T1": 0, "T2": 0, "T3": 0, "T4": 0},
            "C4": {"T1": 0, "T2": 0, "T3": 0, "T4": 0},
        }
        eval_records = []
        for cfg in ("C1", "C2", "C3", "C4"):
            for c in synthetic_cases:
                ex = exec_matrix[cfg][c["id"]]
                eval_records.append({
                    **c,
                    "config": cfg,
                    "executed": ex,
                    "signals": ["alert"] if ex else [],
                    "csp_header": "script-src 'self'; object-src 'none';" if cfg in {"C2", "C4"} else None,
                    "csp_violations": (
                        [{"source": "securitypolicyviolation", "effectiveDirective": "script-src-attr"}]
                        if (cfg == "C2" and not ex)
                        else []
                    ),
                    "http_status": 200,
                    "elapsed_ms": 15,
                    "error": None,
                })

        crosscheck_records = [
            {
                **synthetic_cases[0],
                "config": "DVWA_XSS_R_LOW",
                "executed": 1,
                "signals": ["alert"],
                "csp_header": None,
                "csp_violations": [],
                "http_status": 200,
                "elapsed_ms": 20,
                "error": None,
            }
        ]

        report = summarize_results(
            eval_records=eval_records,
            crosscheck_records=crosscheck_records,
            default_dvwa_headers=[
                {"path": "/vulnerabilities/xss_r/?name=test", "http_status": 200, "content_security_policy": None}
            ],
        )

        # C1 baseline: 0% blocked
        self.assertEqual(report["config_by_context"]["C1"]["ALL"]["executed"], 4)
        self.assertEqual(report["config_by_context"]["C1"]["ALL"]["block_rate_vs_c1_pct"], 0.0)

        # C2: 2/4 executed => 2 blocked vs C1 => 50.0%
        self.assertEqual(report["config_by_context"]["C2"]["ALL"]["executed"], 2)
        self.assertEqual(report["config_by_context"]["C2"]["ALL"]["blocked_vs_c1"], 2)
        self.assertEqual(report["config_by_context"]["C2"]["ALL"]["block_rate_vs_c1_pct"], 50.0)
        self.assertEqual(report["config_by_context"]["C2"]["html_body"]["block_rate_vs_c1_pct"], 100.0)
        self.assertEqual(report["config_by_context"]["C2"]["url"]["block_rate_vs_c1_pct"], 0.0)

        # C3 & C4: 0/4 executed => 100.0% blocked
        self.assertEqual(report["config_by_context"]["C3"]["ALL"]["block_rate_vs_c1_pct"], 100.0)
        self.assertEqual(report["config_by_context"]["C4"]["ALL"]["block_rate_vs_c1_pct"], 100.0)

        # Models bypassed in C2: rf 100% blocked, svm 0% blocked, cnn_lstm 50% blocked (1/2)
        self.assertEqual(report["config_by_model"]["C2"]["rf"]["block_rate_vs_c1_pct"], 100.0)
        self.assertEqual(report["config_by_model"]["C2"]["svm"]["block_rate_vs_c1_pct"], 0.0)
        self.assertEqual(report["config_by_model"]["C2"]["cnn_lstm"]["block_rate_vs_c1_pct"], 50.0)

        # Crosscheck
        self.assertEqual(report["crosscheck_html_body"]["c1_html_body"], {"executed": 1, "total": 1})
        self.assertEqual(report["crosscheck_html_body"]["dvwa_xss_r"], {"executed": 1, "total": 1})
        self.assertEqual(report["crosscheck_html_body"]["both_executed_count"], 1)

        # Special observation for iframe data: in C2
        self.assertEqual(report["c2_iframe_data_observation"]["total"], 1)
        self.assertEqual(report["c2_iframe_data_observation"]["executed"], 1)
        self.assertEqual(report["c2_iframe_data_observation"]["blocked"], 0)

        # Audit
        self.assertEqual(report["audit"]["c1_not_executed"], [])
        self.assertEqual([r["id"] for r in report["audit"]["bypassed_defenses"]["C2"]], ["T3", "T4"])
        self.assertEqual(report["audit"]["bypassed_defenses"]["C3"], [])
        self.assertEqual(report["audit"]["bypassed_defenses"]["C4"], [])

        # Table formatting smoke check
        text = format_report_tables(report)
        self.assertIn("C1", text)
        self.assertIn("50.00%", text)
        self.assertIn("100.00%", text)


if __name__ == "__main__":
    unittest.main()
