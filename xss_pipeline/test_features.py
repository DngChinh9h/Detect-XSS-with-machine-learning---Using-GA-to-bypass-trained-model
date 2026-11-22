"""Pure-Python unit tests for models.features structural regexes and char encoding."""

from __future__ import annotations

import unittest

from models.features import (
    CHAR_TO_IDX,
    MAX_LEN,
    PAD_IDX,
    STRUCTURAL_FEATURE_NAMES,
    UNK_IDX,
    VOCAB_SIZE,
    _compute_structural_vector,
    _encode_single_char_sequence,
)


def _as_dict(vec: list[float]) -> dict[str, float]:
    return dict(zip(STRUCTURAL_FEATURE_NAMES, vec))


class TestStructuralFeatures(unittest.TestCase):
    """Tests for the 11 structural features and their regex patterns."""

    def test_feature_names_count_and_order(self) -> None:
        self.assertEqual(len(STRUCTURAL_FEATURE_NAMES), 11)
        self.assertEqual(
            STRUCTURAL_FEATURE_NAMES,
            (
                "html_tag_count",
                "on_event_attr_count",
                "has_javascript_scheme",
                "has_data_scheme",
                "angle_bracket_count",
                "quote_count",
                "semicolon_count",
                "paren_count",
                "key_tag_keyword_count",
                "length",
                "special_char_ratio",
            ),
        )

    def test_html_tag_count_ignores_closing_tags(self) -> None:
        feats = _as_dict(_compute_structural_vector('<a href="x">ok</a>'))
        self.assertEqual(feats["html_tag_count"], 1.0)
        self.assertEqual(feats["angle_bracket_count"], 4.0)
        self.assertEqual(feats["quote_count"], 2.0)
        self.assertEqual(feats["length"], 18.0)

    def test_on_event_attr_regex(self) -> None:
        self.assertEqual(
            _as_dict(_compute_structural_vector("online"))["on_event_attr_count"],
            0.0,
        )
        self.assertEqual(
            _as_dict(_compute_structural_vector("donerror=alert(1)"))["on_event_attr_count"],
            0.0,
        )
        self.assertEqual(
            _as_dict(_compute_structural_vector("on=1"))["on_event_attr_count"],
            0.0,
        )
        self.assertEqual(
            _as_dict(_compute_structural_vector("onerror=alert(1)"))["on_event_attr_count"],
            1.0,
        )
        self.assertEqual(
            _as_dict(_compute_structural_vector("<IMG OnError = alert(1)>"))["on_event_attr_count"],
            1.0,
        )

    def test_schemes_keywords_and_punctuation(self) -> None:
        js_feats = _as_dict(_compute_structural_vector("javascript:alert(1);"))
        self.assertEqual(js_feats["has_javascript_scheme"], 1.0)
        self.assertEqual(js_feats["has_data_scheme"], 0.0)
        self.assertEqual(js_feats["semicolon_count"], 1.0)
        self.assertEqual(js_feats["paren_count"], 2.0)

        data_feats = _as_dict(
            _compute_structural_vector("data:text/html,<script>alert(1)</script>")
        )
        self.assertEqual(data_feats["has_data_scheme"], 1.0)
        self.assertEqual(data_feats["html_tag_count"], 1.0)
        self.assertEqual(data_feats["key_tag_keyword_count"], 2.0)

        svg_feats = _as_dict(
            _compute_structural_vector("<svg onload=fetch('https://attacker.test')><img src=x>")
        )
        self.assertEqual(svg_feats["html_tag_count"], 2.0)
        self.assertEqual(svg_feats["on_event_attr_count"], 1.0)
        self.assertEqual(svg_feats["key_tag_keyword_count"], 2.0)
        self.assertEqual(svg_feats["quote_count"], 2.0)

    def test_empty_missing_and_special_char_ratio(self) -> None:
        for empty_val in ("", None, float("nan")):
            feats = _as_dict(_compute_structural_vector(empty_val))
            self.assertEqual(feats["length"], 0.0)
            self.assertEqual(feats["special_char_ratio"], 0.0)

        ratio_feats = _as_dict(_compute_structural_vector("a!@ "))
        self.assertEqual(ratio_feats["length"], 4.0)
        self.assertAlmostEqual(ratio_feats["special_char_ratio"], 0.5)


class TestCharEncoding(unittest.TestCase):
    """Tests for CNN+LSTM character vocabulary and fixed-length encoding."""

    def test_vocab_bounds(self) -> None:
        self.assertEqual(VOCAB_SIZE, 97)
        self.assertEqual(MAX_LEN, 200)
        self.assertEqual(PAD_IDX, 0)
        self.assertEqual(UNK_IDX, 1)
        self.assertEqual(len(CHAR_TO_IDX), 95)
        self.assertEqual(CHAR_TO_IDX[chr(32)], 2)
        self.assertEqual(CHAR_TO_IDX[chr(126)], 96)

    def test_padding_truncation_and_unk(self) -> None:
        short_seq = _encode_single_char_sequence("ab\né", max_len=10)
        self.assertEqual(len(short_seq), 10)
        self.assertEqual(short_seq[:4], [CHAR_TO_IDX["a"], CHAR_TO_IDX["b"], UNK_IDX, UNK_IDX])
        self.assertEqual(short_seq[4:], [PAD_IDX] * 6)

        long_seq = _encode_single_char_sequence("x" * 250)
        self.assertEqual(len(long_seq), 200)
        self.assertTrue(all(tok == CHAR_TO_IDX["x"] for tok in long_seq))


if __name__ == "__main__":
    unittest.main()
