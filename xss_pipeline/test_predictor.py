"""Unit tests for models.predictor.ModelPredictor (RF, SVM, CNN+LSTM)."""

from __future__ import annotations

import json
import unittest

import numpy as np

from models._utils import ARTIFACTS_DIR
from models.predictor import ModelPredictor
from preprocessing.decoder import decode_payload


class TestModelPredictor(unittest.TestCase):
    """Tests for ModelPredictor loading, batching, SSOT decoding, and edge cases."""

    @classmethod
    def setUpClass(cls) -> None:
        with open(ARTIFACTS_DIR / "classical_val_report.json", encoding="utf-8") as f:
            cls.classical_report = json.load(f)
        with open(ARTIFACTS_DIR / "cnn_lstm_val_report.json", encoding="utf-8") as f:
            cls.cnn_report = json.load(f)

        cls.predictors: dict[str, ModelPredictor] = {
            "rf": ModelPredictor("rf"),
            "svm": ModelPredictor("svm"),
            "cnn_lstm": ModelPredictor("cnn_lstm"),
        }

    def test_loaded_thresholds_and_metadata(self) -> None:
        expected_rf_thr = float(
            self.classical_report["random_forest"]["best_val_metrics"]["threshold"]
        )
        expected_svm_thr = float(
            self.classical_report["svm"]["best_val_metrics"]["threshold"]
        )
        rep_seed = int(self.cnn_report["representative_seed"])
        expected_cnn_thr = float(
            self.cnn_report["per_seed_val"][str(rep_seed)]["val_metrics"]["threshold"]
        )

        self.assertEqual(self.predictors["rf"].model_name, "rf")
        self.assertIsNone(self.predictors["rf"].seed)
        self.assertAlmostEqual(self.predictors["rf"].threshold, expected_rf_thr, places=6)

        self.assertEqual(self.predictors["svm"].model_name, "svm")
        self.assertIsNone(self.predictors["svm"].seed)
        self.assertAlmostEqual(self.predictors["svm"].threshold, expected_svm_thr, places=6)

        self.assertEqual(self.predictors["cnn_lstm"].model_name, "cnn_lstm")
        self.assertEqual(self.predictors["cnn_lstm"].seed, rep_seed)
        self.assertAlmostEqual(
            self.predictors["cnn_lstm"].threshold, expected_cnn_thr, places=6
        )

        # Case-insensitive model_name check
        rf_upper = ModelPredictor("RF")
        self.assertEqual(rf_upper.model_name, "rf")
        self.assertAlmostEqual(rf_upper.threshold, expected_rf_thr, places=6)

    def test_single_string_and_batch_outputs(self) -> None:
        samples = [
            "<img src=x onerror=alert(1)>",
            "https://example.com/search?q=hello+world",
            "javascript:alert(document.domain)",
        ]
        for name, predictor in self.predictors.items():
            with self.subTest(model=name):
                batch_probs = predictor.predict_proba(samples)
                batch_preds = predictor.predict(samples)

                self.assertIsInstance(batch_probs, np.ndarray)
                self.assertEqual(batch_probs.shape, (len(samples),))
                self.assertEqual(batch_probs.dtype, np.float64)
                self.assertTrue(np.all((batch_probs >= 0.0) & (batch_probs <= 1.0)))

                self.assertIsInstance(batch_preds, np.ndarray)
                self.assertEqual(batch_preds.shape, (len(samples),))
                self.assertEqual(batch_preds.dtype, np.int64)
                self.assertTrue(np.all((batch_preds == 0) | (batch_preds == 1)))

                for i, payload in enumerate(samples):
                    single_prob = predictor.predict_proba(payload)
                    single_pred = predictor.predict(payload)
                    self.assertIsInstance(single_prob, float)
                    self.assertGreaterEqual(single_prob, 0.0)
                    self.assertLessEqual(single_prob, 1.0)
                    self.assertAlmostEqual(single_prob, float(batch_probs[i]), places=6)
                    self.assertIsInstance(single_pred, int)
                    self.assertEqual(single_pred, int(batch_preds[i]))
                    self.assertEqual(single_pred, int(single_prob >= predictor.threshold))

    def test_internal_ssot_decoding_consistency(self) -> None:
        encoded_pairs = [
            ("%3Csvg%20onload%3Dalert(1)%3E", "<svg onload=alert(1)>"),
            ("&lt;IMG SRC=x ONERROR=ALERT(1)&gt;", "<img src=x onerror=alert(1)>"),
            (
                "\\x3cscript\\x3ealert(1)\\x3c/script\\x3e",
                "<script>alert(1)</script>",
            ),
        ]
        for raw_payload, expected_decoded in encoded_pairs:
            self.assertEqual(decode_payload(raw_payload), expected_decoded)
            for name, predictor in self.predictors.items():
                with self.subTest(model=name, raw=raw_payload):
                    p_raw = predictor.predict_proba(raw_payload)
                    p_dec = predictor.predict_proba(expected_decoded)
                    self.assertAlmostEqual(p_raw, p_dec, places=6)

                    batch_pair = predictor.predict_proba([raw_payload, expected_decoded])
                    self.assertAlmostEqual(float(batch_pair[0]), float(batch_pair[1]), places=6)

    def test_empty_string_and_empty_list(self) -> None:
        for name, predictor in self.predictors.items():
            with self.subTest(model=name):
                empty_prob = predictor.predict_proba("")
                empty_pred = predictor.predict("")
                self.assertIsInstance(empty_prob, float)
                self.assertGreaterEqual(empty_prob, 0.0)
                self.assertLessEqual(empty_prob, 1.0)
                self.assertIsInstance(empty_pred, int)
                self.assertIn(empty_pred, (0, 1))

                empty_batch_probs = predictor.predict_proba([])
                empty_batch_preds = predictor.predict([])
                self.assertIsInstance(empty_batch_probs, np.ndarray)
                self.assertEqual(empty_batch_probs.shape, (0,))
                self.assertEqual(empty_batch_probs.dtype, np.float64)

                self.assertIsInstance(empty_batch_preds, np.ndarray)
                self.assertEqual(empty_batch_preds.shape, (0,))
                self.assertEqual(empty_batch_preds.dtype, np.int64)


if __name__ == "__main__":
    unittest.main()
