"""Unified inference wrapper for XSS classification models (RF, SVM, CNN+LSTM).

Designed for final test evaluation and Phase 3 GA fitness scoring. Always applies
:func:`preprocessing.decoder.decode_payload` before feature extraction or character
encoding to guarantee zero train-serving skew.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence, overload

import joblib
import numpy as np
import torch

from models._utils import ARTIFACTS_DIR
from models.features import MAX_LEN, encode_char_sequences
from models.train_cnn_lstm import CharCNNLSTM
from preprocessing.decoder import decode_payload

VALID_MODEL_NAMES: tuple[str, ...] = ("rf", "svm", "cnn_lstm")


class ModelPredictor:
    """Standardized predictor wrapping RF, SVM, or Char-level CNN+LSTM."""

    def __init__(
        self,
        model_name: str,
        artifacts_dir: str | Path = ARTIFACTS_DIR,
        seed: int | None = None,
        device: str | None = None,
    ) -> None:
        normalized_name = model_name.strip().lower()
        if normalized_name not in VALID_MODEL_NAMES:
            raise ValueError(
                f"Unsupported model_name={model_name!r}. Expected one of {VALID_MODEL_NAMES}."
            )

        self.model_name: str = normalized_name
        self.artifacts_dir: Path = Path(artifacts_dir)
        self.seed: int | None = None
        self.threshold: float

        self._extractor: Any = None
        self._model: Any = None
        self._device: torch.device | None = None

        if self.model_name in ("rf", "svm"):
            self._load_classical()
        else:
            self._load_cnn_lstm(seed=seed, device=device)

    def _load_classical(self) -> None:
        extractor_path = self.artifacts_dir / "feature_extractor_full.joblib"
        model_filename = "rf_best.joblib" if self.model_name == "rf" else "svm_best.joblib"
        model_path = self.artifacts_dir / model_filename
        report_path = self.artifacts_dir / "classical_val_report.json"

        self._extractor = joblib.load(extractor_path)
        self._model = joblib.load(model_path)

        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

        report_key = "random_forest" if self.model_name == "rf" else "svm"
        self.threshold = float(report[report_key]["best_val_metrics"]["threshold"])
        self.seed = None

    def _load_cnn_lstm(self, seed: int | None, device: str | None) -> None:
        report_path = self.artifacts_dir / "cnn_lstm_val_report.json"
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

        resolved_seed = int(report["representative_seed"]) if seed is None else int(seed)
        self.seed = resolved_seed

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        ckpt_path = self.artifacts_dir / f"cnn_lstm_seed{resolved_seed}.pt"
        try:
            ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=True)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=self._device)

        model = CharCNNLSTM().to(self._device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self._model = model

        if "best_threshold" in ckpt:
            self.threshold = float(ckpt["best_threshold"])
        else:
            self.threshold = float(
                report["per_seed_val"][str(resolved_seed)]["val_metrics"]["threshold"]
            )

    @overload
    def predict_proba(self, raw_payloads: str, batch_size: int = 256) -> float: ...

    @overload
    def predict_proba(self, raw_payloads: Sequence[str], batch_size: int = 256) -> np.ndarray: ...

    def predict_proba(
        self,
        raw_payloads: str | Sequence[str],
        batch_size: int = 256,
    ) -> float | np.ndarray:
        """Decode raw payload(s) with SSOT decoder and return malicious probability."""
        is_single = isinstance(raw_payloads, str)
        payload_list = [raw_payloads] if is_single else list(raw_payloads)

        if not payload_list:
            return np.zeros((0,), dtype=np.float64)

        decoded = [decode_payload(p) for p in payload_list]

        if self.model_name in ("rf", "svm"):
            features = self._extractor.transform(decoded)
            probs = np.asarray(self._model.predict_proba(features)[:, 1], dtype=np.float64)
        else:
            assert self._device is not None
            x_seq = encode_char_sequences(decoded, max_len=MAX_LEN)
            tensor_x = torch.as_tensor(np.array(x_seq, dtype=np.int64, copy=True), dtype=torch.long)
            batches: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, len(tensor_x), batch_size):
                    batch_x = tensor_x[start : start + batch_size].to(self._device)
                    logits = self._model(batch_x).squeeze(1)
                    batch_probs = torch.sigmoid(logits).detach().cpu().numpy()
                    batches.append(batch_probs)
            probs = np.asarray(np.concatenate(batches, axis=0), dtype=np.float64)

        probs = np.clip(probs, 0.0, 1.0)
        if is_single:
            return float(probs[0])
        return probs

    @overload
    def predict(self, raw_payloads: str, batch_size: int = 256) -> int: ...

    @overload
    def predict(self, raw_payloads: Sequence[str], batch_size: int = 256) -> np.ndarray: ...

    def predict(
        self,
        raw_payloads: str | Sequence[str],
        batch_size: int = 256,
    ) -> int | np.ndarray:
        """Predict binary label (0/1) using the Phase 2A frozen ``best_threshold``."""
        if isinstance(raw_payloads, str):
            prob = self.predict_proba(raw_payloads, batch_size=batch_size)
            return int(prob >= self.threshold)

        probs = self.predict_proba(raw_payloads, batch_size=batch_size)
        return (probs >= self.threshold).astype(np.int64)
