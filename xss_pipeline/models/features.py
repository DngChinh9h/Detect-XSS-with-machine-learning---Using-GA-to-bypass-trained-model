"""Feature extraction and character-level sequence encoding for XSS classifiers."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:
    import numpy as np
    from scipy.sparse import csr_matrix

# Compiled regexes for the 11 structural features (computed on lowercased strings).
_HTML_TAG_RE = re.compile(r"<[a-z][a-z0-9]*\b")
_ON_EVENT_RE = re.compile(r"\bon[a-z]{2,}\s*=")
_SPECIAL_CHAR_RE = re.compile(r"[^\w\s]")

STRUCTURAL_FEATURE_NAMES: tuple[str, ...] = (
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
)

# Character vocabulary for CNN+LSTM: 0 = PAD, 1 = UNK, 2..96 = printable ASCII chr(32)..chr(126).
PAD_IDX: int = 0
UNK_IDX: int = 1
VOCAB_SIZE: int = 97
MAX_LEN: int = 200
CHAR_TO_IDX: dict[str, int] = {chr(code): (code - 30) for code in range(32, 127)}

VALID_EXTRACTOR_MODES: tuple[str, ...] = ("full", "tfidf_only", "structural_only")


def _normalize_text(text: Any) -> str:
    """Convert payload input to a safe string, treating None/NaN as empty."""
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    return str(text)


def _compute_structural_vector(text: Any) -> list[float]:
    """Compute the 11 structural features for a single lowercased payload string."""
    s = _normalize_text(text).lower()
    length = len(s)

    html_tag_count = float(len(_HTML_TAG_RE.findall(s)))
    on_event_attr_count = float(len(_ON_EVENT_RE.findall(s)))
    has_javascript_scheme = 1.0 if "javascript:" in s else 0.0
    has_data_scheme = 1.0 if "data:" in s else 0.0
    angle_bracket_count = float(s.count("<") + s.count(">"))
    quote_count = float(s.count("'") + s.count('"'))
    semicolon_count = float(s.count(";"))
    paren_count = float(s.count("(") + s.count(")"))
    key_tag_keyword_count = float(s.count("script") + s.count("img") + s.count("svg"))
    length_feat = float(length)
    special_char_ratio = float(len(_SPECIAL_CHAR_RE.findall(s)) / max(length, 1))

    return [
        html_tag_count,
        on_event_attr_count,
        has_javascript_scheme,
        has_data_scheme,
        angle_bracket_count,
        quote_count,
        semicolon_count,
        paren_count,
        key_tag_keyword_count,
        length_feat,
        special_char_ratio,
    ]


def extract_structural_features(texts: Iterable[Any]) -> np.ndarray:
    """Extract the 11 structural features for a sequence of decoded payloads."""
    import numpy as np

    rows = [_compute_structural_vector(t) for t in texts]
    if not rows:
        return np.zeros((0, len(STRUCTURAL_FEATURE_NAMES)), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _encode_single_char_sequence(text: Any, max_len: int = MAX_LEN) -> list[int]:
    """Encode a single payload string into fixed-length character token indices."""
    s = _normalize_text(text)
    seq = [CHAR_TO_IDX.get(ch, UNK_IDX) for ch in s[:max_len]]
    if len(seq) < max_len:
        seq.extend([PAD_IDX] * (max_len - len(seq)))
    return seq


def encode_char_sequences(texts: Sequence[Any] | Iterable[Any], max_len: int = MAX_LEN) -> np.ndarray:
    """Encode payload strings into a ``(N, max_len)`` ``np.int64`` index matrix."""
    import numpy as np

    rows = [_encode_single_char_sequence(t, max_len=max_len) for t in texts]
    if not rows:
        return np.zeros((0, max_len), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


class XSSFeatureExtractor:
    """Unified TF-IDF + scaled structural feature extractor for classical models.

    Supports ``mode`` in ``{"full", "tfidf_only", "structural_only"}``.
    Must be fitted exclusively on the training split.
    """

    def __init__(self, mode: str = "full") -> None:
        if mode not in VALID_EXTRACTOR_MODES:
            raise ValueError(
                f"Invalid mode {mode!r}. Expected one of {VALID_EXTRACTOR_MODES}."
            )
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import StandardScaler

        self.mode = mode
        self.tfidf = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            max_features=5000,
            min_df=2,
            sublinear_tf=True,
        )
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(self, texts: Iterable[Any], y: Any = None) -> XSSFeatureExtractor:
        """Fit active feature transformers on training texts only."""
        normalized = [_normalize_text(t) for t in texts]
        if self.mode in ("full", "tfidf_only"):
            self.tfidf.fit(normalized)
        if self.mode in ("full", "structural_only"):
            X_struct = extract_structural_features(normalized)
            self.scaler.fit(X_struct)
        self._is_fitted = True
        return self

    def transform(self, texts: Iterable[Any]) -> csr_matrix:
        """Transform texts into a CSR sparse feature matrix matching ``self.mode``."""
        from scipy.sparse import csr_matrix, hstack

        if not self._is_fitted:
            raise RuntimeError("XSSFeatureExtractor must be fitted before calling transform().")

        normalized = [_normalize_text(t) for t in texts]
        if self.mode == "tfidf_only":
            return self.tfidf.transform(normalized)

        X_struct_scaled = self.scaler.transform(extract_structural_features(normalized))
        X_struct_csr = csr_matrix(X_struct_scaled)

        if self.mode == "structural_only":
            return X_struct_csr

        X_tfidf = self.tfidf.transform(normalized)
        return hstack([X_tfidf, X_struct_csr], format="csr")

    def fit_transform(self, texts: Iterable[Any], y: Any = None) -> csr_matrix:
        """Fit on training texts and return the transformed CSR feature matrix."""
        return self.fit(texts, y=y).transform(texts)

    def get_feature_names(self) -> list[str]:
        """Return output column names in exact order matching ``transform()``."""
        if not self._is_fitted:
            raise RuntimeError("XSSFeatureExtractor must be fitted before calling get_feature_names().")

        if self.mode == "structural_only":
            return list(STRUCTURAL_FEATURE_NAMES)

        tfidf_names = [str(name) for name in self.tfidf.get_feature_names_out()]
        if self.mode == "tfidf_only":
            return tfidf_names

        return tfidf_names + list(STRUCTURAL_FEATURE_NAMES)
