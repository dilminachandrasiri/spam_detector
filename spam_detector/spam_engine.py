"""
spam_engine.py - inference layer for the Spam Detector.

Keeps everything the GUI needs away from the GUI code:
  * text normalisation (shared with training, so it must live in an importable module)
  * model loading
  * single-message analysis with an explanation (which words pushed the score)
  * fast batch scoring
"""
from __future__ import annotations

import json
import os
import re
from typing import Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(DATA_DIR, "spam.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "spam_model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

# --------------------------------------------------------------------------- #
# Text normalisation (used by the TF-IDF vectoriser during training AND inference)
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:com|net|org|info|biz|co\.uk|uk|ly|tk)\b\S*)", re.I)
_LONGNUM_RE = re.compile(r"\d{5,}")
_CURRENCY_RE = re.compile(r"[£$€]")


def fix_encoding(text: str) -> str:
    """The classic SMS-spam CSV has '£' mangled into 'å£' / 'Â£'; repair it."""
    return str(text).replace("å£", "£").replace("Â£", "£").replace("â‚¬", "€")


def normalize_text(text: str) -> str:
    """Lower-case and collapse URLs / long numbers / currency symbols into
    generic tokens so the model learns 'has a link' rather than memorising one URL."""
    t = fix_encoding(text).lower()
    t = _URL_RE.sub(" urltoken ", t)
    t = _CURRENCY_RE.sub(" currencytoken ", t)
    t = _LONGNUM_RE.sub(" longnumtoken ", t)
    return t


# --------------------------------------------------------------------------- #
# Rule-based "red flag" hints. These do NOT affect the prediction - they only
# give the user a human-readable second opinion next to the ML score.
# --------------------------------------------------------------------------- #
_PROMO_WORDS = {
    "free", "win", "winner", "won", "prize", "claim", "urgent", "cash", "offer",
    "congratulations", "guaranteed", "bonus", "txt", "stop", "call", "reply",
    "award", "selected", "limited", "bid", "ringtone", "subscribe", "voucher",
}


def red_flags(text: str) -> list[str]:
    flags: list[str] = []
    if _URL_RE.search(text):
        flags.append("Contains a link or web address")
    if re.search(r"\b\d{7,}\b|\b\d{4,5}\b\s*(?:to|on)\b", text, re.I) or re.search(r"\d{10,}", text):
        flags.append("Contains a long phone / short-code number")
    if _CURRENCY_RE.search(text) or re.search(r"\b\d+\s*(?:p|pounds|usd|dollars)\b", text, re.I):
        flags.append("Mentions money or a prize amount")
    words = {w.strip(".,!?:;\"'()").lower() for w in text.split()}
    hits = sorted(words & _PROMO_WORDS)
    if len(hits) >= 2:
        flags.append("Promotional / urgency wording: " + ", ".join(hits[:6]))
    letters = [c for c in text if c.isalpha()]
    if len(letters) > 15 and sum(c.isupper() for c in letters) / len(letters) > 0.4:
        flags.append("Excessive CAPITAL LETTERS")
    if text.count("!") >= 2:
        flags.append("Multiple exclamation marks")
    return flags


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def risk_label(p: float) -> str:
    if p < 0.20:
        return "Very low risk"
    if p < 0.50:
        return "Low risk"
    if p < 0.80:
        return "High risk"
    return "Very high risk"


# --------------------------------------------------------------------------- #
class SpamDetector:
    """Thin wrapper around the saved scikit-learn pipeline."""

    def __init__(self, model_path: str = MODEL_PATH, metrics_path: str = METRICS_PATH):
        self.model = joblib.load(model_path)
        self.metrics: dict = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as fh:
                self.metrics = json.load(fh)

    # ---- scoring ---------------------------------------------------------- #
    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        """Probability that each message is spam."""
        texts = [str(t) for t in texts]
        if not texts:
            return np.array([])
        return self.model.predict_proba(texts)[:, 1]

    def analyze(self, text: str, threshold: float = 0.5, explain: bool = True) -> dict:
        """Score one message and (optionally) explain it.

        Explanation = leave-one-word-out: remove each word, re-score, and see how
        much the spam probability moves. Works with any classifier.
        """
        text = str(text).strip()
        p = float(self.predict_proba([text])[0])
        result = {
            "text": text,
            "prob": p,
            "is_spam": p >= threshold,
            "risk": risk_label(p),
            "flags": red_flags(text),
            "spam_words": [],
            "ham_words": [],
        }
        if not explain:
            return result

        words = text.split()
        if 0 < len(words) <= 120:
            variants = [" ".join(w for j, w in enumerate(words) if j != i) for i in range(len(words))]
            probs = self.predict_proba(variants)
            # Work in log-odds: probabilities saturate near 0/1 so removing one word
            # barely changes them, but the log-odds still shift measurably.
            base = _logit(p)
            deltas: dict[str, list[float]] = {}
            for w, pv in zip(words, probs):
                key = w.strip(".,!?:;\"'()[]").lower()
                if len(key) < 2:
                    continue
                deltas.setdefault(key, []).append(base - _logit(float(pv)))
            # Filler words ("have", "for") wobble the score slightly but explain nothing,
            # so only show them if they genuinely matter.
            avg = {k: float(np.mean(v)) for k, v in deltas.items()
                   if k not in ENGLISH_STOP_WORDS or abs(np.mean(v)) >= 1.5}
            result["spam_words"] = sorted(
                [(k, v) for k, v in avg.items() if v >= 0.3], key=lambda kv: -kv[1])[:8]
            result["ham_words"] = sorted(
                [(k, v) for k, v in avg.items() if v <= -0.3], key=lambda kv: kv[1])[:6]
        return result

    def analyze_many(self, texts: Iterable[str], threshold: float = 0.5) -> list[dict]:
        texts = [str(t) for t in texts]
        probs = self.predict_proba(texts)
        return [
            {"text": t, "prob": float(p), "is_spam": bool(p >= threshold), "risk": risk_label(float(p))}
            for t, p in zip(texts, probs)
        ]


def model_exists() -> bool:
    return os.path.exists(MODEL_PATH)
