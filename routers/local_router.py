"""Baseline router trained by us: TF-IDF + Logistic Regression.
Fast, free, works offline. Urgency / needs_human use simple keyword rules
(deliberately weak - this is what Jev has to beat on those two)."""
import re, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from schemas import RouteDecision
from .base import Router

URGENT = re.compile(r"\b(urgent|immediately|asap|right now|charged twice|blocked|can'?t access)\b", re.I)
HUMAN = re.compile(r"\b(lawyer|sue|legal|fraud|scam|chargeback|unacceptable|furious|worst)\b", re.I)


class TfidfRouter(Router):
    name = "tfidf"

    def __init__(self):
        self.pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
            LogisticRegression(max_iter=2000, C=5.0),
        )

    def fit(self, texts, labels):
        self.pipe.fit(texts, labels)
        return self

    def route(self, text: str) -> RouteDecision:
        t0 = time.perf_counter()
        p = self.pipe.predict_proba([text])[0]
        classes = self.pipe.classes_
        i = int(np.argmax(p))
        return RouteDecision(
            queue=str(classes[i]), confidence=float(p[i]),
            urgency=2.0 if URGENT.search(text) else 0.0,
            needs_human=0.95 if HUMAN.search(text) else 0.05,
            latency_ms=(time.perf_counter() - t0) * 1000,
            probs={str(c): float(x) for c, x in zip(classes, p)},
        )
