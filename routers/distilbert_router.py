"""Router using a DistilBERT classifier we fine-tune ourselves
(scripts/train_distilbert.py). Applies temperature scaling saved by the
training script, giving calibrated confidences."""
import os, time
import numpy as np

from schemas import RouteDecision
from .base import Router
from .local_router import URGENT, HUMAN


class DistilBertRouter(Router):
    name = "distilbert"

    def __init__(self, model_dir="models/distilbert"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        t = os.path.join(model_dir, "temperature.txt")
        self.T = float(open(t).read()) if os.path.exists(t) else 1.0

    def route(self, text: str) -> RouteDecision:
        t0 = time.perf_counter()
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=128)
        with self.torch.no_grad():
            logits = self.model(**enc).logits[0].numpy() / self.T
        p = np.exp(logits - logits.max()); p /= p.sum()
        i = int(p.argmax())
        return RouteDecision(
            queue=self.id2label[i], confidence=float(p[i]),
            urgency=2.0 if URGENT.search(text) else 0.0,
            needs_human=0.95 if HUMAN.search(text) else 0.05,
            latency_ms=(time.perf_counter() - t0) * 1000,
            probs={self.id2label[j]: float(x) for j, x in enumerate(p)},
        )
