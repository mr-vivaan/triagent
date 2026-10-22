"""Router agent backed by Jev (TypeSafe System One).

ONE request answers three questions about the ticket:
  queue       (Choice) -> which specialist agent
  urgency     (Score)  -> 0..2 on a rubric
  needs_human (Noul)   -> P(this needs a human)
Responses are cached on disk so re-running evaluations costs nothing.
"""
import hashlib, json, os, time

from config import QUEUES, URGENCY_LEVELS, env
from schemas import RouteDecision
from .base import Router


class JevRouter(Router):
    name = "jev"

    def __init__(self, model=None, cache_path=".cache/jev_router.json", client=None):
        from typesafe_sdk import Choice, Noul, Score  # lazy import
        self._Choice, self._Noul, self._Score = Choice, Noul, Score
        self.model = model or env("JEV_MODEL", "jev-latest")
        self._client = client
        self.cache_path = cache_path
        self.cache = {}
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                self.cache = json.load(f)

    @property
    def client(self):
        if self._client is None:
            from typesafe_sdk import TypeSafeClient
            api_key = env("TYPESAFE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "TYPESAFE_API_KEY is not set in environment or .env. "
                    "Please set it before running live Jev calls."
                )
            self._client = TypeSafeClient(api_key=api_key, model=self.model)
        return self._client

    @client.setter
    def client(self, val):
        self._client = val

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _key(self, text):
        return hashlib.sha1(f"{self.model}|{text}".encode()).hexdigest()

    def _save(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f)

    def route(self, text: str) -> RouteDecision:
        k = self._key(text)
        if k in self.cache:
            return RouteDecision(**{**self.cache[k], "latency_ms": 0.0})

        t0 = time.perf_counter()
        resp = self.client.system_one(
            state=text,
            questions={
                "queue": self._Choice(
                    instructions="Which support team should handle this ticket?",
                    criteria=QUEUES,
                ),
                "urgency": self._Score(
                    instructions="How urgent is this ticket?",
                    criteria=URGENCY_LEVELS,
                ),
                "needs_human": self._Noul(
                    instructions=("This ticket needs a human agent: it involves a legal "
                                  "threat, fraud, a safety issue, or a very upset customer."),
                ),
            },
            model=self.model,
        )
        ms = (time.perf_counter() - t0) * 1000
        q = resp.choices["queue"]
        d = dict(
            queue=q.choice,
            confidence=float(q.confidence),
            urgency=float(resp.scores["urgency"].score),
            needs_human=float(resp.nouls["needs_human"].noul),
            probs=dict(getattr(q, "probabilities", {}) or {}),
        )
        self.cache[k] = d
        self._save()
        return RouteDecision(**d, latency_ms=ms)
