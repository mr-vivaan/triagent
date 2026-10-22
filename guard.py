"""Guard agent: before a reply is sent, ask 'is this draft safe & policy-compliant?'
JevGuard uses one Noul per check. HeuristicGuard is the offline fallback / baseline."""
import hashlib, json, os, re
from config import env

CHECKS = {
    "policy_ok": "The reply only states things supported by the policy snippets and makes no unauthorised promises (refunds, compensation, deadlines).",
    "answers_issue": "The reply actually addresses the customer's problem.",
    "tone_ok": "The reply is polite and professional.",
}
RISKY = re.compile(r"\b(guarantee|full refund|compensat|100%|definitely will|within \d+ (minutes|hours))\b", re.I)


class HeuristicGuard:
    name = "heuristic"

    def check(self, ticket, draft, sources):
        return 0.1 if RISKY.search(draft) else 0.95


class JevGuard:
    name = "jev"

    def __init__(self, model=None, cache_path=".cache/jev_guard.json", client=None):
        from typesafe_sdk import Noul  # lazy import
        self._Noul = Noul
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

    def _key(self, ticket, draft, sources):
        payload = json.dumps({"model": self.model, "ticket": ticket, "draft": draft, "sources": sources}, sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()

    def _save(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f)

    def check(self, ticket, draft, sources):
        k = self._key(ticket, draft, sources)
        if k in self.cache:
            return float(self.cache[k])

        state = {"customer_message": ticket, "policy_snippets": sources, "draft_reply": draft}
        resp = self.client.system_one(
            state=state,
            questions={k: self._Noul(instructions=v) for k, v in CHECKS.items()},
            model=self.model,
        )
        # weakest check decides
        score = min(float(resp.nouls[k].noul) for k in CHECKS)
        self.cache[k] = score
        self._save()
        return score


def get_guard(name=None):
    name = (name or env("GUARD", "heuristic")).lower()
    return JevGuard() if name == "jev" else HeuristicGuard()
