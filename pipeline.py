"""Triagent orchestrator:  route -> (escalate?) -> specialist draft -> guard -> send / escalate."""
from config import ROUTE_CONF_MIN, HUMAN_NOUL_MIN, GUARD_NOUL_MIN
from agents import SpecialistAgent
from schemas import TicketResult


class Triagent:
    def __init__(self, router, kb, llm, guard):
        self.router, self.guard = router, guard
        self.agents = {q: SpecialistAgent(q, kb, llm) for q in kb.chunks}

    def handle(self, ticket: str) -> TicketResult:
        r = self.router.route(ticket)
        res = TicketResult(ticket=ticket, route=r)
        res.trace.append(f"[router:{self.router.name}] queue={r.queue} conf={r.confidence:.2f} "
                         f"urgency={r.urgency:.1f} needs_human={r.needs_human:.2f} ({r.latency_ms:.0f} ms)")

        if r.needs_human >= HUMAN_NOUL_MIN:
            res.escalated = True; res.reasons.append("router: ticket flagged as needing a human")
        if r.confidence < ROUTE_CONF_MIN:
            res.escalated = True; res.reasons.append(f"router: low routing confidence ({r.confidence:.2f})")
        if r.queue not in self.agents:
            res.escalated = True; res.reasons.append(f"no specialist for queue '{r.queue}'")
        if res.escalated:
            res.trace.append("[escalate] " + "; ".join(res.reasons))
            return res

        draft, sources = self.agents[r.queue].draft(ticket)
        res.draft, res.sources = draft, sources
        res.trace.append(f"[agent:{r.queue}] drafted reply using {len(sources)} policy snippets")

        res.guard_score = self.guard.check(ticket, draft, sources)
        res.trace.append(f"[guard:{self.guard.name}] compliance={res.guard_score:.2f}")
        if res.guard_score < GUARD_NOUL_MIN:
            res.escalated = True
            res.reasons.append(f"guard: draft failed compliance check ({res.guard_score:.2f})")
            res.trace.append("[escalate] " + res.reasons[-1])
        else:
            res.trace.append("[send] reply approved")
        return res
