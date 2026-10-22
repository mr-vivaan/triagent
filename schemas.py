from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RouteDecision:
    queue: str
    confidence: float          # confidence in the queue choice (0-1)
    urgency: float             # 0..2 (position on the urgency rubric)
    needs_human: float         # P(needs a human), 0-1
    latency_ms: float = 0.0
    probs: Optional[dict] = None


@dataclass
class TicketResult:
    ticket: str
    route: RouteDecision
    draft: Optional[str] = None
    sources: list = field(default_factory=list)
    guard_score: Optional[float] = None
    escalated: bool = False
    reasons: list = field(default_factory=list)
    trace: list = field(default_factory=list)   # human-readable steps for the demo
