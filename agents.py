"""Specialist agents: retrieve policy snippets for their queue, then draft a reply."""
from config import QUEUES

SYSTEM = ("You are a customer-support agent for the {queue} team ({desc}). "
          "Write a short, polite, accurate reply. Use ONLY the policy snippets given. "
          "Never promise refunds, compensation or timelines that the policy does not state. "
          "If the policy does not cover the question, say you will pass it to a colleague.")


class SpecialistAgent:
    def __init__(self, queue, kb, llm):
        self.queue, self.kb, self.llm = queue, kb, llm

    def draft(self, ticket: str):
        sources = self.kb.search(self.queue, ticket, k=3)
        prompt = ("Customer message:\n" + ticket + "\n\nRelevant policy:\n" +
                  "\n".join(f"- {s}" for s in sources) + "\n\nWrite the reply.")
        system = SYSTEM.format(queue=self.queue, desc=QUEUES[self.queue])
        return self.llm.generate(system, prompt), sources
