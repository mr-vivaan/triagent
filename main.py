"""Run Triagent on tickets.   python main.py "I was charged twice"   |   python main.py   (interactive)
Config via .env:  ROUTER=jev|tfidf|distilbert  GUARD=jev|heuristic  LLM=mock|gemini|ollama"""
import sys, pandas as pd
from config import env
from kb import KnowledgeBase
from llm import get_llm
from guard import get_guard
from pipeline import Triagent


def build_router(name):
    if name == "jev":
        from routers.jev_router import JevRouter
        return JevRouter()
    if name == "distilbert":
        from routers.distilbert_router import DistilBertRouter
        return DistilBertRouter()
    from routers.local_router import TfidfRouter
    df = pd.read_csv("data/train.csv" if __import__("os").path.exists("data/train.csv") else "data/sample_tickets.csv")
    return TfidfRouter().fit(df.text.tolist(), df.queue.tolist())


def show(res):
    print("\n".join(res.trace))
    print("\nREPLY:" if not res.escalated else "\n>> ESCALATED TO HUMAN")
    print(res.draft if not res.escalated else "; ".join(res.reasons))
    print("-" * 70)


if __name__ == "__main__":
    app = Triagent(build_router(env("ROUTER", "tfidf")), KnowledgeBase(), get_llm(), get_guard())
    if len(sys.argv) > 1:
        show(app.handle(" ".join(sys.argv[1:])))
    else:
        while True:
            t = input("ticket> ").strip()
            if not t:
                break
            show(app.handle(t))
