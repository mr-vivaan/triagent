# AGENTS.md: Triagent project context

## What this project is
Triagent is a college deep-learning project: a **multi-agent customer-support triage system**.
Academic question: *can a decision model (Jev) and/or a small model we train ourselves route and
safety-check support tickets accurately, with trustworthy confidence, at low cost and latency?*
It must stay **simple, reproducible and easy to explain to faculty**. Prefer small, readable code over clever abstractions.

## Architecture
```
ticket -> Router agent  -> queue + confidence + urgency + needs_human
            |-- needs_human high OR confidence low -----------> ESCALATE (human)
            v
          Specialist agent (billing | orders | account | general): retrieve policy -> draft reply (LLM)
            v
          Guard agent: Noul checks on the draft (policy ok? answers issue? polite?)
            |-- weakest check below threshold ----------------> ESCALATE (human)
            v
          SEND
```
Three interchangeable routers (same interface, compared in `evaluate.py`):
`jev` (TypeSafe API), `tfidf` (TF-IDF + LogReg baseline), `distilbert` (our fine-tune + temperature scaling).

## File map
- `config.py`: queues, Bitext category->queue map, urgency rubric, **all thresholds** (ROUTE_CONF_MIN, HUMAN_NOUL_MIN, GUARD_NOUL_MIN)
- `schemas.py`: `RouteDecision`, `TicketResult`
- `routers/`: `base.py` (interface), `jev_router.py`, `local_router.py` (TF-IDF), `distilbert_router.py`
- `agents.py`: `SpecialistAgent` (retrieve + draft) · `kb.py` + `kb/*.md`: policy docs, one bullet = one retrievable chunk
- `guard.py`: `JevGuard` and `HeuristicGuard` · `llm.py`: `mock | gemini | ollama`
- `pipeline.py`: `Triagent.handle()` orchestration + human-readable trace
- `evaluate.py`: accuracy, macro-F1, ECE, selective accuracy @ coverage, latency; writes `results_router_comparison.csv`
- `data/prepare_data.py` (Bitext -> train/val/test CSV), `data/sample_tickets.csv` (tiny dev set only)
- `scripts/train_distilbert.py`: fine-tune + temperature scaling (needs GPU, run on Colab)
- Config via `.env` (see `.env.example`): `ROUTER`, `GUARD`, `LLM`, `JEV_MODEL`, API keys

## Commands
```
pip install -r requirements.txt
python main.py "I was charged twice"          # run one ticket (works offline with defaults)
python data/prepare_data.py                   # needs `pip install datasets` + internet
python evaluate.py --train data/train.csv --test data/test.csv --routers tfidf,jev --limit 300
python scripts/train_distilbert.py            # Phase 2, GPU
```

## Jev facts (TypeSafe "System One" decision model, NOT a text generator)
- Endpoint `POST https://api.typesafe.ai/v1/systemone`; Python SDK: `pip install typesafe-sdk`; key in `TYPESAFE_API_KEY`.
- One call = a `state` (string or JSON object) + named `questions`. Three primitives:
  `Choice` (pick one option; returns `.choice`, `.confidence`, `.probabilities`; max 255 options),
  `Score` (rubric position, 2-10 levels, **not** normalised to 0-1),
  `Noul` (probability 0-1 that a statement is true; **not a boolean**: compare to a threshold).
- SDK access pattern: `resp.choices[id]`, `resp.scores[id]`, `resp.nouls[id]`.
- Jev is weak at arithmetic/math reasoning: never ask it to calculate; keep maths in code.
- **Docs index for the agent: https://docs.typesafe.ai/llms.txt**. Read the relevant docs page before changing any Jev call.
  Optional: install the TypeSafe agent skill (`npx skills add typesafe-ai/skills --skill typesafe-ai`).
- The Jev code in this repo was written from the quickstart and has NOT yet been run against the live API. Verify it first.

## Rules for the agent
1. **Never invent results.** Do not fabricate metrics, accuracies or API outputs. Numbers in the report must come from `evaluate.py` runs.
2. **Never commit secrets.** Keys only in `.env` (gitignored). Add `.env`, `.cache/`, `models/`, `runs/` to `.gitignore`.
3. **Keep experiments reproducible:** pin `JEV_MODEL` (e.g. `jev-1.13.0`) for evaluation runs, keep the on-disk Jev cache in `.cache/`, fixed random seeds (42).
4. **Do not silently change thresholds** in `config.py`. Threshold changes must be intentional and noted (they affect the report).
5. **Keep the Router interface stable** (`route(text) -> RouteDecision`) so routers stay swappable and comparable.
6. Evaluation must use held-out data: never report accuracy on training data (the sample CSV is train==test, for smoke tests only).
7. Jev calls cost money: batch questions into one request, always go through the cache, use `--limit` while developing.
8. Ask before adding heavy dependencies or frameworks. Plain Python + the libraries in `requirements.txt` is the default.
9. Run `python main.py "test ticket"` after any change to the pipeline to confirm nothing broke.

## Status / roadmap (update as work progresses)
- [x] Offline pipeline works end to end (tfidf router, mock LLM, heuristic guard)
- [ ] Verify Jev router + Jev guard against the live API
- [x] Prepare Bitext data; run tfidf vs jev comparison on the test split (Bitext prepared: 21,497 train / 2,687 val / 2,688 test)
- [x] Real LLM for specialist agents (Gemini live with gemini-3.5-flash-lite)
- [ ] Train DistilBERT router + temperature scaling; add to comparison
- [ ] Threshold study (automation rate vs. bad replies let through)
- [ ] Ablation: no router / no guard
- [ ] Tricky-ticket test set (angry, multi-issue, policy traps) for guard evaluation
- [ ] Streamlit demo UI
- [ ] Report + slides

## Definition of done
Everything runs from a fresh clone with `README.md` instructions; a results table compares tfidf / distilbert / jev
on accuracy, ECE, selective accuracy, latency and cost; the threshold study and ablation are plotted; a live demo handles
normal, ambiguous and risky tickets with a visible trace.
