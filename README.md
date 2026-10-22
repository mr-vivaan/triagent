# Triagent: multi-agent customer-support triage

Router agent (Jev / DistilBERT / TF-IDF) -> specialist agents (LLM + policy retrieval) -> guard agent (Jev) -> send or escalate to human.

```
ticket -> [Router]  queue + confidence + urgency + needs_human
            |-- needs human / low confidence ---------------> HUMAN
            v
          [Specialist agent: billing | orders | account | general]  (retrieves policy, drafts reply)
            v
          [Guard]  Noul checks: policy-compliant? answers issue? polite?
            |-- fails ---------------------------------------> HUMAN
            v
          SEND
```

## Run (works offline, no keys)
    pip install -r requirements.txt
    python main.py "I was charged twice for my order"
    python evaluate.py --routers tfidf        # on the tiny sample file (train==test, so 100% is meaningless)

## Real data (Phase 1)
    pip install datasets
    python data/prepare_data.py               # Bitext dataset -> data/{train,val,test}.csv
    python evaluate.py --train data/train.csv --test data/test.csv --routers tfidf

## Turn on Jev
    cp .env.example .env     # add TYPESAFE_API_KEY, set ROUTER=jev, GUARD=jev
    # pin JEV_MODEL=jev-1.13.0 while running experiments so results are reproducible
    python evaluate.py --train data/train.csv --test data/test.csv --routers tfidf,jev --limit 300
(Jev calls are cached in .cache/ so re-runs are free.)

## Real LLM for specialist agents
    LLM=gemini (set GEMINI_API_KEY)   or   LLM=ollama (local, free)

## DistilBERT router (Phase 2, Colab GPU)
    python scripts/train_distilbert.py        # also fits temperature scaling
    python evaluate.py --routers tfidf,distilbert,jev
