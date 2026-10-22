"""Compare routers on a labelled CSV (columns: text, queue).
Reports accuracy, macro-F1, ECE (calibration), selective accuracy @ coverage, latency.
Usage:  python evaluate.py --train data/train.csv --test data/test.csv --routers tfidf,jev [--limit 300]"""
import argparse, json
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1); total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total


def selective(conf, correct, coverages=(1.0, 0.8, 0.6, 0.4)):
    order = np.argsort(-conf); out = {}
    for c in coverages:
        n = max(1, int(round(c * len(conf))))
        out[f"acc@{int(c*100)}%cov"] = float(correct[order[:n]].mean())
    return out


def build(name, train_df):
    if name == "tfidf":
        from routers.local_router import TfidfRouter
        return TfidfRouter().fit(train_df.text.tolist(), train_df.queue.tolist())
    if name == "jev":
        from routers.jev_router import JevRouter
        return JevRouter()
    if name == "distilbert":
        from routers.distilbert_router import DistilBertRouter
        return DistilBertRouter()
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/sample_tickets.csv")
    ap.add_argument("--test", default="data/sample_tickets.csv")
    ap.add_argument("--routers", default="tfidf")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    train_df, test_df = pd.read_csv(a.train), pd.read_csv(a.test)
    for df in [train_df, test_df]:
        if "text" not in df.columns and "ticket" in df.columns:
            df["text"] = df["ticket"]
        if "queue" not in df.columns and "expected_queue" in df.columns:
            df["queue"] = df["expected_queue"]
    if a.limit:
        test_df = test_df.sample(min(a.limit, len(test_df)), random_state=0)

    rows = []
    for name in a.routers.split(","):
        r = build(name, train_df)
        preds, conf, lat = [], [], []
        for t in test_df.text:
            d = r.route(t); preds.append(d.queue); conf.append(d.confidence); lat.append(d.latency_ms)
        y = test_df.queue.values; preds = np.array(preds); conf = np.array(conf)
        correct = (preds == y).astype(float)
        rows.append({"router": name, "n": len(y),
                     "accuracy": accuracy_score(y, preds),
                     "macro_f1": f1_score(y, preds, average="macro"),
                     "ECE": ece(conf, correct), **selective(conf, correct),
                     "avg_latency_ms": float(np.mean(lat))})
    out = pd.DataFrame(rows).round(3)
    print(out.to_string(index=False))
    out.to_csv("results_router_comparison.csv", index=False)


if __name__ == "__main__":
    main()
