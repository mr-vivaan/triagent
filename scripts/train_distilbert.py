"""Fine-tune DistilBERT as an alternative router + temperature scaling for calibration.
(Phase 2 - needs torch + transformers; use a free Colab GPU.)
python scripts/train_distilbert.py --train data/train.csv --val data/val.csv --out models/distilbert"""
import argparse, os
import numpy as np, pandas as pd, torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)


class DS(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def fit_temperature(logits, y):
    """Grid-search T minimising NLL on validation logits."""
    best_T, best = 1.0, 1e9
    for T in np.arange(0.5, 5.01, 0.05):
        z = logits / T; z = z - z.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        nll = -logp[np.arange(len(y)), y].mean()
        if nll < best:
            best_T, best = float(T), nll
    return best_T


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.csv"); ap.add_argument("--val", default="data/val.csv")
    ap.add_argument("--out", default="models/distilbert"); ap.add_argument("--epochs", type=int, default=3)
    a = ap.parse_args()

    tr, va = pd.read_csv(a.train), pd.read_csv(a.val)
    labels = sorted(tr.queue.unique()); l2i = {l: i for i, l in enumerate(labels)}
    name = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=l2i)

    enc = lambda df: tok(df.text.tolist(), truncation=True, padding=True, max_length=128)
    tr_ds = DS(enc(tr), [l2i[x] for x in tr.queue]); va_y = [l2i[x] for x in va.queue]
    va_ds = DS(enc(va), va_y)

    trainer = Trainer(model=model, train_dataset=tr_ds, args=TrainingArguments(
        output_dir="runs", num_train_epochs=a.epochs, per_device_train_batch_size=32,
        learning_rate=5e-5, save_strategy="no", report_to="none"))
    trainer.train()

    logits = trainer.predict(va_ds).predictions
    T = fit_temperature(logits, np.array(va_y))
    print(f"val accuracy={np.mean(logits.argmax(1) == np.array(va_y)):.3f}  temperature={T:.2f}")

    os.makedirs(a.out, exist_ok=True)
    trainer.save_model(a.out); tok.save_pretrained(a.out)
    open(os.path.join(a.out, "temperature.txt"), "w").write(str(T))
