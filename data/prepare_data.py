"""Download the Bitext customer-support dataset and write train/val/test CSVs with a 4-queue label.
Needs internet + `pip install datasets`.   Run:  python data/prepare_data.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.model_selection import train_test_split
from config import CATEGORY_TO_QUEUE

from datasets import load_dataset

ds = load_dataset("bitext/Bitext-Customer-Support-LLM-Chatbot-Training-Dataset", split="train")
df = ds.to_pandas()
print("columns:", list(df.columns)); print(df["category"].value_counts())

unknown = set(df["category"]) - set(CATEGORY_TO_QUEUE)
if unknown:
    print("WARNING: unmapped categories (add to config.CATEGORY_TO_QUEUE):", unknown)

df = df[df["category"].isin(CATEGORY_TO_QUEUE)].copy()
df["queue"] = df["category"].map(CATEGORY_TO_QUEUE)
df = df.rename(columns={"instruction": "text"})[["text", "queue", "category", "intent"]]

train, rest = train_test_split(df, test_size=0.2, stratify=df["queue"], random_state=42)
val, test = train_test_split(rest, test_size=0.5, stratify=rest["queue"], random_state=42)
for name, part in [("train", train), ("val", val), ("test", test)]:
    part.to_csv(f"data/{name}.csv", index=False)
    print(name, len(part))
