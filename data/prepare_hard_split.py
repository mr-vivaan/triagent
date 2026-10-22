"""Create intent-holdout (hard) splits for Triagent.
Unseen intents are placed in test_hard.csv so the router must generalize
semantically rather than memorizing intent templates.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.model_selection import train_test_split

# 1-2 representative intents held out per queue
HELD_OUT_INTENTS = {
    "billing": ["check_cancellation_fee", "check_refund_policy"],
    "orders": ["cancel_order", "change_shipping_address"],
    "account": ["delete_account", "newsletter_subscription"],
    "general": ["complaint"],
}

def main():
    parts = []
    for split in ["train", "val", "test"]:
        p = f"data/{split}.csv"
        if os.path.exists(p):
            parts.append(pd.read_csv(p))
    df = pd.concat(parts, ignore_index=True)

    held_out_list = [intent for group in HELD_OUT_INTENTS.values() for intent in group]
    print(f"Total dataset size: {len(df)}")
    print(f"Held out intents ({len(held_out_list)}): {held_out_list}")

    test_hard = df[df["intent"].isin(held_out_list)].copy()
    rest = df[~df["intent"].isin(held_out_list)].copy()

    # Split remaining into train_hard (85%) and val_hard (15%)
    train_hard, val_hard = train_test_split(
        rest, test_size=0.15, stratify=rest["queue"], random_state=42
    )

    train_hard.to_csv("data/train_hard.csv", index=False)
    val_hard.to_csv("data/val_hard.csv", index=False)
    test_hard.to_csv("data/test_hard.csv", index=False)

    print(f"train_hard: {len(train_hard)} rows")
    print(f"val_hard:   {len(val_hard)} rows")
    print(f"test_hard:  {len(test_hard)} rows")
    print("\nQueue distribution in test_hard:")
    print(test_hard["queue"].value_counts())

if __name__ == "__main__":
    main()
