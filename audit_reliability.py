"""
Benchmarking Reliability Audit
Tasks:
  1. Full dev set score vs sampled score
  2. Score variance across different sample seeds (simulate instability)
  3. Determinism check (same inputs -> same outputs)
  4. Train/dev leakage check (ped_id overlap)
  5. Feature output determinism
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from predict import predict, _engineered_features
from grade import score, _flatten, REQUEST_FIELDS, HORIZONS

DATA = Path("data")
OUT_COLS = ["ped_id", "intent"] + [f"{h}_{c}" for h in HORIZONS for c in ("x1","y1","x2","y2")]
GRADER_SAMPLE_N = 5000
GRADER_SEED     = 42

# ------------------------------------------------------------------
# 1. FULL DEV SET vs. 5k SAMPLE SCORE
# ------------------------------------------------------------------
print("=" * 60)
print("1. FULL DEV vs 5k SAMPLE SCORE")
print("=" * 60)

dev = pd.read_parquet(DATA / "dev.parquet")
print(f"   Full dev size : {len(dev):,} rows")
print(f"   Grader sample : {GRADER_SAMPLE_N:,} rows (seed={GRADER_SEED})")

def run_score(df):
    records = df[REQUEST_FIELDS].to_dict("records")
    flat = [_flatten(predict(r), r["ped_id"]) for r in records]
    preds_df = pd.DataFrame(flat, columns=OUT_COLS)
    return score(preds_df, df)

# Full dev
s_full = run_score(dev)

# Grader-style sample
dev_sample = dev.sample(n=GRADER_SAMPLE_N, random_state=GRADER_SEED).reset_index(drop=True)
s_sample = run_score(dev_sample)

print(f"\n   Full  dev : score={s_full['score']:.4f}  BCE={s_full['intent_bce']:.4f}  ADE={s_full['mean_ade_px']:.2f}px")
print(f"   5k sample : score={s_sample['score']:.4f}  BCE={s_sample['intent_bce']:.4f}  ADE={s_sample['mean_ade_px']:.2f}px")
print(f"   Delta     : score={s_full['score']-s_sample['score']:+.4f}  BCE={s_full['intent_bce']-s_sample['intent_bce']:+.4f}")

# ------------------------------------------------------------------
# 2. VARIANCE ACROSS DIFFERENT SAMPLE SEEDS
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. SCORE VARIANCE ACROSS 10 DIFFERENT SAMPLE SEEDS")
print("=" * 60)

SEEDS = [0, 1, 7, 13, 42, 99, 123, 256, 512, 999]
seed_scores = []
for seed in SEEDS:
    s_dev = dev.sample(n=GRADER_SAMPLE_N, random_state=seed).reset_index(drop=True)
    s = run_score(s_dev)
    seed_scores.append(s['score'])
    print(f"   seed={seed:4d} : score={s['score']:.4f}")

scores_arr = np.array(seed_scores)
print(f"\n   Mean  : {scores_arr.mean():.4f}")
print(f"   Std   : {scores_arr.std():.4f}")
print(f"   Min   : {scores_arr.min():.4f}")
print(f"   Max   : {scores_arr.max():.4f}")
print(f"   Range : {scores_arr.max()-scores_arr.min():.4f}")

# ------------------------------------------------------------------
# 3. DETERMINISM CHECK
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. DETERMINISM CHECK (same request -> same output, 5 repeats)")
print("=" * 60)

test_row = dev_sample.iloc[0]
req = {k: test_row[k] for k in REQUEST_FIELDS}

results = [predict(req) for _ in range(5)]
intents = [r["intent"] for r in results]
bboxes  = [r["bbox_2000ms"] for r in results]

intent_same = len(set(intents)) == 1
bbox_same   = all(bboxes[i] == bboxes[0] for i in range(1, 5))

print(f"   Intent values : {[round(i, 6) for i in intents]}")
print(f"   Intent deterministic : {intent_same}")
print(f"   Bbox deterministic   : {bbox_same}")

# ------------------------------------------------------------------
# 4. TRAIN / DEV LEAKAGE CHECK
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. TRAIN / DEV LEAKAGE CHECK")
print("=" * 60)

train = pd.read_parquet(DATA / "train.parquet")
train_peds = set(train["ped_id"].unique())
dev_peds   = set(dev["ped_id"].unique())
overlap    = train_peds & dev_peds

print(f"   Train unique ped_ids : {len(train_peds):,}")
print(f"   Dev unique ped_ids   : {len(dev_peds):,}")
print(f"   Overlapping ped_ids  : {len(overlap)}")
print(f"   Leakage              : {'YES - PROBLEM' if overlap else 'None detected'}")

# Also check requested_at_frame overlap (temporal)
train_frames = set(zip(train["ped_id"], train["requested_at_frame"]))
dev_frames   = set(zip(dev["ped_id"], dev["requested_at_frame"]))
frame_overlap = train_frames & dev_frames
print(f"   (ped_id, frame) exact overlap : {len(frame_overlap)}")

# ------------------------------------------------------------------
# 5. FEATURE DETERMINISM
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. FEATURE VECTOR DETERMINISM (same row, 3 calls)")
print("=" * 60)

f1 = _engineered_features(req)
f2 = _engineered_features(req)
f3 = _engineered_features(req)
feat_match = np.allclose(f1, f2) and np.allclose(f1, f3)
print(f"   Feature shape      : {f1.shape}")
print(f"   Features identical : {feat_match}")
print(f"   Any NaN            : {np.isnan(f1).any()}")
print(f"   Any Inf            : {np.isinf(f1).any()}")

# ------------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"   Grader uses 5k sample (not full 6,065) — introduces sampling noise.")
print(f"   Sample seed=42 is fixed in grade.py -> score IS reproducible.")
print(f"   Score std across 10 random seeds  : {scores_arr.std():.4f}")
print(f"   Full dev score vs grader sample   : {abs(s_full['score']-s_sample['score']):.4f} delta")
print(f"   Predict() is deterministic        : {intent_same and bbox_same}")
print(f"   Train/dev ped_id leakage          : {'YES' if overlap else 'None'}")
print(f"   Feature vector is deterministic   : {feat_match}")
