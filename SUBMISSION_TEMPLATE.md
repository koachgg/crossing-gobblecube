# Submission: Belo Abhigyan

---

## Final Dev Score

```
Score: 0.8190  (intent_term 0.848, traj_term 0.790; BCE 0.2109, ADE 39.4 px)
```

Evaluated on the grader-default 5,000-row sample of `dev.parquet` (seed=42).  
Full 6,065-row dev score: **0.8245** (BCE 0.2168, ADE 38.73 px).

---

## Approach

**Intent model:** XGBoost classifier retrained on 31 hand-engineered tracklet features. The original 20-feature baseline was extended with: short-term vs long-term horizontal velocity contrast, explicit acceleration (Δv over an 8-frame window), scalar speed magnitude, a binary stopping signal (speed < 1 px/frame), distance from frame center, aspect-ratio change (to detect body rotation), and one-hot encodings for all `weather` and `location` values. No class rebalancing was applied — XGBoost's native log-loss minimisation produces well-calibrated probabilities on this dataset, and adding `scale_pos_weight` or isotonic calibration both degraded BCE.

**Trajectory model:** Pure kinematic heuristic — no neural network. Per-axis EMA-smoothed velocity (α=0.3) over the full 16-frame history, with a secondary EMA-smoothed acceleration term (α=0.1) to model deceleration and restart events. Bounding box size is also EMA-smoothed to handle gradual size changes. The acceleration term is damped at long horizons (× 0.5 on the h² coefficient) to prevent quadratic blow-up when acceleration noise exceeds signal.

---

## What Didn't Work

1. **Ego-motion compensation.** Subtracting camera-induced apparent motion using ego-yaw and an estimated focal length destroyed trajectory ADE (from ~39 px to >300 px). Without verified camera intrinsics the geometric correction is too sensitive to calibration error.

2. **Class balancing (`scale_pos_weight`).**  With the 7.9% positive rate, setting `scale_pos_weight ≈ 11.6` pushed probabilities toward 1.0, increasing BCE from 0.2168 to 0.2212 on the full dev set. XGBoost is already minimising log-loss directly; reweighting shifts the decision boundary without improving calibration.

3. **GRU residual correction.** A tiny GRU (hidden=16, 673 total parameters) trained with NumPy-only inference to predict residuals from the EMA heuristic worsened ADE from 38.7 to 43.1 px on the full dev set. With only 1.07 s of past history and high-frequency bbox noise, the sequence model learns to overfit the residual distribution in-sample. The kinematic model generalises better.

---

## Where AI Tooling Helped

Used AI assistance throughout: feature ideation, debugging (catching the float64/float32 dtype error in the GRU training loop), writing the reliability audit script, and drafting this README. The workflow was fast for mechanical tasks (editing, running experiments, updating documentation). It was less useful for deciding *what to try* — that still required reading the data distribution and understanding where the scoring formula was most sensitive.

---

## Next Experiments

1. **Median/Huber loss on trajectory.** MSE rewards large-residual corrections; Huber would be more robust to the long-tail trajectories that dominate ADE at 2.0 s.
2. **Cross-attention intent features.** Feed the EMA velocity profile as a sequence into a single-head attention layer (no GPU required, < 5 k params) rather than hand-engineering the velocity contrast.
3. **Camera-intrinsics recovery.** If per-video focal length can be estimated from the vanishing point of straight-line trajectories, ego-motion compensation becomes feasible.
4. **Conditional trajectory by intent.** Pedestrians about to cross follow a different trajectory distribution than those staying on the kerb. Conditioning trajectory prediction on the intent probability could reduce the mixed-population ADE.

---

## Reproducibility

```bash
pip install -r requirements.txt
python baseline.py        # retrains model.pkl (~30 s on CPU)
python grade.py           # scores 5k dev sample, prints Score: 0.8190
python -m pytest tests/   # 8/8 tests should pass
```

Docker:
```bash
docker build -t crossing .
docker run --rm --network=none \
  -v $(pwd)/data:/data \
  crossing /data/dev.parquet /data/preds.csv
```

---

## External Data / Pretrained Weights

None. Training uses only `data/train.parquet` (derived from JAAD/PIE, provided by the challenge).

---

*Approximate time: ~6 hours across 4 working sessions.*
