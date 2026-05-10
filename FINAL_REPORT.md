# FINAL_REPORT.md

**Candidate:** Belo Abhigyan  
**Branch at submission:** `m4-sequence-model` (working off `m3-intent`)  
**Date:** 2026-05-10

---

## 1. Final Scores

| Evaluation | Score | BCE | ADE |
|---|---|---|---|
| Grader default — 5k sample, seed=42 | **0.8190** | 0.2109 | 39.4 px |
| Full dev set — 6,065 rows | **0.8245** | 0.2168 | 38.73 px |
| Original baseline (M1) | 0.8311 | 0.2129 | 40.2 px |
| **Total improvement (grader)** | **-0.0121** | **-0.0020** | **-0.8 px** |
| **Total improvement (full dev)** | **-0.0066** | | |

Score components at final state:

| Term | Value | Floor | Ratio |
|---|---|---|---|
| Intent (BCE) | 0.2109 | 0.2488 | **0.848** |
| Trajectory (ADE) | 39.4 px | 49.80 px | **0.790** |

---

## 2. Milestone Summary

### M1 — Baseline Understanding
- Ran and profiled the starter: composite 0.8311, ADE error concentrated at long horizons (77 px at +2.0 s).
- Identified two independent, roughly equal-weight improvement axes: intent features and trajectory smoothing.
- Documented findings in `ARCHITECTURE.md`.

### M2 — Trajectory Heuristic
- Replaced the 4-frame mean-velocity estimator with a 16-frame EMA velocity smoother (α=0.3).
- Added EMA-smoothed acceleration to the projection (α=0.1, damped at long horizons).
- Added EMA-smoothed width/height tracking for adaptive bounding box sizing.
- **Result:** ADE 40.2 → 39.4 px (−0.8 px), composite 0.8311 → 0.8231.

### M3 — Intent Feature Engineering
- Extended feature set from 20 to 31 features.
- Key additions: acceleration (Δv over 8 frames), speed magnitude, stopping signal, distance to frame centre, aspect-ratio change (body rotation proxy), full metadata one-hot encoding.
- Retuned XGBoost (`lr=0.03`, `n_estimators=400`).
- Tested and discarded `scale_pos_weight` and `CalibratedClassifierCV` — both degraded BCE.
- **Result:** BCE 0.2129 → 0.2109, composite 0.8231 → 0.8190.

### Reliability Audit
- Confirmed grader uses 5k sample (seed=42), not the full 6,065-row dev set.
- Measured score std = ±0.0044 across 10 seeds — noise floor for meaningful improvements.
- Confirmed zero train/dev leakage (disjoint ped_id sets, disjoint (ped_id, frame) pairs).
- Confirmed `predict()` is bit-deterministic.

### M4 — Sequence Model Experiment (Aborted)
- Trained a GRU (hidden=16, 673 parameters) to predict trajectory residuals from the M2 kinematic baseline.
- Implemented a pure NumPy forward-pass for zero Docker footprint.
- ADE worsened: 38.73 → 43.10 px on the full dev set.
- **Reverted.** Kinematic model generalises better under high-frequency bbox noise.

---

## 3. Key Engineering Decisions

| Decision | Rationale |
|---|---|
| EMA over sliding-window mean velocity | EMA uses all 16 frames with exponential decay; sliding window throws away ~75% of available history |
| Dampened acceleration (× 0.5 on h²) | Prevents quadratic blow-up when acceleration noise dominates at long horizons |
| No `scale_pos_weight` | XGBoost minimises log-loss directly; reweighting shifts boundary without improving calibration |
| GRU reverted | With 1.07 s of noisy input history and no map context, a sequence model overfits residuals in-sample |
| Pure NumPy GRU inference | Eliminates PyTorch as an inference dependency, keeping Docker image small |
| Full dev evaluation for M4 | Used `audit_reliability.py` (all 6,065 rows) to avoid the ±0.0044 noise floor of the 5k sampler |

---

## 4. Tradeoffs

- **EMA α=0.3** is a moderate smoothing choice. Lower α over-smooths (ignores recent abrupt direction changes); higher α reverts toward noisy instantaneous velocity.
- **Stopping signal threshold at 1 px/frame** (≈1.5 cm/s at typical image scale) was chosen heuristically. A threshold tuned on dev positives vs negatives might improve intent recall.
- **Metadata one-hot encoding** added 8 features. These fields are sparse and infrequently populated; they contribute marginal information. They did not hurt and were cheap to add.
- **No ego-motion compensation.** Without verified focal length, the geometric correction is more likely to inject noise than remove it.

---

## 5. Lessons

1. **Noise floor matters before claiming wins.** Small improvements (< 0.005) on a 5k sample are within the sampling variance. Always confirm on the full dev set.
2. **Kinematic priors outperform black-box models at short sequence lengths.** A GRU with 1 second of history has fewer effective degrees of freedom than the problem's noise level. Physics-motivated constraints (EMA, damped acceleration) generalise better.
3. **BCE calibration is fragile.** Adding any form of class reweighting or post-hoc calibration degraded log-loss because XGBoost's native estimator is already well-calibrated for this data size and class balance ratio.
4. **git history matters.** Structured milestones with focused commits give a clear experimental trace.

---

## 6. Repository Cleanliness Checklist

| Item | Status |
|---|---|
| `predict.py` — clean, commented, no dead code | ✅ |
| `baseline.py` — retrains model from scratch | ✅ |
| `model.pkl` — current weights | ✅ |
| `requirements.txt` — minimal, no torch in inference deps | ✅ |
| `Dockerfile` — unchanged from reference; builds ≤ 2 GB | ✅ |
| `tests/` — 8/8 passing | ✅ |
| Scratch/temp files removed | ✅ |
| `temp_arena/` directory removed | ✅ |
| `scratch_traj.py` removed | ✅ |
| `audit_reliability.py` removed | ✅ |
| `SUBMISSION_TEMPLATE.md` filled in | ✅ |
| `ARCHITECTURE.md` updated through M4 | ✅ |
