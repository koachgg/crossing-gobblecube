# Architecture Notes — M1 Exploration

## Repository Overview

Pedestrian crossing intent + trajectory prediction challenge.

- Input: 16 frames of history at 15 Hz (~1.07 s) per pedestrian.
- Output: P(crossing within 2s) + 4 future bounding boxes at 0.5/1.0/1.5/2.0 s.
- Data: video-disjoint train/dev split. No video or pedestrian overlaps between splits.
- Frame resolution: always 1920 × 1080 px.

### File Map

| File | Role |
|---|---|
| `predict.py` | Submission entry point. `predict(request) -> dict`. Do NOT change signature. |
| `baseline.py` | Training script. XGBoost on 20 hand-crafted features. |
| `grade.py` | Local grader + Docker grader mode. Computes composite score. |
| `data/schema.md` | Full column definitions and dataset notes. |
| `data/train.parquet` | 28,680 prediction windows, 1,482 unique pedestrians. |
| `data/dev.parquet` | 6,065 prediction windows, 282 unique pedestrians. |
| `model.pkl` | Serialized dict: `{"intent": XGBClassifier}` |
| `Dockerfile` | Python 3.11-slim. Target image size ≤ 2.5 GB (baseline ~2.02 GB). |
| `tests/test_predict.py` | 8 contract tests — shape, not quality. Must all pass. |

---

## Current Baseline Pipeline

### Training (baseline.py)

1. Load `train.parquet` + `dev.parquet`.
2. Featurize each row via `_engineered_features()`.
3. Train `XGBClassifier(n_estimators=300, max_depth=5, lr=0.05)` on intent label.
4. Serialize model as `{"intent": clf}` → `model.pkl`.

### Inference (predict.py)

Two independent components:

**Intent:**
- Load XGBClassifier from `model.pkl`.
- Run `_engineered_features()` → `predict_proba()`.
- Clip NaN/inf → return `intent` probability.

**Trajectory:**
- Pure constant-velocity extrapolation.
- Average velocity over last 4 intervals (`vx = diff(cx[-5:]).mean()`).
- Project forward at each horizon (`nx = cx[-1] + vx * h_frames`).
- Bounding box size held fixed at last observed size.
- **No model involved. Entirely heuristic.**

---

## Feature Engineering (20 features)

| # | Feature | Notes |
|---|---|---|
| 1-2 | `cx[-1]/fw`, `cy[-1]/fh` | Current normalized centre |
| 3-4 | `w[-1]/fw`, `h[-1]/fh` | Current normalized size |
| 5-6 | `vx[-4:].mean()/fw`, `vy[-4:].mean()/fh` | Short-window mean velocity |
| 7-8 | `vx.std()/fw`, `vy.std()/fh` | Velocity variance (jitter) |
| 9 | `(h/w).mean()` | Average aspect ratio (tallness) |
| 10 | `ego_available` | Bool flag |
| 11-13 | `ego_s.mean()`, `ego_s[-1]`, `ego_s.max()` | Ego speed stats |
| 14-16 | `ego_y.mean()`, `ego_y[-1]`, `abs(ego_y).max()` | Ego yaw stats |
| 17-18 | `time_of_day == daytime`, `nighttime` | One-hot |
| 19-20 | `weather == rain`, `weather == snow` | One-hot |

**Missing from features:**
- Acceleration (velocity change rate)
- Heading direction
- Pedestrian stopping / hesitation signals
- Trajectory curvature
- Absolute x-position relative to image thirds (proxy for curb proximity)
- `location` field completely unused
- Weather categories `clear` / `cloudy` not encoded
- Full velocity history (only last 4 frames used for mean; history ignored)

---

## Dataset Characteristics

| Property | Value |
|---|---|
| Train size | 28,680 rows |
| Dev size | 6,065 rows |
| Train positive rate | 7.93% |
| Dev positive rate | 9.08% |
| Class imbalance | ~12:1 negative:positive |
| ego_available=True | 92.8% of train |
| time_of_day populated | <8% of rows (most are empty string) |
| weather populated | <8% of rows |
| location populated | <8% of rows |

> **Implication:** Context metadata (time, weather, location) is sparse and unreliable.
> The model must primarily rely on bbox motion signals.

---

## Scoring

```
composite = 0.5 * (BCE / BCE_FLOOR) + 0.5 * (mean_pixel_ADE / ADE_FLOOR)
```

| Constant | Value | Meaning |
|---|---|---|
| `BCE_FLOOR` | 0.2488 | Entropy of class prior on Eval set |
| `ADE_FLOOR` | 49.80 px | Zero-velocity ADE on Eval set |

**Lower composite = better.** A zero-effort submission scores ~1.0.

---

## M2 Scores (Trajectory Improvements)

| Metric | Value (M1) | Value (M2) | Delta |
|---|---|---|---|
| **Composite score** | **0.8311** | **0.8231** | **-0.0080** |
| Intent term (BCE/floor) | 0.856 | 0.856 | 0.000 |
| Trajectory term (ADE/floor) | 0.806 | 0.790 | -0.016 |
| Raw mean ADE | 40.2 px | 39.4 px | -0.8 px |

**M2 Changes Implemented:**
- Swapped constant velocity for Exponential Moving Average (EMA) smoothed velocity (alpha=0.3).
- Added EMA-smoothed acceleration-aware projection to better model stopping/starting behavior.
- Added adaptive bounding box sizing (tracking EMA of width/height velocities) rather than fixed projection sizes.
- *Discarded:* Ego-motion compensation. Attempts to use ego-yaw and an estimated focal length degraded performance significantly (from 39px ADE to 300+px ADE), likely due to incomplete intrinsics or sign ambiguity.

---

## Identified Weaknesses (Updated post-M2)

### Trajectory (addressed in M2)

- The constant-velocity baseline was replaced with EMA+Acceleration.
- Further trajectory gains might require sequence models (M4) or external tracking state.

### Intent (higher ROI now — worth ~50% of score)

6. **Class imbalance not handled.** 7.9% positives, but no `scale_pos_weight` or SMOTE. XGBoost will under-predict the minority class.
7. **No acceleration feature.** Change in velocity is a direct signal for crossing intent (pedestrians slow down, stop, then cross).
8. **No stopping/hesitation signal.** A pedestrian with near-zero velocity variance who then starts moving is a strong crossing predictor.
9. **`location` field completely ignored.** Though sparse, it could be one-hot encoded.
10. **Weak temporal features.** Only mean + std of velocity used; no trend (is the pedestrian speeding up or slowing down?).
11. **Full history unused for ego yaw.** Only mean/last/max used; trajectory curvature due to ego turning is not decomputed.
12. **No calibration step.** XGBoost probabilities are known to be poorly calibrated; isotonic regression or Platt scaling could improve BCE.

---

## Improvement Roadmap (Prioritized)

### M2 — Trajectory (expected −0.05 to −0.15 on traj_term)

| Priority | Change | Rationale |
|---|---|---|
| 🔴 High | Velocity smoothing (e.g. exponential moving average or median over last 8 frames) | Reduces jitter accumulation at long horizons |
| 🔴 High | Acceleration-aware projection (fit linear velocity trend, extrapolate with deceleration) | ADE at +1.5/2.0 s is worst — non-constant motion is the primary driver |
| 🟡 Medium | Ego-motion subtraction (subtract camera-induced pixel shift using ego speed + yaw) | Improves velocity estimation when ego is moving |
| 🟡 Medium | Adaptive box size (estimate size change rate from history) | Marginal but free |

### M3 — Intent (expected −0.05 to −0.10 on intent_term)

| Priority | Change | Rationale |
|---|---|---|
| 🔴 High | Add acceleration features (velocity trend slope, speed_change = vx[-1] - vx[-8]) | Strong crossing signal |
| 🔴 High | Add stopping signal (near-zero velocity after motion) | Hesitation before crossing is common |
| 🟡 Medium | `scale_pos_weight` in XGBoost (or set `class_weight`) | Fix imbalance recall for the positive class |
| 🟡 Medium | Probability calibration (CalibratedClassifierCV / isotonic) | Directly improves BCE |
| 🟢 Low | One-hot `location` + full weather categories | Sparse but adds signal for populated subsets |
| 🟢 Low | Lateral position relative to frame thirds | Curb proximity proxy |

### M4 — Optional Sequence Model (only if M2+M3 plateau)

- Lightweight GRU over the 16-frame bbox sequence for trajectory.
- Must stay CPU-friendly and fit within 4 GB / 4 CPU limits.
- Risk: adds complexity; benefit uncertain given quality of current features.

### M5 — Finalization

- Docker validation.
- README with experiment log and score history.
- Dependency cleanup.
- Reproducibility audit.

---

## Known Constraints

- Offline inference (no external APIs at predict time)
- CPU-friendly (≤4 CPUs, ≤4 GB RAM)
- Dockerized submission (Python 3.11-slim, ≤2.5 GB image)
- `predict(request: dict) -> dict` signature must not change
- `model.pkl` must be included in image
- All 8 contract tests must pass