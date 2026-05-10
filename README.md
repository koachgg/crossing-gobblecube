# The Crossing Challenge

*A Gobblecube take-home: build the pedestrian-crossing predictor.*

---

## Solution — Belo Abhigyan

### Final Score

```
Score: 0.8190  (intent_term 0.848, traj_term 0.790; BCE 0.2109, ADE 39.4 px)
```

Evaluated using the grader-default 5k sample of `dev.parquet` (seed=42).  
Full 6,065-row dev score: **0.8245** (BCE 0.2168, ADE 38.73 px).

### What Changed vs the Baseline

**Trajectory — EMA Kinematic Model (M2)**

The constant-velocity estimator was replaced with a two-stage smoother:

1. **EMA velocity (α=0.3)** over all 16 history frames — reduces per-frame bbox detection jitter without discarding recent motion information.
2. **EMA acceleration (α=0.1)** — models deceleration events (pedestrian stopping before crossing) and restart events. The acceleration term is damped by 0.5 on the h² coefficient to prevent quadratic blow-up at the 2-second horizon where acceleration noise exceeds signal.
3. **Adaptive bbox size** — EMA-smoothed width/height velocity tracks gradual size changes as the pedestrian moves towards or away from the camera.

This is a deterministic, CPU-friendly heuristic with no external dependencies.

**Intent — Expanded Feature Set (M3)**

Extended from 20 to 31 hand-engineered features:

| Feature group | New features added |
|---|---|
| Velocity | Short-term (4f) vs long-term (12f) horizontal velocity contrast |
| Acceleration | Explicit Δv over 8-frame window |
| Speed | Scalar speed magnitude (last frame, mean over history) |
| Stopping | Binary flag: speed < 1 px/frame |
| Structure | Distance to frame horizontal centre; aspect-ratio change (body rotation proxy) |
| Metadata | Full one-hot for all `weather` and `location` values |

XGBoost was retuned (`lr=0.03`, `n_estimators=400`). No class rebalancing or post-hoc calibration was applied — both degraded BCE on the dev set (tested explicitly).

### What Didn't Work

1. **Ego-motion compensation.** Subtracting camera-induced motion via ego-yaw and an estimated focal length pushed ADE from ~39 px to >300 px. Without verified camera intrinsics the geometric correction is too sensitive to calibration error. `ego_speed_history` and `ego_yaw_history` are retained as XGBoost intent features where they carry useful context.

2. **Class rebalancing.** `scale_pos_weight ≈ 11.6` (negatives/positives) shifted probabilities toward 1.0 and increased BCE from 0.2109 to 0.2163. XGBoost already minimises log-loss directly; reweighting adjusts the decision boundary without improving calibration.

3. **GRU residual correction (M4).** A 673-parameter GRU trained to predict trajectory residuals from the kinematic baseline, with a pure NumPy forward pass in `predict.py` for zero Docker footprint. ADE worsened from 38.73 to 43.10 px on the full 6,065-row dev set. With only 1.07 s of noisy bbox history and no map context, the sequence model overfits the residual distribution in-sample. Kinematic model reverted.

### Benchmark Methodology

A [reliability audit](ARCHITECTURE.md) confirmed:
- The default `python grade.py` uses a 5k sample (seed=42), not the full 6,065-row dev set. Score std across 10 seeds = **±0.0044** — the effective noise floor for meaningful improvements.
- `predict()` output is bit-deterministic.
- Zero train/dev leakage (disjoint ped_id sets, confirmed by exact (ped_id, frame) pair check).
- All M4 evaluations were run on the full dev set to avoid the sampler noise floor.

### Next Experiments

1. **Conditional trajectory by intent.** Pedestrians with high crossing intent follow a different trajectory distribution. Conditioning trajectory prediction on the intent probability could reduce the mixed-population ADE.
2. **Huber loss for trajectory.** MSE over-weights large-residual tails; Huber loss would be more robust at the 2.0 s horizon.
3. **Focal-length estimation.** If per-video focal length can be recovered from vanishing-point geometry of straight-line trajectories, ego-motion compensation becomes feasible.
4. **Cross-attention intent.** Feed the EMA velocity profile as a short sequence into a single-head attention layer (< 5k params) rather than compressing it into scalar features.

### Reproduce From Scratch

```bash
pip install -r requirements.txt
python baseline.py        # retrains model.pkl (~30 s on CPU)
python grade.py           # Score: 0.8190
python -m pytest tests/   # 8/8 passed
```

### Docker

```bash
docker build -t crossing .
docker run --rm --network=none \
  -v $(pwd)/data:/data \
  crossing /data/dev.parquet /data/preds.csv
```

---


## The Problem

You've joined the perception team at a startup building a fleet of
slow-speed neighborhood autonomous delivery vehicles. At every
intersection the vehicle has to decide, in real time, whether a nearby
pedestrian is about to step into the road, and where they're going to
be two seconds from now. Get it wrong in one direction and the vehicle
hesitates for every pedestrian in San Francisco. Get it wrong in the
other direction and someone gets hurt.

Your job: given the last second of a pedestrian's observed tracklet and
the vehicle's own motion, predict two things:

1. **Will they cross in the next 2 s?** A single probability.
2. **Where will they be?** Bounding boxes at +0.5, +1.0, +1.5, +2.0 s.

This is the kind of problem real self-driving companies spend a lot of
money on. We want to see how you approach it.

---

## What You Ship

A Python function:

```python
def predict(request: dict) -> dict:
    """
    Input: one request dict with these keys (see data/schema.md):
        ped_id, frame_w, frame_h,
        time_of_day, weather, location,    # may be empty strings
        ego_available,                     # True when ego motion is valid
        bbox_history,                      # list[16] of [x1,y1,x2,y2], 15 Hz
        ego_speed_history,                 # list[16] of floats (m/s)
        ego_yaw_history,                   # list[16] of floats (rad/s)
        requested_at_frame,                # int, native 30 fps frame id

    Output: dict with these keys:
        intent       float in [0, 1]
        bbox_500ms   [x1,y1,x2,y2]
        bbox_1000ms  [x1,y1,x2,y2]
        bbox_1500ms  [x1,y1,x2,y2]
        bbox_2000ms  [x1,y1,x2,y2]
    """
```

Packaged as a GitHub repo containing:

- `predict.py` exposing the function above
- A `Dockerfile` that builds your submission
- Your trained model weights (`model.pkl` or equivalent)
- A `README.md` describing your approach

**Constraints:**

- Total Docker image ≤ 2 GB
- 4 GB RAM / 4 CPUs / 30-minute wall-clock cap at scoring
- No external API calls at inference time. The container runs with
  `--network=none`

We call `predict()` **row-by-row** against the held-out Eval parquet;
output order must match input order exactly.

---

## The Data

We hand you pre-built tracklets derived from two public datasets (JAAD
and PIE). **You do not need to download the raw videos.**

- `data/train.parquet`: ~29 k training windows
- `data/dev.parquet`: ~6 k dev windows for your own grading

Each row is a prediction window: 16 frames of bbox + ego history at
15 Hz (≈ 1.07 s of past), plus the ground-truth future. The split holds
out entire **videos**. No training video appears in Dev or Eval.

The Eval set we grade on is the same shape and schema, just different
videos. Schema details: `data/schema.md`.

---

## Scoring

**Your score = intent skill + trajectory skill, averaged.** Each term is
normalized by a zero-work floor, so 1.0 = "did literally nothing,"
0.0 = "perfect."

```
intent_term = BCE(intent) / BCE_FLOOR
traj_term   = mean_pixel_ADE / ADE_FLOOR
score       = 0.5 * intent_term + 0.5 * traj_term
```

- `BCE(intent)` is binary cross-entropy against the `will_cross_2s`
  label (natural log). Non-finite intents are replaced with 0.5 before
  clipping to `[1e-6, 1-1e-6]`.
- `mean_pixel_ADE` is the mean Euclidean pixel distance between your
  predicted bbox centers and the truth, averaged over all 4 horizons.
  Non-finite bbox coordinates are replaced with center-of-frame and
  clamped to `[-2000, 4000]`.
- `BCE_FLOOR = 0.2488`: entropy of the class prior on Eval.
- `ADE_FLOOR = 49.80 px`: zero-velocity mean ADE on Eval.

**Lower is better.** **1.0 = you tied the zero-work floor** (class-prior
intent + predict the current bbox for every horizon). **0.0 = perfect**.
**>1.0 = you did worse than doing nothing**. There's no ceiling, just a
real floor at zero.

For reference, measured on the held-out Eval set:

| Approach | Eval score |
|---|---|
| Class-prior intent + zero-velocity trajectory (the floor) | 1.00 |
| Class-prior intent + constant-velocity trajectory | ~0.81 |
| **GBT intent + constant-velocity trajectory (this repo)** | **0.74** |

Use Dev as a self-check. Baseline scores 0.83 on Dev vs 0.74 on Eval,
a ~0.09 gap driven by Dev having a slightly higher positive rate.
Budget for landing within about 0.05–0.10 of your Dev number on the
real grader. Beat the baseline by as much as you can. There's no
posted target; we'll tell you how your number stacks up.

The baseline's trajectory ADE explodes at long horizons: **7.9 px** at
+0.5 s, **18.7 px** at +1.0 s, **37.4 px** at +1.5 s, **61.1 px** at
+2.0 s. Constant velocity doesn't model acceleration, turns, or
"pedestrian saw the vehicle and stopped." That's where the points are.

---

## Submission

- Push your repo and send us the link when your submission is
  something you'd put your name on.
- One submission per candidate. Do not share work with other candidates.

---

## Rules

- Use any ML framework, any pretrained model, any language inside the
  container (entry point must be Python to match the grader).
- You **may** use additional public datasets for pretraining / data
  augmentation. Document them in your repo README.
- No external API calls at scoring time. Scoring runs offline.
- No ensembling across candidates. Your submission is one image.

---

## Baseline

```bash
python baseline.py     # trains GBT intent classifier, ~5 s on laptop CPU
python grade.py        # scores a 5 k Dev sample, ~2 s
```

What the baseline does:

- `baseline.py` fits an `XGBoost` classifier on 20 engineered tracklet
  features (bbox size & aspect ratio, past velocities, ego speed/yaw
  stats, scene-condition flags). No class rebalancing; log-loss wants
  calibrated probabilities.
- `predict.py` trajectory is pure constant-velocity extrapolation from
  the last 4 bbox centers. Misses acceleration, turns, and the classic
  "pedestrian changed their mind" case.

### Repo layout

```
predict.py              entry point (candidates: edit this)
baseline.py             training script
grade.py                local dev grader; scoring logic matches the grader
Dockerfile              reference build
requirements.txt
model.pkl               built by baseline.py (candidates: overwrite this)
data/
  train.parquet         training windows
  dev.parquet           dev windows (self-grade against this)
  schema.md             column-by-column reference
  tracklets_raw.parquet internal pre-windowing artifact; ignore unless you
                        want to re-slice (we built it with build_tracklets.py)
  build_tracklets.py    parses JAAD/PIE annotations (internal)
  build_windows.py      window slicer + video-ID splitter (internal)
tests/
  test_predict.py       shape/contract tests
```

### Running order

1. `pip install -r requirements.txt`
2. `python baseline.py` (produces `model.pkl`)
3. `python grade.py` (validates on Dev, prints score)
4. `python -m pytest tests/` (shape/contract tests; note `-m pytest`)
5. `docker build -t my-crossing .` then `docker run ...` to verify
   container runs end-to-end.

Your job is to ship something better than `baseline.py`.

---

## FAQ

**Is this hard?**
Yes. The baseline is deliberately honest. Meaningfully beating it takes
focus and good engineering judgment. The strongest approaches to this
type of problem train a sequence model end-to-end.

**What AI tooling should I use?**
Whatever helps you ship. Claude Code is our in-house default and the
fastest path we've seen on these challenges, but Cursor, Aider, Copilot,
ChatGPT, direct API calls, or no LLM at all are all fine. The role is
about shipping fast with AI pair-programming generally, not about any
one tool. Your git history is part of the signal. We read commits, not
just the final state. Real iteration with AI help looks different from a
polished from-scratch dump.

**I've never trained a sequence model. Should I apply?**
Yes. The baseline is CPU-only. You can beat it by engineering better
intent features and improving the constant-velocity trajectory, no deep
learning required. The strongest approaches to this type of problem
typically use a small sequence model, though, and the trajectory term
is where they win. Picking one up for this is very doable.

**Your data is dashcam-at-30mph and your product framing is a sidewalk
robot. What gives?**
The public data we can use under a permissive license comes from
vehicle-mounted cameras. The perception problem (given a pedestrian's
tracklet plus ego motion, predict intent and future path) transfers to
slow-speed urban AVs. Candidates who notice and handle the ego-motion
distribution shift in their training pipeline tend to score better.

**What if I don't have access to a GPU?**
You can submit a valid entry without one. The baseline is CPU-only.
Strong submissions may require GPU training, though, and free-tier
notebook environments (Kaggle, Colab, Lightning.ai) each offer ~30
GPU-hours per week, which has typically been enough. More detail in
the Resources section.

**What do you actually care about?**
Your final Eval score, your README, and your git log, roughly in that
order. A clean submission with a thoughtful write-up will beat a
slightly better score with no explanation.

**Can I collaborate with a friend?**
No. Individual submissions only.

**How do I submit?**
Push your repo public on GitHub and send us the link. Your Dockerfile
should build directly from the repo root. If you need your repo to
stay private, add `gobblecube-hiring` as a read-only collaborator on
GitHub.

---

## Resources

### Libraries we've seen work well
`pandas`, `numpy`, `polars`, `scikit-learn`, `xgboost`, `lightgbm`,
`torch`, `transformers`, `pytorch-lightning`.

### Datasets you may find useful
- **JAAD** (York U., MIT license): pedestrian-action-annotated dashcam
  data. Used for part of our train/dev/eval windows.
  https://github.com/ykotseruba/JAAD
- **PIE** (York U., MIT license): pedestrian-intent-annotated dashcam
  data with OBD ego speed/yaw. The bulk of our data.
  https://github.com/aras62/PIE
- **PedestrianActionBenchmark (WACV '21)**: reference SOTA numbers for
  PIE intent and action prediction. Different scoring metric than ours
  (AUC/accuracy, not BCE), but the techniques transfer.

### Compute

The baseline trains in ~5 s on a laptop CPU. You do not need a GPU to
submit a valid entry.

Deep-learning approaches benefit substantially from GPU training. If
you don't have one:
free-tier notebook environments (Kaggle, Colab, Lightning.ai) each
offer ~30 GPU-hours per week, which has typically been enough. How
you use that compute is part of the test.

### Things that will disqualify you

- Re-extracting tracklets from the JAAD or PIE raw video data to look
  up Eval pedestrians' ground-truth futures. Our Eval ped IDs are
  hashed, but a motivated candidate could still fingerprint them by
  bbox trajectory. **Don't.** We review training code.
- Submitting something that does not run inside `--network=none` with
  4 GB RAM / 4 CPUs.
- Hardcoding per-request predictions (we fuzz requests).
- Calling any external API or model at inference time.

---

## What We Actually Care About

We are **not** hiring a specialist self-driving ML engineer. We don't
run a self-driving company. We use this problem because it's
well-defined, has open data, and is outside our business.

What we're hiring for: **an engineer who can pick up a problem they've
never seen before, pair effectively with modern AI tooling, and ship
something that works.** The Crossing problem is just an excuse to watch
you do that.

Your submission tells us three things:

1. **Do you ship?** The number on the leaderboard.
2. **Can you learn fast?** Your git log shows the trajectory.
   First commits usually look nothing like final commits.
3. **Can you reason about a problem that isn't handed to you as a
   spec?** Your README explains what you tried, what failed, and what
   the next experiment would be if you kept going.

You don't need an ML background to do well here. The difference is
rarely background. It's almost always mindset.

---

*Submit your repo URL and LinkedIn profile to agentic-hiring@gobblecube.ai. Questions welcome at the same address.*
