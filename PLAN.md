# Crossing Challenge Plan

## Objective

Improve the provided baseline for pedestrian crossing intent and trajectory prediction.

Goals:
- beat baseline dev score
- maintain lightweight CPU-friendly inference
- preserve Docker compatibility
- keep code understandable and reproducible
- produce strong README and git history

This is an engineering challenge, not a research benchmark.

---

# Repository Context

Important files:
- baseline.py
- predict.py
- grade.py
- data/schema.md
- tests/

Scoring:
- lower is better
- score combines:
  - intent BCE
  - trajectory ADE

Trajectory quality is likely highest ROI.

---

# Engineering Principles

Prioritize:
1. measurable improvements
2. simple robust approaches
3. clean implementation
4. reproducibility
5. fast iteration

Avoid:
- overengineering
- giant frameworks
- unnecessary abstractions
- brittle pipelines

---

# Planning Instructions

When planning a milestone:

1. Inspect current implementation.
2. Identify baseline weaknesses.
3. Propose multiple improvement ideas.
4. Estimate:
   - implementation complexity
   - likely score impact
   - inference cost
5. Recommend the highest-ROI approach.
6. Produce:
   - implementation steps
   - validation steps
   - rollback considerations
   - expected risks

Write plans into:
- PLAN-MX.md

Do NOT immediately implement during planning.

---

# Implementation Instructions

When implementing a milestone:

1. Read the relevant PLAN-MX.md file.
2. Implement incrementally.
3. Keep changes scoped.
4. Preserve existing APIs.
5. Run validation frequently.
6. Benchmark against grade.py.
7. Document score changes.
8. Avoid breaking Docker compatibility.

At the end:
- summarize changes
- summarize scores
- list remaining weaknesses
- suggest next experiments

---

# Validation Requirements

After each milestone:
- run tests
- run grade.py
- verify predict() contract
- verify deterministic inference
- check runtime sanity

---

# Milestones

## M1 — Baseline Understanding
Goals:
- understand scoring
- inspect features
- benchmark baseline
- identify weaknesses

Deliverables:
- baseline score
- repo understanding
- improvement roadmap

---

## M2 — Trajectory Improvements
Focus:
- acceleration-aware forecasting
- velocity smoothing
- ego-motion compensation
- heading estimation
- hesitation modeling

Primary metric:
- ADE reduction

---

## M3 — Intent Improvements
Focus:
- better motion features
- richer context features
- calibration improvements
- improved classifiers

Primary metric:
- BCE reduction

---

## M4 — Optional Sequence Model
ONLY pursue if:
- previous milestones plateau
- implementation remains lightweight

Possible approaches:
- GRU
- LSTM
- tiny transformer

---

## M5 — Finalization
Focus:
- Docker validation
- README
- cleanup
- reproducibility
- dependency minimization