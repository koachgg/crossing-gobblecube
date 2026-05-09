# Agent Engineering Guidelines

## General Expectations

Code should be:
- readable
- modular
- reproducible
- minimally complex

Avoid unnecessary rewrites.

Prefer improving existing pipelines before replacing them.

---

# Git Practices

Make focused commits.

Good examples:
- feat: add acceleration-aware trajectory prediction
- exp: evaluate smoothed velocity estimation
- fix: handle missing ego motion gracefully
- docs: summarize dev-set findings

Avoid giant unrelated commits.

---

# ML Engineering Philosophy

Prioritize:
- practical gains
- measurable improvements
- lightweight inference
- robustness

Avoid:
- giant architectures without justification
- premature optimization
- unnecessary dependencies

---

# Validation Expectations

Before concluding work:
- run tests
- run grade.py
- verify no API regressions
- verify Docker compatibility

Always summarize:
- what changed
- expected impact
- observed score impact
- remaining weaknesses

---

# Code Style

Prefer:
- clear helper functions
- comments explaining reasoning
- explicit feature calculations
- deterministic behavior

Avoid:
- magic constants without explanation
- deeply nested logic
- dead code

---

# Important Constraint

Inference must:
- run offline
- remain CPU-friendly
- stay within provided resource limits