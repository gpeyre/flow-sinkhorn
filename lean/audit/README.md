# Lean Audit Repository

This directory contains detailed audit material for the Lean formalization. It is intentionally
separate from the top-level `lean/README.md`, which is the short user-facing entry point.

Use this directory when reviewing certification status, Comparator deployment, or historical
internalization decisions.

## Contents

- `comparator.md`: detailed Comparator plan, environment notes, and current status.
- `comparator-paper-manifest.json`: ordered paper statement list and canonical Lean targets.
- `comparator-paper-config.template.json`: Comparator theorem list and permitted axioms.
- `comparator-paper-review.json` / `comparator-paper-review.md`: paper, Challenge, Solution, and
  implementation-source review dossier.
- `comparator-challenge-audit.json` / `comparator-challenge-audit.md`: independent semantic audit
  of Challenge statements against the LaTeX paper.
- `comparator-challenge-lock.json`: frozen hash lock for the reviewed Challenge file.
- `internalization-gaps.md`: historical internalization ledger; current scripts require it to have
  no active open-gap markers for paper-facing endpoints.

## Regeneration

Run these from `lean/` after an intentional paper-facing statement change:

```bash
python3 scripts/generate_comparator_manifest.py
python3 scripts/generate_comparator_challenge.py
python3 scripts/generate_comparator_solution.py
python3 scripts/generate_comparator_review.py
python3 scripts/generate_comparator_challenge_lock.py
```

Then verify:

```bash
python3 scripts/check_comparator_challenge_lock.py
python3 scripts/check_comparator_scaffold.py
python3 scripts/check_comparator_trust_boundary.py
```
