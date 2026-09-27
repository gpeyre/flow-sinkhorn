# Lean Formalization Guide

This directory contains the Lean 4 formalization for the paper.  The goal is to make the
paper-facing theorem list easy to audit, while keeping the proof-producing implementation organized
by reusable mathematical components.

Use this file as the main entry point.  Detailed audit ledgers and generated Comparator artifacts
live under `audit/`.

## Current Status

The September 2026 paper revision is **not fully formalized**. See
[`PAPER_COVERAGE.md`](PAPER_COVERAGE.md) for the distinction between reusable
proved lemmas, conditional certificates, historical aliases, and missing bridges.
The previous `27/27` structural counts did not establish semantic equivalence to
the paper: several endpoints assumed the mathematical estimates they were meant
to certify. Those counts and the generated Comparator “faithful” classifications
must not be interpreted as a complete correctness certificate.

The historical `audit/` ledgers and Comparator artifacts are retained for
traceability, not as current certification claims. Challenge modules intentionally
contain statement-only `sorry` declarations and are not proof-producing modules.

## Fast Verification

Run these commands from `lean/`:

```bash
lake build FlowSinkhorn.KLProjection.StatementMap
lake build FlowSinkhorn.Comparator.Challenge FlowSinkhorn.Comparator.Solution FlowSinkhorn.Paper
python3 scripts/check_comparator_scaffold.py
python3 scripts/check_comparator_challenge_lock.py
python3 scripts/check_comparator_trust_boundary.py
```

The optional synchronization scripts read `paper/paper.aux` to recover compiled statement numbering.  If
that file is absent after a clean checkout, regenerate it from the repository root with:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
```

The optional label and structural audits currently report unresolved synchronization
and certification gaps; this is expected until the revised statements are formalized.
A successful build or challenge/solution shape match does not close those gaps.

Useful hygiene scans:

```bash
rg '^\s*(sorry|admit|axiom)\b' FlowSinkhorn/KLProjection
rg '_of_assumption' FlowSinkhorn/KLProjection/StatementMap.lean FlowSinkhorn/Paper FlowSinkhorn/Comparator/Solution.lean
```

The Comparator smoke test is available for local wiring only:

```bash
COMPARATOR_ALLOW_FAKE_LANDRUN=1 scripts/run_comparator_bootstrap.sh
```

This smoke test is not a final certificate.  A meaningful paper certificate additionally requires independent statement
alignment, all mathematical bridges, and a clean hardened comparison run; a Linux
run with real `landrun` alone would still not prove that alignment.

## Directory Map

Primary entry points:

- `FlowSinkhorn.lean`: package umbrella.
- `FlowSinkhorn/KLProjection.lean`: proof-producing backend umbrella.
- `FlowSinkhorn/Paper.lean`: manuscript-ordered facade.
- `FlowSinkhorn/KLProjection/StatementMap.lean`: canonical paper-to-Lean alias map.
- `FlowSinkhorn/Paper/StatementMap.lean`: paper-facade re-export of the alias map.
- `FlowSinkhorn/Comparator/Challenge.lean`: trusted Comparator challenge statements.
- `FlowSinkhorn/Comparator/Solution.lean`: untrusted Comparator solution theorem names.
- `FlowSinkhorn/Comparator/Vocabulary.lean`: navigation umbrella for proof-free statement vocabulary.

Implementation backend:

- `KLProjection/Legacy/`: Section 2 duality/primal-dual certificate layer.
- `KLProjection/DualConvergence/`: Pinsker, per-step ascent, gap-residual, rate, and approximation
  transfer machinery.
- `KLProjection/PrimalDualBounds/`: fixed-point control and primal-from-dual mass bounds.
- `KLProjection/Setup/`: variation geometry, monotonicity, and translation laws.
- `KLProjection/Applications/OT/`: optimal-transport specialization.
- `KLProjection/Applications/GraphW1/`: graph-W1 closed forms, H_gamma, kappa, and complexity.

Paper facade:

- `Paper/Section2.lean` through `Paper/Section5.lean` mirror main-paper sections.
- `Paper/AppendixA.lean`, `Paper/AppendixB.lean`, `Paper/AppendixE.lean`,
  `Paper/AppendixF.lean`, and `Paper/AppendixG.lean` mirror appendix groups.
- Legacy facade file names such as `S3DualConvergence.lean` are retained for compatibility.

Comparator layer:

- `Comparator/Vocabulary/` contains proof-free definitions needed to state the challenge.
- `Comparator/Challenge.lean` may import only `Mathlib` and explicit `Comparator.Vocabulary.*`
  modules.
- `Comparator/Solution.lean` may import only `FlowSinkhorn.KLProjection.StatementMap`.
- `KLProjection/*Vocabulary.lean` files are compatibility shims into `Comparator/Vocabulary/` and
  should remain import-only.

## Finding a Proof From a Paper Statement

Use this workflow:

1. Start from the paper label, for example `prop:graphw1-flow-sinkhorn-update`.
2. Open `FlowSinkhorn/KLProjection/StatementMap.lean`.
3. Find the label alias, for example `prop_graphw1_flow_sinkhorn_update`.
4. Read the implementation comment on that alias.
5. Jump to the theorem constant in the indicated file.

Example:

```lean
abbrev prop_graphw1_flow_sinkhorn_update :=
  @graphW1_flowSinkhorn_stableDualUpdate_from_pointwiseBlockIdentities
  -- impl: Applications/GraphW1/ClosedForms.lean
```

The alias file is proof-free by design.  The actual proof is in the implementation module named in
the comment.

## Statement Map and Audit Rules

`StatementMap.lean` is the single canonical synchronization layer.  It must remain stable and easy
to read.

Rules:

- Every paper theorem/proposition/lemma/corollary must have a label alias and a numbered alias.
- Each alias must point to one canonical Lean theorem constant.
- Each alias should carry an implementation-file comment.
- Paper facade aliases in `FlowSinkhorn/Paper/*.lean` must stay synchronized with `StatementMap`.
- If a paper statement changes, first independently validate the hypotheses and mathematical
  proof obligations. Only then synchronize the manifest, Challenge, Solution, review, and lock.
  Regeneration from the implementation is not an independent paper-faithfulness check.

Synchronization command:

```bash
python3 scripts/check_statementmap_sync.py
```

Structural audit command:

```bash
python3 scripts/audit_paper_certification.py
```

## Comparator Certification Artifacts

The Comparator artifacts are generated and locked so that future edits cannot silently change the
trusted statement surface.

Important files:

- `audit/comparator.md`: detailed Comparator deployment and status log.
- `audit/comparator-paper-manifest.json`: ordered paper statement list and Lean target metadata.
- `audit/comparator-paper-config.template.json`: Comparator theorem list and permitted axioms.
- `audit/comparator-paper-review.json` and `audit/comparator-paper-review.md`:
  statement-by-statement review dossier.
- `audit/comparator-challenge-audit.json` and `audit/comparator-challenge-audit.md`: independent
  semantic audit of Challenge against the LaTeX statements.
- `audit/comparator-challenge-lock.json`: frozen Challenge hash, import list, theorem order,
  comment hashes, statement hashes, and audit status date.

Regenerate only after a real paper-to-Challenge review:

```bash
python3 scripts/generate_comparator_manifest.py
python3 scripts/generate_comparator_challenge.py
python3 scripts/generate_comparator_solution.py
python3 scripts/generate_comparator_review.py
python3 scripts/generate_comparator_challenge_lock.py
```

Then check:

```bash
python3 scripts/check_comparator_challenge_lock.py
python3 scripts/check_comparator_scaffold.py
python3 scripts/check_comparator_trust_boundary.py
```

The lock generator and scaffold checker are fail-closed: they reject stale labels, stale numbering,
stale implementation locations, non-faithful audit verdicts, import-boundary drift, and
Challenge/Solution statement mismatches.

## Maintenance Rules

Use these rules when extending or refactoring the formalization:

- Do not introduce `sorry`, `admit`, or local `axiom` in proof-producing modules.
- Keep `StatementMap.lean` proof-free.
- Keep Challenge proof-free and statement-only.
- Keep Comparator vocabulary proof-free and implementation-free.
- Prefer adding reusable backend lemmas in thematic `KLProjection/*` modules rather than putting
  proof logic in paper facade files.
- Preserve public paper-facing alias names unless the LaTeX label itself changes.
- Run the verification commands above after any paper-facing theorem or alias change.

## Removed Redundant Docs

Older standalone planning notes were folded into this README to keep navigation simple.  The source
of truth is now:

- `README.md` for orientation and maintenance rules.
- `audit/comparator.md` for detailed Comparator deployment and status.
- `audit/comparator-challenge-audit.md` for statement-by-statement Challenge faithfulness.
- `audit/comparator-paper-review.md` for the paper/Challenge/Solution review dossier.
