# Lean Formalization: Purpose and Navigation

This `lean/` project formalizes the main results of the submitted
KL-projection paper. The supplementary archive contains the Lean sources and paper-facing theorem map,
but it does not need to include the LaTeX paper source.

This README is the primary entrypoint for contributors.

## 0) Naming and directory model (important)

Why both `KLProjection/` and `Paper/`?
- `KLProjection/` = implementation backend (where proofs are actually developed).
- `Paper/` = paper-structured facade (how to navigate by section/appendix).

Recommended entrypoint:
- use `FlowSinkhorn.Paper` for reading/auditing aligned with manuscript structure;
- use `FlowSinkhorn.KLProjection` only for backend proof engineering.

## 1) What this formalization guarantees

- Every paper theorem/proposition/lemma/corollary has a mapped Lean endpoint.
- The mapped endpoints compile in the global build.
- No placeholders in `FlowSinkhorn/KLProjection`:
  - no `sorry`
  - no `admit`
  - no local `axiom`
- Important quality note:
  - endpoint coverage and build success do not automatically mean every intermediate derivation is fully internalized from first principles at that endpoint; see Section 3 and `AUDIT_INTERNALIZATION_GAPS.md`.

## 2) Current certification status (2026-05-06)

From `lean/`:

```bash
lake build FlowSinkhorn.KLProjection.StatementMap
python3 scripts/check_statementmap_sync.py
python3 scripts/audit_paper_certification.py
rg '^\s*theorem\b' FlowSinkhorn/KLProjection | wc -l
rg '^\s*(def|structure)\b' FlowSinkhorn/KLProjection | wc -l
rg '^\s*(sorry|admit|axiom)\b' FlowSinkhorn/KLProjection | wc -l
```

Snapshot:
- actual non-comment, non-blank Lean code lines: `26596` (with block comments stripped)
- theorem/lemma declarations: `1511`
- direct `def`/`structure` declarations: `36` with the simple README counter above
- placeholders: `0`

## 3) Certification status tiers

Use this distinction when auditing:
- Tier 1: mapped + build-checked endpoint (all paper labels currently satisfy this).
- Tier 2: endpoint theorem is structurally nontrivial (passes `scripts/audit_paper_certification.py` shape checks).
- Tier 3: fully internalized paper-style derivation (no key model-specific inequality is assumed as an external hypothesis at the paper-facing endpoint).

## 4) Where paper-to-Lean linking happens

The synchronization layer is:
- `FlowSinkhorn/KLProjection/StatementMap.lean`
- facade import: `FlowSinkhorn.Paper.StatementMap`

Important:
- `StatementMap.lean` is an alias map only.
- The actual proofs are in thematic modules (`DualConvergence`, `PrimalDualBounds`, `Setup`, `Applications`, `Legacy`).

To inspect mapping consistency against the compiled paper labels/numbering:

```bash
python3 scripts/check_statementmap_sync.py
```

To run the stricter endpoint-structure audit:

```bash
python3 scripts/audit_paper_certification.py
```

Internalization-gap audit (assumption-packaging vs fully derived paper-style proof):

```bash
cat AUDIT_INTERNALIZATION_GAPS.md
```

## 5) How to find proof code from a paper label

Workflow:
1. Find alias in `StatementMap.lean` (`prop_*`, `thm_*`, `lem_*`, `cor_*`).
2. Read theorem constant name on the right-hand side.
3. Use search to jump to proof:

```bash
rg -n "theorem <name>|lemma <name>" FlowSinkhorn/KLProjection
```

## 6) Current directory organization (and why)

Current namespace root is:
- `FlowSinkhorn.KLProjection`

Historical context:
- `FlowSinkhorn` is the repository-level Lean package root.
- `KLProjection` scopes the paper-specific formalization inside that package.

Thematic subfolders:
- `Setup/`: monotonicity, translation, topical/nonexpansive geometry primitives.
- `DualConvergence/`: abstract rate machinery and approximation transfer.
- `PrimalDualBounds/`: fixed-point budget and primal-from-dual transfer.
- `Applications/OT/`: balanced OT instantiation.
- `Applications/GraphW1/`: graph-W1 instantiation.
- `Legacy/`: historical compatibility layer used by the paper map (not preferred for new work).

## 7) Paper-structured facade (implemented)

A non-breaking paper-first namespace is now available:
- umbrella: `FlowSinkhorn.Paper`
- main body: `FlowSinkhorn.Paper.MainBody`
- appendices: `FlowSinkhorn.Paper.Appendix`
- statement map facade: `FlowSinkhorn.Paper.StatementMap`

Section/appendix modules (preferred names):
- `FlowSinkhorn.Paper.Section2`
- `FlowSinkhorn.Paper.Section3`
- `FlowSinkhorn.Paper.Section4`
- `FlowSinkhorn.Paper.Section5`
- `FlowSinkhorn.Paper.AppendixA`
- `FlowSinkhorn.Paper.AppendixB`
- `FlowSinkhorn.Paper.AppendixE`
- `FlowSinkhorn.Paper.AppendixF`
- `FlowSinkhorn.Paper.AppendixG`

Legacy-compatible facade names kept:
- `FlowSinkhorn.Paper.S2Duality`
- `FlowSinkhorn.Paper.S3DualConvergence`
- `FlowSinkhorn.Paper.S4PrimalDualBounds`
- `FlowSinkhorn.Paper.S5GraphW1Main`
- `FlowSinkhorn.Paper.AppendixEOT`
- `FlowSinkhorn.Paper.AppendixFGraphW1`
- `FlowSinkhorn.Paper.AppendixGSetup`

Design:
- canonical proofs remain in `FlowSinkhorn.KLProjection.*`;
- paper-oriented navigation and aliases are provided by `FlowSinkhorn.Paper.*`.

## 8) Minimal build matrix

```bash
lake build FlowSinkhorn.KLProjection.Duality
lake build FlowSinkhorn.KLProjection.Convergence
lake build FlowSinkhorn.KLProjection.Geometry
lake build FlowSinkhorn.KLProjection.Applications
lake build FlowSinkhorn.KLProjection.StatementMap
lake build FlowSinkhorn.KLProjection.Certification
lake build
```
