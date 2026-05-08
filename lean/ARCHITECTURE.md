# Lean Architecture (Concise)

This file records structural invariants. Operational status lives in `README.md`.

## Scope

Formalization root:
- `FlowSinkhorn/KLProjection`
- paper facade root: `FlowSinkhorn/Paper`

## Layering rules

1. Foundation/setup layer
- `Variation.lean`
- `BlockQuotient.lean`
- `Sweep.lean`
- `UniformBound.lean`
- `Topical.lean`
- `Setup/*`

2. Generic theorem layer
- `DualConvergence/*`
- `PrimalDualBounds/*`

3. Application layer
- `Applications/OT/*`
- `Applications/GraphW1/*`

Rule:
- Applications may depend on generic/foundation.
- Generic/foundation must not depend on applications.

## Synchronization layer

- `StatementMap.lean` is the single paper-label alias layer.
- It must stay proof-free and stable in names.
- Paper-facing re-export is `FlowSinkhorn.Paper.StatementMap`.

## Entrypoints

- `FlowSinkhorn.KLProjection` (full package)
- `FlowSinkhorn.KLProjection.Certification` (paper certification chain)
- `FlowSinkhorn.KLProjection.StatementMap` (paper label aliases)
- `FlowSinkhorn.Paper` (paper-structured facade)

## Contributor constraints

1. No `sorry`, `admit`, or new `axiom`.
2. Keep public theorem names stable.
3. Preserve layer boundaries.
4. When a paper-facing theorem changes, re-run:
   - `lake build FlowSinkhorn.KLProjection.StatementMap`
   - `python3 scripts/check_statementmap_sync.py`
