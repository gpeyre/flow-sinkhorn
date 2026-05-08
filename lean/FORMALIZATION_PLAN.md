# Formalization Plan (Forward-Looking)

This file tracks *next* structural improvements. Current status is in `README.md`.

## Goals

1. Keep all paper-mapped results independently certified.
2. Improve navigability for contributors who follow paper section order.
3. Reduce naming/structure ambiguity (`FlowSinkhorn` vs whole-paper scope).

## Near-term plan

Completed:
1. Added paper-section facades (non-breaking) under `FlowSinkhorn.Paper.*`.
2. Re-exported canonical theorem endpoints from these facades.
3. Kept `StatementMap.lean` aliases targeting canonical constants.

Next:
1. Extend facade coverage if new paper statements are added.
2. Keep `SectionN/AppendixX` and legacy facade names synchronized.

## Mid-term plan

1. Decide whether to keep thematic implementation layout or physically move files to paper-section folders.
2. If moving files, do it in one migration wave with compatibility re-exports.

## Quality gates for every wave

1. `lake build FlowSinkhorn.KLProjection.StatementMap`
2. `python3 scripts/check_statementmap_sync.py`
3. `lake build`
4. Placeholder scan remains empty.

## Non-goals

- No broad renaming of proven canonical theorem constants unless absolutely necessary.
- No breakage of existing imports used by paper and CI.
