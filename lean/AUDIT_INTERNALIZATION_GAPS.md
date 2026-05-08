# Internalization Audit Ledger

Status date: 2026-05-08.

This ledger records the current paper-facing internalization audit for the Lean formalization.
The machine-readable audit is `scripts/audit_paper_certification.py`.

## Audit Semantics

- Tier 1 endpoint coverage: every compiled paper label has both label-key and numbered aliases in `FlowSinkhorn/KLProjection/StatementMap.lean`, and both aliases resolve to the same Lean declaration.
- Tier 2 endpoint shape: the target declaration exists and is not an obvious thin pass-through endpoint such as direct forwarding or first-step `exact` forwarding.
- Tier 3 internalization candidate: the structural audit found no mapped `_of_assumption` endpoint, no thin paper-facing forwarder, and no unreviewed long-assumption endpoint.
- Reviewed long-interface endpoint: the paper-facing theorem has a long signature because it exposes primitive mathematical certificates, such as positivity, finite support, topicality, path witnesses, block monotonicity, or master-rate hypotheses. These are not treated as gaps when the endpoint has been reviewed and is the intended theorem interface.

## Current Summary

From `python3 lean/scripts/audit_paper_certification.py`:

- Tier 1 endpoint coverage: `27/27` labels covered.
- Tier 2 endpoint shape: `27/27` labels pass structural shape checks.
- Tier 3 internalization candidates: `27/27` labels have no structural gap flags.
- Internalization gap/review flags: `0` labels.
- `_of_assumption` endpoint/alias flags in the paper map: `0` labels.
- Unreviewed assumption-heavy endpoint flags, with threshold `>= 4`: `0` labels.
- Reviewed long-interface endpoints: `11` labels.

## Explicit `_of_assumption` Paper-Map Gaps

None currently mapped from `StatementMap.lean`.

The audit will flag any future paper alias whose target or alias ends in `_of_assumption` or `_of_assumptions` as an explicit internalization gap.

## Reviewed Long-Interface Paper Labels

The following labels have long theorem signatures, but the exposed hypotheses are now reviewed as the intended primitive mathematical interface rather than hidden proof conclusions.

| Label | Number | Lean target | Status |
|---|---:|---|---|
| `thm:kl-dual-rate` | `3.1` | `dualRate_masterAbstractRateStatement` | abstract master-rate interface |
| `thm:approx-linprog` | `3.2` | `regularizedApproximation_complexity_of_closedFormIterationThreshold` | abstract approximation/threshold interface |
| `prop:graphw1-flow-sinkhorn-update` | `5.2` | `graphW1_flowSinkhorn_update_as_stated_of_forward_nonneg` | local update theorem; nonnegativity is derived from row certificates |
| `thm:graphw1-complexity` | `5.1` | `graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index` | deep graph-W1 endpoint from topicality, path witness, HGamma budget, and rate hypotheses |
| `app-lem:per-step-ascent` | `A.1` | `perStepAscent_residualProxy_of_finiteMassShellExactSupportBlockUpdateCertificates_commonMass` | exact support block-update certificate route, with computed Pinsker machinery upstream |
| `app-lem:gap-vs-res-quotient` | `A.2` | `dualGap_le_twoUmax_of_pairingBound_quotientSup_lt_Umax` | quotient-sup pairing interface |
| `app-prop:pinsker-normalized` | `A.1` | `normalizedPinsker_of_finiteProbabilityMeasure_klDiv_computed_mathlib_hoeffding` | normalized Pinsker route using computed finite probability KL and Mathlib Hoeffding bridge |
| `app-lem:pinsker-nonnormalized` | `A.3` | `pinsker_nonnormalized_of_massShell_finiteProbabilityMeasure_klDiv_computed_mathlib_hoeffding` | non-normalized Pinsker obtained from the normalized computed theorem through mass-shell reduction |
| `app-cor:ot-xgamma-ugamma` | `E.1` | `ot_orbit_bound_from_separable_and_blockConditions_zeroStart` | OT orbit bound from block conditions plus separable decomposition |
| `app-prop:kappa-graph-diameter` | `F.5` | `graphW1_kappa_le_graphDiameter` | graph-W1 kappa bound from two-step path witnesses |
| `app-cor:graphw1-xgamma-ugamma` | `F.5` | `graphW1_orbit_bound_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma` | graph-W1 orbit bound from topicality, zero seed, two-step path data, and HGamma budget |

## Practical Review Protocol

1. Run `python3 scripts/check_statementmap_sync.py` after changing `StatementMap.lean`.
2. Run `python3 scripts/audit_paper_certification.py` after changing any paper-facing endpoint.
3. Treat any mapped `_of_assumption` endpoint as an immediate gap.
4. Treat any new `assumption-heavy:n` flag as a review item: either internalize the missing derivation, or add the target to `REVIEWED_INTERFACE_TARGETS` only after confirming that the long signature exposes primitive theorem hypotheses rather than a hidden paper conclusion.
5. Prefer mapping paper aliases to the deepest endpoint that still matches the paper statement semantics.
