# Internalization Gap Ledger

Status date: 2026-05-08.

This ledger distinguishes paper-label endpoint coverage from full paper-style internalization.
The corresponding machine-readable structural audit is `scripts/audit_paper_certification.py`.

## Audit Semantics

- Tier 1 endpoint coverage: the compiled paper label has both label-key and numbered aliases in `FlowSinkhorn/KLProjection/StatementMap.lean`, and both aliases resolve to the same Lean declaration.
- Tier 2 endpoint shape: the target declaration exists and is not an obvious thin pass-through endpoint such as direct forwarding or first-step `exact` forwarding.
- Tier 3 internalization candidate: the structural audit found no obvious internalization-risk flag. This is not a semantic guarantee that every manuscript derivation has been internalized from first principles.
- Internalization gap/review flag: the endpoint is build-checked, but the paper-facing statement still appears to expose substantial model-specific facts as hypotheses or uses a naming convention that indicates assumption packaging.

## Current Summary

From `python3 lean/scripts/audit_paper_certification.py`:

- Tier 1 endpoint coverage: `27/27` labels covered.
- Tier 2 endpoint shape: `27/27` labels pass structural shape checks.
- Tier 3 internalization candidates: `16/27` labels have no structural gap flags.
- Internalization gap/review flags: `11` labels.
- `_of_assumption` endpoint/alias flags in the paper map: `0` labels.
- Assumption-heavy endpoint flags, with threshold `>= 4`: `11` labels.

## Explicit `_of_assumption` Paper-Map Gaps

None currently mapped from `StatementMap.lean`.

The audit will flag any future paper alias whose target or alias ends in `_of_assumption` or `_of_assumptions` as an explicit internalization gap.

## Current Assumption-Heavy Paper Labels

These labels have build-checked endpoints and pass the endpoint-shape audit, but their paper-facing theorem signatures expose at least four assumption-like binder groups. They should be treated as Tier 3 review items until the corresponding assumptions are either discharged by upstream Lean theorems or explicitly justified as the intended abstract theorem interface.

| Label | Number | Lean target | Flag |
|---|---:|---|---|
| `thm:kl-dual-rate` | `3.1` | `dualRate_masterAbstractRateStatement` | `assumption-heavy:5` |
| `thm:approx-linprog` | `3.2` | `regularizedApproximation_complexity_of_closedFormIterationThreshold` | `assumption-heavy:5` |
| `prop:graphw1-flow-sinkhorn-update` | `5.2` | `graphW1_flowSinkhorn_update_as_stated_of_forward_nonneg` | `assumption-heavy:6` |
| `thm:graphw1-complexity` | `5.1` | `graphW1_Sinkhorn_iterationComplexity` | `assumption-heavy:5` |
| `app-lem:per-step-ascent` | `A.1` | `perStepAscent_residualProxy_of_massShellVariationalMathlibPinsker_klGains_commonMass` | `assumption-heavy:10` |
| `app-lem:gap-vs-res-quotient` | `A.2` | `dualGap_le_twoUmax_of_pairingBound_quotientSup_lt_Umax` | `assumption-heavy:4` |
| `app-prop:pinsker-normalized` | `A.1` | `normalizedPinsker_of_finite_variational_mathlib_hoeffding` | `assumption-heavy:4` |
| `app-lem:pinsker-nonnormalized` | `A.3` | `pinsker_nonnormalized_of_massShell_variational_mathlib_hoeffding` | `assumption-heavy:5` |
| `app-cor:ot-xgamma-ugamma` | `E.1` | `ot_explicit_XGamma_UGamma` | `assumption-heavy:7` |
| `app-prop:kappa-graph-diameter` | `F.5` | `graphW1_kappa_le_graphDiameter` | `assumption-heavy:7` |
| `app-cor:graphw1-xgamma-ugamma` | `F.5` | `graphW1_explicit_XGamma_UGamma` | `assumption-heavy:5` |

## Interpretation By Gap Type

`assumption-heavy` means the endpoint is a theorem, not an axiom or placeholder, but its statement still requires several hypotheses such as per-step inequalities, budget bounds, nonexpansiveness assumptions, monotonicity assumptions, positivity assumptions, or application-specific estimate hypotheses.

The previous compatibility endpoint for `prop:graphw1-flow-sinkhorn-update` has been superseded in
the statement map by `graphW1_flowSinkhorn_update_as_stated_of_forward_nonneg`, which discharges
the explicit `q ≥ 0` side condition from forward-row nonnegativity and the positive backward row
sum.  It is still marked assumption-heavy because the theorem intentionally remains a paper-shaped
local update statement rather than a full graph-model derivation from all primitive data.

## Practical Review Order

1. Review `_of_assumption` targets first if any become mapped in `StatementMap.lean`; these are explicit assumption-packaging gaps.
2. Review assumption-heavy application corollaries, especially `app-cor:ot-xgamma-ugamma`,
   `app-prop:kappa-graph-diameter`, and `app-cor:graphw1-xgamma-ugamma`, to decide which
   assumptions are intended abstract inputs and which should be discharged internally.
3. Review abstract rate and appendix inequalities to decide whether their many hypotheses are the
   intended theorem interface or remaining paper-derivation obligations.
