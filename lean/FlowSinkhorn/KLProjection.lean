import FlowSinkhorn.KLProjection.Certification

/-!
# KLProjection Backend

Implementation backend for the paper formalization.

Use `FlowSinkhorn.Paper` for paper-first navigation.
Use this module when working on internal proof organization and core implementation layers.

Canonical trust chain (constants proved in backend modules):
- `DualConvergence.Rate.dualRate_masterAbstractRateStatement`;
- `PrimalDualBounds.FixedPointControl.uniformIterateBound_of_nonexpansive_of_HGamma_kappa`;
- `DualConvergence.Rate.regularizedApproximation_complexity_of_closedFormIterationThreshold`;
- `Applications.OT.Complexity.ot_explicit_XGamma_UGamma`;
- `Applications.GraphW1.Complexity.graphW1_explicit_XGamma_UGamma`.
-/
