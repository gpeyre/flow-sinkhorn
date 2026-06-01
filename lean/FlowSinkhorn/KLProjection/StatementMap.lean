import FlowSinkhorn.KLProjection.Duality
import FlowSinkhorn.KLProjection.Convergence
import FlowSinkhorn.KLProjection.Geometry
import FlowSinkhorn.KLProjection.Applications.OT
import FlowSinkhorn.KLProjection.Applications.GraphW1
import FlowSinkhorn.KLProjection.Applications.GraphW1.ClosedForms
import FlowSinkhorn.KLProjection.Applications.GraphW1.HGamma
import FlowSinkhorn.KLProjection.Applications.GraphW1.Kappa
import FlowSinkhorn.KLProjection.Applications.GraphW1.Complexity
import FlowSinkhorn.KLProjection.Applications.OT.HGamma
import FlowSinkhorn.KLProjection.Applications.OT.Kappa
import FlowSinkhorn.KLProjection.Applications.OT.Complexity

set_option linter.style.longLine false

/-!
# Statement map

Stable aliases from paper statement identifiers to Lean theorem constants.
The codebase organization remains thematic; this file is the only synchronization layer.

These aliases certify the paper-facing synchronization surface: each paper label below resolves to
one build-checked Lean theorem/lemma constant, and each alias records the implementation file where
the proof-producing declaration lives.  Use `scripts/check_statementmap_sync.py` to verify label,
numbering, facade, and implementation-location synchronization.  Use
`scripts/audit_paper_certification.py` for the structural endpoint audit.

## How to use this file

- Left side: stable paper-facing alias.
- Right side: canonical theorem constant.
- Actual proof code is **not** in this file; jump to the theorem constant in its home module.
- Alias targets ending in `_of_assumption` are rejected by the paper-certification audit.
- Long paper-facing theorem interfaces are reviewed by `scripts/audit_paper_certification.py`.
  In the current audit, all 27 paper-facing endpoints pass the structural checks.

Main home modules:
- Section 2 duality certificate: `Legacy/Section2Duality.lean`
- Section 3 rates: `DualConvergence/Rate.lean`, `DualConvergence/PerStepAscent.lean`,
  `DualConvergence/GapResidual.lean`
- Section 4 bounds: `PrimalDualBounds/FixedPointControl.lean`,
  `PrimalDualBounds/PrimalFromDual.lean`
- Setup/nonexpansiveness: `Setup/VariationGeometry.lean`, `Setup/BlockMonotonicity.lean`,
  `Setup/Translation.lean`
- OT application: `Applications/OT/HGamma.lean`, `Applications/OT/Kappa.lean`,
  `Applications/OT/Complexity.lean`
- Graph-W1 application: `Applications/GraphW1/ClosedForms.lean`,
  `Applications/GraphW1/HGamma.lean`, `Applications/GraphW1/Kappa.lean`,
  `Applications/GraphW1/Complexity.lean`
-/

namespace FlowSinkhorn
namespace KLProjection
namespace StatementMap

open Section2Duality
open DualConvergence
open PrimalDualBounds
open Setup
open Applications
open Applications.OT
open Applications.GraphW1

/- Core paper map (numbered aliases) -/
/-
Numbered aliases mirror the compiled paper numbering
(`2.1`, `3.1`, `4.2`, `F.5`, ...).
These names are checked automatically by `scripts/check_statementmap_sync.py`.
-/
/- Section 2. -/
abbrev prop_2_1 := @dualGammaCorrect_from_primalDualCertificate -- impl: Legacy/Section2Duality.lean

/- Section 3. -/
abbrev thm_3_1 := @dualRate_KL_paperConstant_from_ascentGapResidual -- impl: DualConvergence/Rate.lean
abbrev lem_A_1 :=
  @perStepAscent_twoHalfSteps_paperConstants_of_gammaExactSupportBlockUpdateCertificates_commonMass -- impl: DualConvergence/PerStepAscent.lean
abbrev lem_A_2 := @dualGap_le_twoUmax_of_pairingBound_quotientSup_lt_Umax -- impl: DualConvergence/GapResidual.lean
abbrev prop_A_1 :=
  @normalizedPinsker_of_finiteProbabilityMeasure_klDiv_computed_mathlib_hoeffding -- impl: DualConvergence/Pinsker.lean
abbrev lem_A_3 :=
  @pinsker_nonnormalized_of_massShell_finiteProbabilityMeasure_klDiv_computed_mathlib_hoeffding -- impl: DualConvergence/Pinsker.lean
abbrev lem_B_1 :=
  @klBias_regularizedGap_from_minimizers_finiteSumKL_cardLogReference -- impl: DualConvergence/Rate.lean
abbrev thm_3_2 :=
  @regularizedApproximation_paperEpsilon_of_certificate_closedFormThreshold -- impl: DualConvergence/Rate.lean

/- Section 4. -/
abbrev prop_4_1 := @uniformIterateBound_of_nonexpansive_of_budget -- impl: PrimalDualBounds/FixedPointControl.lean
abbrev prop_4_2 :=
  @primalMassBound_from_zeroStartFinitePairing_exactL1_card_quotientRadiusCertificate -- impl: PrimalDualBounds/PrimalFromDual.lean

/- Appendix G (setup/nonexpansiveness). -/
abbrev prop_G_1 := @variationSeminorm_nonexpansive_of_topical -- impl: Topical.lean
abbrev prop_G_2 := @blockUpdate_antitoneRelation_then_sweep_monotone -- impl: Setup/BlockMonotonicity.lean
abbrev lem_G_1 :=
  @momentMap_monotone_of_nonnegative_linear_layers -- impl: Setup/BlockMonotonicity.lean
abbrev prop_G_3 := @translationEquivariance_of_pairedBalance_blockLaws -- impl: Setup/Translation.lean

/- Appendix E (OT instantiation). -/
abbrev prop_E_1 :=
  @ot_HGamma_formula_uniform_logRatio_bound_from_typedRightScaling -- impl: Applications/OT/HGamma.lean
abbrev prop_E_2 := @ot_kappa_coordSupNorm_le -- impl: Applications/OT/Kappa.lean
abbrev cor_E_1 :=
  @ot_XGamma_eq_one_and_UGamma_bound_from_structuredCertificates -- impl: Applications/OT/Complexity.lean

/- Section 5 main text. -/
abbrev prop_5_1 :=
  @graphW1_projection_closedForm_maps_with_variationalCertificate -- impl: Applications/GraphW1/ClosedForms.lean
abbrev prop_5_2 :=
  @graphW1_flowSinkhorn_stableDualUpdate_from_pointwiseBlockIdentities -- impl: Applications/GraphW1/ClosedForms.lean
abbrev thm_5_1 :=
  @graphW1_sinkhornFlow_complexity_from_operationBounds -- impl: Applications/GraphW1/Complexity.lean

/- Appendix F (graph-W1 technical statements). -/
abbrev prop_F_1 := @graphW1_blockQuotient_closedForm -- impl: Applications/GraphW1/ClosedForms.lean
abbrev prop_F_2 := @graphW1_signedStructure_fullSweep_variationSeminorm_nonexpansive -- impl: Applications/GraphW1/ClosedForms.lean
abbrev prop_F_3 := @graphW1_Psi2_closedForm_nonexpansive -- impl: Applications/GraphW1/ClosedForms.lean
abbrev prop_F_4 :=
  @graphW1_HGamma_formula_uniform_logRatio_bound_from_positiveFields_oppositeLog_logEnvelope -- impl: Applications/GraphW1/HGamma.lean
abbrev lem_F_1 :=
  @graphW1_primalL1Bound_from_nonnegativeFeasibleSet_minCost_coordinateSumKL_posGamma -- impl: Applications/GraphW1/HGamma.lean
abbrev prop_F_5 :=
  @graphW1_kappa_le_graphDiameter_from_rootedPathFamily -- impl: Applications/GraphW1/Kappa.lean
abbrev cor_F_1 :=
  @graphW1_XGamma_UGamma_bounds_from_structuredCertificates_twoStep_path -- impl: Applications/GraphW1/Complexity.lean

/- Label-key aliases used in the NeurIPS source. -/
/-
Label-key aliases mirror LaTeX keys (`prop:*`, `thm:*`, `lem:*`, `cor:*`).
Use these aliases when auditing paper/Lean synchronization by label.
-/
abbrev prop_dual_gamma_correct :=
  @dualGammaCorrect_from_primalDualCertificate -- impl: Legacy/Section2Duality.lean
abbrev thm_kl_dual_rate := @dualRate_KL_paperConstant_from_ascentGapResidual -- impl: DualConvergence/Rate.lean
abbrev lem_per_step_ascent :=
  @perStepAscent_twoHalfSteps_paperConstants_of_gammaExactSupportBlockUpdateCertificates_commonMass -- impl: DualConvergence/PerStepAscent.lean
abbrev lem_gap_vs_res_quotient := @dualGap_le_twoUmax_of_pairingBound_quotientSup_lt_Umax -- impl: DualConvergence/GapResidual.lean
abbrev prop_pinsker_normalized :=
  @normalizedPinsker_of_finiteProbabilityMeasure_klDiv_computed_mathlib_hoeffding -- impl: DualConvergence/Pinsker.lean
abbrev lem_pinsker_nonnormalized :=
  @pinsker_nonnormalized_of_massShell_finiteProbabilityMeasure_klDiv_computed_mathlib_hoeffding -- impl: DualConvergence/Pinsker.lean
abbrev lem_kl_bias :=
  @klBias_regularizedGap_from_minimizers_finiteSumKL_cardLogReference -- impl: DualConvergence/Rate.lean
abbrev thm_approx_linprog :=
  @regularizedApproximation_paperEpsilon_of_certificate_closedFormThreshold -- impl: DualConvergence/Rate.lean
abbrev prop_mass_bound_block :=
  @primalMassBound_from_zeroStartFinitePairing_exactL1_card_quotientRadiusCertificate -- impl: PrimalDualBounds/PrimalFromDual.lean
abbrev prop_uniform_iter_final := @uniformIterateBound_of_nonexpansive_of_budget -- impl: PrimalDualBounds/FixedPointControl.lean
abbrev prop_topical_nonexpansive := @variationSeminorm_nonexpansive_of_topical -- impl: Topical.lean
abbrev prop_block_monotone := @blockUpdate_antitoneRelation_then_sweep_monotone -- impl: Setup/BlockMonotonicity.lean
abbrev lem_moment_monotone :=
  @momentMap_monotone_of_nonnegative_linear_layers -- impl: Setup/BlockMonotonicity.lean
abbrev prop_translation_equivariance := @translationEquivariance_of_pairedBalance_blockLaws -- impl: Setup/Translation.lean
abbrev prop_hgamma_ot :=
  @ot_HGamma_formula_uniform_logRatio_bound_from_typedRightScaling -- impl: Applications/OT/HGamma.lean
abbrev prop_kappa_ot := @ot_kappa_coordSupNorm_le -- impl: Applications/OT/Kappa.lean
abbrev cor_ot_xgamma_ugamma :=
  @ot_XGamma_eq_one_and_UGamma_bound_from_structuredCertificates -- impl: Applications/OT/Complexity.lean
abbrev prop_graphw1_projection_closed_form :=
  @graphW1_projection_closedForm_maps_with_variationalCertificate -- impl: Applications/GraphW1/ClosedForms.lean
abbrev prop_graphw1_flow_sinkhorn_update :=
  @graphW1_flowSinkhorn_stableDualUpdate_from_pointwiseBlockIdentities -- impl: Applications/GraphW1/ClosedForms.lean
abbrev prop_graphw1_v1v2_closed_form := @graphW1_blockQuotient_closedForm -- impl: Applications/GraphW1/ClosedForms.lean
abbrev prop_graphw1_signed_structure :=
  @graphW1_signedStructure_fullSweep_variationSeminorm_nonexpansive -- impl: Applications/GraphW1/ClosedForms.lean
abbrev prop_graphw1_psi2_closed_nonexp := @graphW1_Psi2_closedForm_nonexpansive -- impl: Applications/GraphW1/ClosedForms.lean
abbrev prop_hgamma_graphw1 :=
  @graphW1_HGamma_formula_uniform_logRatio_bound_from_positiveFields_oppositeLog_logEnvelope -- impl: Applications/GraphW1/HGamma.lean
abbrev lem_l1_bound_from_feasible :=
  @graphW1_primalL1Bound_from_nonnegativeFeasibleSet_minCost_coordinateSumKL_posGamma -- impl: Applications/GraphW1/HGamma.lean
abbrev prop_kappa_graph_diameter :=
  @graphW1_kappa_le_graphDiameter_from_rootedPathFamily -- impl: Applications/GraphW1/Kappa.lean
abbrev cor_graphw1_xgamma_ugamma :=
  @graphW1_XGamma_UGamma_bounds_from_structuredCertificates_twoStep_path -- impl: Applications/GraphW1/Complexity.lean
abbrev thm_graphw1_complexity :=
  @graphW1_sinkhornFlow_complexity_from_operationBounds -- impl: Applications/GraphW1/Complexity.lean

end StatementMap
end KLProjection
end FlowSinkhorn
