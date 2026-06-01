import FlowSinkhorn.KLProjection.DualConvergence

set_option linter.style.longLine false

/-!
# Paper Section 3: Dual Convergence and Approximation

Paper-facing facade for Section 3 results and Appendix A/B lemmas.
Canonical proof homes remain under `FlowSinkhorn.KLProjection.DualConvergence.*`.
-/

namespace FlowSinkhorn
namespace Paper
namespace S3DualConvergence

open KLProjection
open KLProjection.DualConvergence

abbrev thm_3_1 := @dualRate_KL_paperConstant_from_ascentGapResidual -- impl: lean/FlowSinkhorn/KLProjection/DualConvergence/Rate.lean
abbrev thm_3_2 :=
  @regularizedApproximation_paperEpsilon_of_certificate_closedFormThreshold -- impl: lean/FlowSinkhorn/KLProjection/DualConvergence/Rate.lean

abbrev thm_kl_dual_rate := @dualRate_KL_paperConstant_from_ascentGapResidual -- impl: lean/FlowSinkhorn/KLProjection/DualConvergence/Rate.lean
abbrev thm_approx_linprog :=
  @regularizedApproximation_paperEpsilon_of_certificate_closedFormThreshold -- impl: lean/FlowSinkhorn/KLProjection/DualConvergence/Rate.lean

abbrev lem_A_1 :=
  @perStepAscent_twoHalfSteps_paperConstants_of_gammaExactSupportBlockUpdateCertificates_commonMass -- impl: lean/FlowSinkhorn/KLProjection/DualConvergence/PerStepAscent.lean
abbrev lem_A_2 := @dualGap_le_twoUmax_of_pairingBound_quotientSup_lt_Umax -- impl: lean/FlowSinkhorn/KLProjection/DualConvergence/GapResidual.lean
abbrev prop_A_1 :=
  @normalizedPinsker_of_finiteProbabilityMeasure_klDiv_computed_mathlib_hoeffding -- impl: lean/FlowSinkhorn/KLProjection/DualConvergence/Pinsker.lean
abbrev lem_A_3 :=
  @pinsker_nonnormalized_of_massShell_finiteProbabilityMeasure_klDiv_computed_mathlib_hoeffding -- impl: lean/FlowSinkhorn/KLProjection/DualConvergence/Pinsker.lean
abbrev lem_B_1 :=
  @klBias_regularizedGap_from_minimizers_finiteSumKL_cardLogReference -- impl: lean/FlowSinkhorn/KLProjection/DualConvergence/Rate.lean

abbrev lem_per_step_ascent :=
  @perStepAscent_twoHalfSteps_paperConstants_of_gammaExactSupportBlockUpdateCertificates_commonMass -- impl: lean/FlowSinkhorn/KLProjection/DualConvergence/PerStepAscent.lean
abbrev lem_gap_vs_res_quotient := @dualGap_le_twoUmax_of_pairingBound_quotientSup_lt_Umax -- impl: lean/FlowSinkhorn/KLProjection/DualConvergence/GapResidual.lean
abbrev prop_pinsker_normalized :=
  @normalizedPinsker_of_finiteProbabilityMeasure_klDiv_computed_mathlib_hoeffding -- impl: lean/FlowSinkhorn/KLProjection/DualConvergence/Pinsker.lean
abbrev lem_pinsker_nonnormalized :=
  @pinsker_nonnormalized_of_massShell_finiteProbabilityMeasure_klDiv_computed_mathlib_hoeffding -- impl: lean/FlowSinkhorn/KLProjection/DualConvergence/Pinsker.lean
abbrev lem_kl_bias :=
  @klBias_regularizedGap_from_minimizers_finiteSumKL_cardLogReference -- impl: lean/FlowSinkhorn/KLProjection/DualConvergence/Rate.lean

end S3DualConvergence
end Paper
end FlowSinkhorn
