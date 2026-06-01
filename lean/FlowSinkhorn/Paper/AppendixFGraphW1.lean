import FlowSinkhorn.KLProjection.Applications.GraphW1

set_option linter.style.longLine false

/-!
# Paper Appendix F: Graph-W1 Technical Statements

Paper-facing facade for Appendix F results.
Canonical proof homes remain under `FlowSinkhorn.KLProjection.Applications.GraphW1.*`.
-/

namespace FlowSinkhorn
namespace Paper
namespace AppendixFGraphW1

open KLProjection
open KLProjection.Applications
open KLProjection.Applications.GraphW1

abbrev prop_F_1 := @graphW1_blockQuotient_closedForm -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/ClosedForms.lean
abbrev prop_F_2 := @graphW1_signedStructure_fullSweep_variationSeminorm_nonexpansive -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/ClosedForms.lean
abbrev prop_F_3 := @graphW1_Psi2_closedForm_nonexpansive -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/ClosedForms.lean
abbrev prop_F_4 :=
  @graphW1_HGamma_formula_uniform_logRatio_bound_from_positiveFields_oppositeLog_logEnvelope -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/HGamma.lean
abbrev lem_F_1 :=
  @graphW1_primalL1Bound_from_nonnegativeFeasibleSet_minCost_coordinateSumKL_posGamma -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/HGamma.lean
abbrev prop_F_5 :=
  @graphW1_kappa_le_graphDiameter_from_rootedPathFamily -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/Kappa.lean
abbrev cor_F_1 :=
  @graphW1_XGamma_UGamma_bounds_from_structuredCertificates_twoStep_path -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/Complexity.lean

abbrev prop_graphw1_v1v2_closed_form := @graphW1_blockQuotient_closedForm -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/ClosedForms.lean
abbrev prop_graphw1_signed_structure :=
  @graphW1_signedStructure_fullSweep_variationSeminorm_nonexpansive -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/ClosedForms.lean
abbrev prop_graphw1_psi2_closed_nonexp := @graphW1_Psi2_closedForm_nonexpansive -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/ClosedForms.lean
abbrev prop_hgamma_graphw1 :=
  @graphW1_HGamma_formula_uniform_logRatio_bound_from_positiveFields_oppositeLog_logEnvelope -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/HGamma.lean
abbrev lem_l1_bound_from_feasible :=
  @graphW1_primalL1Bound_from_nonnegativeFeasibleSet_minCost_coordinateSumKL_posGamma -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/HGamma.lean
abbrev prop_kappa_graph_diameter :=
  @graphW1_kappa_le_graphDiameter_from_rootedPathFamily -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/Kappa.lean
abbrev cor_graphw1_xgamma_ugamma :=
  @graphW1_XGamma_UGamma_bounds_from_structuredCertificates_twoStep_path -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/Complexity.lean

end AppendixFGraphW1
end Paper
end FlowSinkhorn
