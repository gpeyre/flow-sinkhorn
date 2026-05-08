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
abbrev prop_F_2 := @graphW1_flowSinkhorn_closedForm_variationSeminorm_nonexpansive_of_isTopical -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/ClosedForms.lean
abbrev prop_F_3 := @graphW1_Psi2_nonexpansive -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/ClosedForms.lean
abbrev prop_F_4 := @graphW1_HGamma_bound -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/HGamma.lean
abbrev lem_F_1 := @graphW1_primalL1Bound_positiveCosts -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/HGamma.lean
abbrev prop_F_5 := @graphW1_kappa_le_graphDiameter -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/Kappa.lean
abbrev cor_F_5 :=
  @graphW1_orbit_bound_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/Complexity.lean

abbrev prop_graphw1_v1v2_closed_form := @graphW1_blockQuotient_closedForm -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/ClosedForms.lean
abbrev prop_graphw1_signed_structure :=
  @graphW1_flowSinkhorn_closedForm_variationSeminorm_nonexpansive_of_isTopical -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/ClosedForms.lean
abbrev prop_graphw1_psi2_closed_nonexp := @graphW1_Psi2_nonexpansive -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/ClosedForms.lean
abbrev prop_hgamma_graphw1 := @graphW1_HGamma_bound -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/HGamma.lean
abbrev lem_l1_bound_from_feasible := @graphW1_primalL1Bound_positiveCosts -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/HGamma.lean
abbrev prop_kappa_graph_diameter := @graphW1_kappa_le_graphDiameter -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/Kappa.lean
abbrev cor_graphw1_xgamma_ugamma :=
  @graphW1_orbit_bound_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/Complexity.lean

end AppendixFGraphW1
end Paper
end FlowSinkhorn
