import FlowSinkhorn.KLProjection.Applications.GraphW1

set_option linter.style.longLine false

/-!
# Paper Section 5: Graph-W1 Main Statements

Paper-facing facade for Section 5 main-text statements.
Canonical proof homes remain under `FlowSinkhorn.KLProjection.Applications.GraphW1.*`.
-/

namespace FlowSinkhorn
namespace Paper
namespace S5GraphW1Main

open KLProjection
open KLProjection.Applications
open KLProjection.Applications.GraphW1

abbrev prop_5_1 := @graphW1_projection_closedForm -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/ClosedForms.lean
abbrev prop_5_2 := @graphW1_flowSinkhorn_update_as_stated_of_forward_nonneg -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/ClosedForms.lean
abbrev thm_5_1 :=
  @graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/Complexity.lean

abbrev prop_graphw1_projection_closed_form := @graphW1_projection_closedForm -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/ClosedForms.lean
abbrev prop_graphw1_flow_sinkhorn_update :=
  @graphW1_flowSinkhorn_update_as_stated_of_forward_nonneg -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/ClosedForms.lean
abbrev thm_graphw1_complexity :=
  @graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/Complexity.lean

end S5GraphW1Main
end Paper
end FlowSinkhorn
