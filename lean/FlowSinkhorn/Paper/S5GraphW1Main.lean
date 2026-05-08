import FlowSinkhorn.KLProjection.Applications.GraphW1

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
abbrev prop_5_2 := @graphW1_flowSinkhorn_update_as_stated -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/ClosedForms.lean
abbrev thm_5_1 := @graphW1_Sinkhorn_iterationComplexity -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/Complexity.lean

abbrev prop_graphw1_projection_closed_form := @graphW1_projection_closedForm -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/ClosedForms.lean
abbrev prop_graphw1_flow_sinkhorn_update := @graphW1_flowSinkhorn_update_as_stated -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/ClosedForms.lean
abbrev thm_graphw1_complexity := @graphW1_Sinkhorn_iterationComplexity -- impl: lean/FlowSinkhorn/KLProjection/Applications/GraphW1/Complexity.lean

end S5GraphW1Main
end Paper
end FlowSinkhorn
