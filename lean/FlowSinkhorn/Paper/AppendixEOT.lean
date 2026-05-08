import FlowSinkhorn.KLProjection.Applications.OT

/-!
# Paper Appendix E: OT Instantiation

Paper-facing facade for Appendix E results.
Canonical proof homes remain under `FlowSinkhorn.KLProjection.Applications.OT.*`.
-/

namespace FlowSinkhorn
namespace Paper
namespace AppendixEOT

open KLProjection
open KLProjection.Applications
open KLProjection.Applications.OT

abbrev prop_E_1 := @ot_HGamma_bound -- impl: lean/FlowSinkhorn/KLProjection/Applications/OT/HGamma.lean
abbrev prop_E_2 := @ot_kappa_eq_one -- impl: lean/FlowSinkhorn/KLProjection/Applications/OT/Kappa.lean
abbrev cor_E_1 := @ot_explicit_XGamma_UGamma -- impl: lean/FlowSinkhorn/KLProjection/Applications/OT/Complexity.lean

abbrev prop_hgamma_ot := @ot_HGamma_bound -- impl: lean/FlowSinkhorn/KLProjection/Applications/OT/HGamma.lean
abbrev prop_kappa_ot := @ot_kappa_eq_one -- impl: lean/FlowSinkhorn/KLProjection/Applications/OT/Kappa.lean
abbrev cor_ot_xgamma_ugamma := @ot_explicit_XGamma_UGamma -- impl: lean/FlowSinkhorn/KLProjection/Applications/OT/Complexity.lean

end AppendixEOT
end Paper
end FlowSinkhorn
