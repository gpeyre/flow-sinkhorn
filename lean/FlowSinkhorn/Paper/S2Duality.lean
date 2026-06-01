import FlowSinkhorn.KLProjection.Duality

/-!
# Paper Section 2: Duality Core

Paper-facing facade for Section 2 results.
Canonical proof home remains `FlowSinkhorn.KLProjection.Legacy.Section2Duality`.
-/

namespace FlowSinkhorn
namespace Paper
namespace S2Duality

open KLProjection
open KLProjection.Section2Duality

-- impl: lean/FlowSinkhorn/KLProjection/Legacy/Section2Duality.lean
abbrev prop_2_1 := @dualGammaCorrect_from_primalDualCertificate
-- impl: lean/FlowSinkhorn/KLProjection/Legacy/Section2Duality.lean
abbrev prop_dual_gamma_correct := @dualGammaCorrect_from_primalDualCertificate

end S2Duality
end Paper
end FlowSinkhorn
