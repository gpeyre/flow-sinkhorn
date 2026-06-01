import FlowSinkhorn.KLProjection.PrimalDualBounds

set_option linter.style.longLine false

/-!
# Paper Section 4: Primal/Dual Bounds

Paper-facing facade for Section 4 results.
Canonical proof homes remain under `FlowSinkhorn.KLProjection.PrimalDualBounds.*`.
-/

namespace FlowSinkhorn
namespace Paper
namespace S4PrimalDualBounds

open KLProjection
open KLProjection.PrimalDualBounds

abbrev prop_4_1 :=
  @uniformIterateBound_of_nonexpansive_of_budget -- impl: lean/FlowSinkhorn/KLProjection/PrimalDualBounds/FixedPointControl.lean
abbrev prop_4_2 :=
  @primalMassBound_from_zeroStartFinitePairing_exactL1_card_quotientRadiusCertificate -- impl: lean/FlowSinkhorn/KLProjection/PrimalDualBounds/PrimalFromDual.lean

abbrev prop_uniform_iter_final :=
  @uniformIterateBound_of_nonexpansive_of_budget -- impl: lean/FlowSinkhorn/KLProjection/PrimalDualBounds/FixedPointControl.lean
abbrev prop_mass_bound_block :=
  @primalMassBound_from_zeroStartFinitePairing_exactL1_card_quotientRadiusCertificate -- impl: lean/FlowSinkhorn/KLProjection/PrimalDualBounds/PrimalFromDual.lean

end S4PrimalDualBounds
end Paper
end FlowSinkhorn
