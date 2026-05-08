import FlowSinkhorn.KLProjection.Setup

/-!
# Paper Appendix G: Setup / Nonexpansiveness

Paper-facing facade for Appendix G results.
Canonical proof homes remain under `FlowSinkhorn.KLProjection.Setup.*`.
-/

namespace FlowSinkhorn
namespace Paper
namespace AppendixGSetup

open KLProjection
open KLProjection.Setup

abbrev prop_G_1 := @variationSeminorm_nonexpansive_of_topical_shifted -- impl: lean/FlowSinkhorn/KLProjection/Setup/VariationGeometry.lean
abbrev prop_G_2 := @blockUpdate_monotone -- impl: lean/FlowSinkhorn/KLProjection/Setup/BlockMonotonicity.lean
abbrev lem_G_1 := @momentMap_monotone -- impl: lean/FlowSinkhorn/KLProjection/Setup/BlockMonotonicity.lean
abbrev prop_G_3 := @sweep_translationEquivariant -- impl: lean/FlowSinkhorn/KLProjection/Setup/Translation.lean

abbrev prop_topical_nonexpansive := @variationSeminorm_nonexpansive_of_topical_shifted -- impl: lean/FlowSinkhorn/KLProjection/Setup/VariationGeometry.lean
abbrev prop_block_monotone := @blockUpdate_monotone -- impl: lean/FlowSinkhorn/KLProjection/Setup/BlockMonotonicity.lean
abbrev lem_moment_monotone := @momentMap_monotone -- impl: lean/FlowSinkhorn/KLProjection/Setup/BlockMonotonicity.lean
abbrev prop_translation_equivariance := @sweep_translationEquivariant -- impl: lean/FlowSinkhorn/KLProjection/Setup/Translation.lean

end AppendixGSetup
end Paper
end FlowSinkhorn
