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

abbrev prop_G_1 := @variationSeminorm_nonexpansive_of_topical -- impl: lean/FlowSinkhorn/KLProjection/Topical.lean
abbrev prop_G_2 := @blockUpdate_antitoneRelation_then_sweep_monotone -- impl: lean/FlowSinkhorn/KLProjection/Setup/BlockMonotonicity.lean
abbrev lem_G_1 :=
  @momentMap_monotone_of_nonnegative_linear_layers -- impl: lean/FlowSinkhorn/KLProjection/Setup/BlockMonotonicity.lean
abbrev prop_G_3 := @translationEquivariance_of_pairedBalance_blockLaws -- impl: lean/FlowSinkhorn/KLProjection/Setup/Translation.lean

abbrev prop_topical_nonexpansive := @variationSeminorm_nonexpansive_of_topical -- impl: lean/FlowSinkhorn/KLProjection/Topical.lean
abbrev prop_block_monotone := @blockUpdate_antitoneRelation_then_sweep_monotone -- impl: lean/FlowSinkhorn/KLProjection/Setup/BlockMonotonicity.lean
abbrev lem_moment_monotone :=
  @momentMap_monotone_of_nonnegative_linear_layers -- impl: lean/FlowSinkhorn/KLProjection/Setup/BlockMonotonicity.lean
abbrev prop_translation_equivariance := @translationEquivariance_of_pairedBalance_blockLaws -- impl: lean/FlowSinkhorn/KLProjection/Setup/Translation.lean

end AppendixGSetup
end Paper
end FlowSinkhorn
