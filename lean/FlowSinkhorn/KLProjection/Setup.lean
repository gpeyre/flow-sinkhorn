import FlowSinkhorn.KLProjection.Topical
import FlowSinkhorn.KLProjection.Sweep
import FlowSinkhorn.KLProjection.Setup.VariationGeometry
import FlowSinkhorn.KLProjection.Setup.BlockMonotonicity
import FlowSinkhorn.KLProjection.Setup.Translation

/-!
# KL-Projection setup layer

This section-level module mirrors the setup portion of the paper.

It is designed to gather, in the same logical order as the manuscript:
- quotient / variation geometry;
- monotonicity of the block maps and moment maps;
- signed paired-balance translation structure;
- sweep-level translation equivariance.

Paper role:
this module should eventually provide the exact certified setup needed by the generic convergence
blueprint without importing any application-specific constant.

See also:
- `lean/FORMALIZATION_PLAN.md`
- `papers/kl-projections/sections/sec-lean-formalization.tex`
-/

namespace FlowSinkhorn
namespace KLProjection

end KLProjection
end FlowSinkhorn
