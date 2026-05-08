import FlowSinkhorn.KLProjection.DualConvergence.PerStepAscent
import FlowSinkhorn.KLProjection.DualConvergence.GapResidual
import FlowSinkhorn.KLProjection.DualConvergence.Rate
import FlowSinkhorn.KLProjection.DualConvergence.Pinsker
import FlowSinkhorn.KLProjection.DualConvergence.Variational

/-!
# Dual convergence blueprint

This section-level module mirrors `papers/kl-projections/sections/sec-dual-convergence.tex`.

Planned paper-faithful order:
1. per-step ascent;
2. gap versus residuals in the quotient seminorm;
3. robust `O(1/k)` dual rate;
4. bias and complexity corollaries.

Design goal:
this should become the Lean entry point for the generic rate theorem before Section 4 is used to
replace abstract orbit bounds by explicit constants.
-/

namespace FlowSinkhorn
namespace KLProjection

end KLProjection
end FlowSinkhorn
