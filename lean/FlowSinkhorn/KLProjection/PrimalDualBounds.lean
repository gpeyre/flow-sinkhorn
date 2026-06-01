import FlowSinkhorn.KLProjection.PrimalDualBounds.PrimalFromDual
import FlowSinkhorn.KLProjection.PrimalDualBounds.FixedPointControl
import FlowSinkhorn.KLProjection.PrimalDualBounds.Blueprint

/-!
# Primal/dual bounding layer

This section-level module mirrors the primal-dual bounds material in `neurips/paper.tex`.

Planned paper-faithful order:
1. primal bound from a dual bound (`X_γ` side);
2. definitions and estimates for `H_γ` and `κ`;
3. uniform dual orbit bound from non-expansiveness plus fixed-point control;
4. packaging with the generic dual-rate theorem;
5. one abstract Section 3+4 bundle for applications.

Design goal:
this module should be the exact bridge between the abstract convergence theorem and the explicit
constants proved later in the OT and graph-`W₁` applications.
-/

namespace FlowSinkhorn
namespace KLProjection

end KLProjection
end FlowSinkhorn
