import FlowSinkhorn.KLProjection.Applications.OT.HGamma
import FlowSinkhorn.KLProjection.Applications.OT.Kappa
import FlowSinkhorn.KLProjection.Applications.OT.ComplexityVocabulary
import FlowSinkhorn.KLProjection.Applications.OT.Complexity

/-!
# Balanced optimal transport instantiation

This section-level module mirrors the OT material in `neurips/paper.tex`.

Planned paper-faithful order:
1. `H_γ` bound;
2. `κ = 1`;
3. explicit `X_γ`, `U_γ`, and the resulting complexity theorem.

Design goal:
this application should read like a clean instantiation of the generic blueprint, with no changes
back to the generic theory.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace Applications
namespace OT

end OT
end Applications
end KLProjection
end FlowSinkhorn
