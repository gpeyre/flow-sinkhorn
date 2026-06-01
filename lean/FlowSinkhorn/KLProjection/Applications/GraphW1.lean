import FlowSinkhorn.KLProjection.Applications.GraphW1.ClosedFormsVocabulary
import FlowSinkhorn.KLProjection.Applications.GraphW1.ClosedForms
import FlowSinkhorn.KLProjection.Applications.GraphW1.HGammaVocabulary
import FlowSinkhorn.KLProjection.Applications.GraphW1.HGamma
import FlowSinkhorn.KLProjection.Applications.GraphW1.Kappa
import FlowSinkhorn.KLProjection.Applications.GraphW1.ComplexityVocabulary
import FlowSinkhorn.KLProjection.Applications.GraphW1.Complexity

/-!
# Graph Wasserstein-1 / flow-Sinkhorn instantiation

This section-level module mirrors the graph-W1 material in `neurips/paper.tex`.

Planned paper-faithful order:
1. closed-form block updates and non-expansiveness ingredients;
2. `H_γ` bound;
3. `κ` bound;
4. explicit `X_γ`, `U_γ`, and the final complexity corollary.

Design goal:
this module should become the clean application layer sitting on top of the generic blueprint and
its certified setup/non-expansiveness infrastructure.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace Applications
namespace GraphW1

end GraphW1
end Applications
end KLProjection
end FlowSinkhorn
