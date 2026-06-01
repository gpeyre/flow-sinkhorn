import Mathlib.Analysis.Seminorm

/-!
Canonical proof-free Comparator vocabulary.

This module is part of the trusted statement language imported by
`FlowSinkhorn.Comparator.Challenge`.  It may define structures, predicates,
and auxiliary notation used to state paper theorems, but it must not contain
paper-facing proofs, theorem declarations, axioms, or opaque constants.
The implementation imports this same vocabulary through compatibility shims,
so Challenge and Solution share one statement language without duplicating
definitions.
-/

/-!
# Uniform-bound statement vocabulary

This module contains the definition needed to state nonexpansive-iterate bounds
without importing the proof-bearing orbit estimates from `UniformBound.lean`.
-/

namespace FlowSinkhorn
namespace KLProjection

variable {𝕜 E : Type*}
variable [NormedField 𝕜] [AddCommGroup E] [Module 𝕜 E]

/-- Non-expansiveness of a map with respect to a seminorm-induced difference distance. -/
def SeminormNonexpansive (p : Seminorm 𝕜 E) (Ψ : E → E) : Prop :=
  ∀ x y, p (Ψ x - Ψ y) ≤ p (x - y)

end KLProjection
end FlowSinkhorn
