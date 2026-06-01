import FlowSinkhorn.Comparator.Vocabulary.Variation

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
# Topical-map statement vocabulary

This proof-free module contains the bundled `IsTopical` predicate used in paper
statements.  The nonexpansiveness theory for topical maps remains in
`Topical.lean`.
-/

namespace FlowSinkhorn
namespace KLProjection

variable {ι : Type*}

/--
A map on finite real-valued functions is **topical** if it is monotone and
translation-equivariant.

This bundles the two hypotheses of Proposition `prop:topical-nonexpansive`:
monotone + translation-equivariant implies variation-seminorm-nonexpansive.
-/
structure IsTopical (T : (ι → ℝ) → (ι → ℝ)) : Prop where
  /-- The map is monotone with respect to the pointwise order. -/
  mono : Monotone T
  /-- The map commutes with adding a constant to all coordinates. -/
  trans : TranslationEquivariant T

end KLProjection
end FlowSinkhorn
