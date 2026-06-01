import Mathlib.Data.Real.Basic
import Mathlib.Order.ConditionallyCompleteLattice.Finset
import Mathlib.Order.Monotone.Basic

noncomputable section

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
# Variation-seminorm statement vocabulary

This module contains the raw definitions needed to state variation-seminorm and
topical-map results.  Proofs about these definitions remain in
`FlowSinkhorn.KLProjection.Variation` and `FlowSinkhorn.KLProjection.Topical`.
-/

namespace FlowSinkhorn
namespace KLProjection

variable {ι : Type*} [Fintype ι] [Nonempty ι]

/-- Maximum coordinate of a finite real vector. -/
def coordMax (x : ι → ℝ) : ℝ :=
  Finset.sup' Finset.univ Finset.univ_nonempty x

/-- Minimum coordinate of a finite real vector. -/
def coordMin (x : ι → ℝ) : ℝ :=
  Finset.inf' Finset.univ Finset.univ_nonempty x

/-- Oscillation of a finite real vector, i.e. `max(x) - min(x)`. -/
def oscillation (x : ι → ℝ) : ℝ :=
  coordMax x - coordMin x

/-- Variation seminorm, equal to half the oscillation. -/
def variationSeminorm (x : ι → ℝ) : ℝ :=
  oscillation x / 2

/-- Translation equivariance for pointwise addition of constants. -/
def TranslationEquivariant (T : (ι → ℝ) → (ι → ℝ)) : Prop :=
  ∀ x : ι → ℝ, ∀ c : ℝ, T (fun i => x i + c) = fun i => T x i + c

/-- Shift placing the interval `[coordMin x, coordMax x]` symmetrically around the origin. -/
def centeringShift (x : ι → ℝ) : ℝ :=
  -((coordMax x + coordMin x) / 2)

end KLProjection
end FlowSinkhorn
