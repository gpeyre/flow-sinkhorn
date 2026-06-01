import FlowSinkhorn.Comparator.Vocabulary.BlockQuotient

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
# Sweep statement vocabulary

This module contains the block-translation and sweep definitions used in paper
statements.  Proofs about translation equivariance and iterates remain in
`FlowSinkhorn.KLProjection.Sweep`.
-/

namespace FlowSinkhorn
namespace KLProjection

variable {ι₁ ι₂ : Type*}

/-- First block-translation equivariance relation from Proposition A.3. -/
def SignedBlockTranslationEquivariant1
    (τ : PairedSign) (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ)) : Prop :=
  ∀ u₂ : ι₂ → ℝ, ∀ c : ℝ,
    Ψ₁ (fun j => u₂ j + c) = fun i => Ψ₁ u₂ i + τ.toReal * c

/-- Second block-translation equivariance relation from Proposition A.3. -/
def SignedBlockTranslationEquivariant2
    (τ : PairedSign) (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ)) : Prop :=
  ∀ u₁ : ι₁ → ℝ, ∀ c : ℝ,
    Ψ₂ (fun i => u₁ i + c) = fun j => Ψ₂ u₁ j + τ.toReal * c

/-- Full sweep obtained by composing the two block maps. -/
def sweep (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ)) (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ)) :
    (ι₁ → ℝ) → (ι₁ → ℝ) :=
  fun u₁ => Ψ₁ (Ψ₂ u₁)

end KLProjection
end FlowSinkhorn
