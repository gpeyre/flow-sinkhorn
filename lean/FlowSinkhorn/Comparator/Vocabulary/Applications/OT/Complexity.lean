import FlowSinkhorn.Comparator.Vocabulary.BlockQuotient
import FlowSinkhorn.Comparator.Vocabulary.PrimalDualBounds
import FlowSinkhorn.Comparator.Vocabulary.Sweep

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
# OT complexity statement vocabulary

This proof-free module contains structured certificates used to state the OT
zero-start orbit corollary in the Comparator challenge without importing the
proof-bearing OT complexity file.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace Applications
namespace OT

/--
Two OT Sinkhorn block maps with the order and signed-translation laws needed
to derive topicality of the full sweep.

The record only packages hypotheses stated in the paper.  The proof that these
laws imply the zero-start orbit bound lives in `Applications/OT/Complexity.lean`.
-/
structure SignedBlockSweepData (ι₁ ι₂ : Type*) where
  τ : PairedSign
  Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ)
  Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ)
  mono₁ : Monotone Ψ₁
  mono₂ : Monotone Ψ₂
  trans₁ : SignedBlockTranslationEquivariant1 τ Ψ₁
  trans₂ : SignedBlockTranslationEquivariant2 τ Ψ₂

/--
Scalar side conditions for the OT zero-start orbit corollary.

This record is proof-free bookkeeping for the paper assumptions
`gamma > 0`, `min_b > 0`, and `C_max >= 0`.
-/
structure ComplexityScalars (gamma min_b C_max : ℝ) where
  gamma_pos : 0 < gamma
  min_b_pos : 0 < min_b
  C_max_nonneg : 0 ≤ C_max

/--
Separable fixed-point certificate for the OT zero-start orbit corollary.

It records exactly the fixed point, the auxiliary `β*`, the reference column,
the separable decomposition `α*_i + β*_j = Y_ij`, and the displayed
`H_gamma`/`kappa` budget used in Appendix E.
-/
structure SeparableFixedPointCertificate
    (ι₁ ι₂ : Type*) [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (block : SignedBlockSweepData ι₁ ι₂)
    (gamma min_b C_max : ℝ) where
  alphaStar : ι₁ → ℝ
  betaStar : ι₂ → ℝ
  j₀ : ι₂
  Y : ι₁ × ι₂ → ℝ
  fixed : (sweep block.Ψ₁ block.Ψ₂) alphaStar = alphaStar
  separable : ∀ i j, alphaStar i + betaStar j = Y (i, j)
  budget :
    coordSupNorm Y ≤
      PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma)

end OT
end Applications
end KLProjection
end FlowSinkhorn
