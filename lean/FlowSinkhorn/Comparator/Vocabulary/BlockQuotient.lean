import FlowSinkhorn.Comparator.Vocabulary.Variation
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
# Block-quotient statement vocabulary

This module contains the quotient-norm and paired-gauge definitions needed to
state paper-facing results.  Proofs about these objects remain in
`FlowSinkhorn.KLProjection.BlockQuotient`.
-/

namespace FlowSinkhorn
namespace KLProjection

/-- Coordinatewise sup norm on a finite real vector. -/
noncomputable def coordSupNorm {ι : Type*} [Fintype ι] [Nonempty ι] (x : ι → ℝ) : ℝ :=
  Finset.sup' Finset.univ Finset.univ_nonempty fun i => |x i|

noncomputable section BlockPairs

variable {ι₁ ι₂ : Type*}
variable [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]

/-- A pair of finite-dimensional real dual blocks. -/
abbrev BlockPair (ι₁ ι₂ : Type*) :=
  (ι₁ → ℝ) × (ι₂ → ℝ)

/-- Sup norm on a pair of blocks. -/
def blockSupNorm (u : BlockPair ι₁ ι₂) : ℝ :=
  max (coordSupNorm u.1) (coordSupNorm u.2)

/--
Classical OT paired gauge shift: add a constant to the first block and subtract it from the second.
-/
def pairedShift (c : ℝ) (u : BlockPair ι₁ ι₂) : BlockPair ι₁ ι₂ :=
  (fun i => u.1 i + c, fun j => u.2 j - c)

/-- The orbit of block sup norms obtained by all paired shifts of `u`. -/
def pairedShiftNormSet (u : BlockPair ι₁ ι₂) : Set ℝ :=
  Set.range fun c : ℝ => blockSupNorm (pairedShift c u)

/-- A simple infimum-based quotient sup seminorm for the paired OT gauge direction. -/
def pairedQuotientSupSeminorm (u : BlockPair ι₁ ι₂) : ℝ :=
  sInf (pairedShiftNormSet u)

/--
A shift-invariant lower bound coming from the single-block variation seminorms.

This is the first bridge toward the block-quotient seminorm used in the paper.
-/
def pairedVariationLowerBound (u : BlockPair ι₁ ι₂) : ℝ :=
  max (variationSeminorm u.1) (variationSeminorm u.2)

/--
Sign parameter for the two-block paired-balance action.

`minus` recovers the classical OT gauge direction, while `plus` models the same-sign paired shift
used in the graph-flow splitting from the paper.
-/
inductive PairedSign where
  | plus
  | minus
  deriving DecidableEq, Repr

/-- Real value of the paired-balance sign, equal to `±1`. -/
def PairedSign.toReal : PairedSign → ℝ
  | .plus => 1
  | .minus => -1

/-- Signed two-block paired shift, with second block shifted by `tau * c`. -/
def signedPairedShift (τ : PairedSign) (c : ℝ) (u : BlockPair ι₁ ι₂) : BlockPair ι₁ ι₂ :=
  (fun i => u.1 i + c, fun j => u.2 j + τ.toReal * c)

/-- Orbit of block sup norms for the signed paired-balance action. -/
def signedPairedShiftNormSet (τ : PairedSign) (u : BlockPair ι₁ ι₂) : Set ℝ :=
  Set.range fun c : ℝ => blockSupNorm (signedPairedShift τ c u)

/-- Infimum-based quotient sup seminorm for the signed paired-balance action. -/
def signedPairedQuotientSupSeminorm (τ : PairedSign) (u : BlockPair ι₁ ι₂) : ℝ :=
  sInf (signedPairedShiftNormSet τ u)

end BlockPairs
end KLProjection
end FlowSinkhorn
