import FlowSinkhorn.Comparator.Vocabulary.PrimalDualBounds
import FlowSinkhorn.Comparator.Vocabulary.Sweep
import FlowSinkhorn.Comparator.Vocabulary.Variation
import Mathlib.Analysis.SpecialFunctions.Log.Basic

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
# Graph-W1 complexity statement vocabulary

This file contains proof-free vocabulary needed to state the graph-W1 complexity
result.  It is intentionally separated from `GraphW1/Complexity.lean` so that
the Comparator challenge can import the definition without importing the
proof-bearing graph-W1 complexity endpoint.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace Applications
namespace GraphW1

/--
Two graph-`W1` block maps with the order and signed-translation laws needed for the
topical full-sweep argument.

This proof-free record is used by the Comparator challenge to keep the paper statement
structured: the actual proof that these laws imply a nonexpansive orbit bound remains in
`GraphW1/Complexity.lean`.
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
Fixed point together with its `H_gamma`/`kappa` budget.

This records the paper's fixed-point hypothesis and its certified budget in one
named finite-dimensional certificate.
-/
structure SweepFixedPointBudget
    (ι : Type*) [Fintype ι] [Nonempty ι]
    (Ψ : (ι → ℝ) → (ι → ℝ))
    (kappa lengthMax gamma hGamma : ℝ) where
  vStar : ι → ℝ
  fixed : Ψ vStar = vStar
  budget :
    variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa lengthMax gamma hGamma

/--
Two bounded edge fields used by the graph-`W1` two-step path estimate.

The scalar bound `B ≤ 1` is stored explicitly.  Nonnegativity of `B` is derived in
the proof from the absolute-value bounds and nonemptiness of the vertex set.
-/
structure UnitBoundedTwoStepFields (ι : Type*) where
  B : ℝ
  le_one : B ≤ 1
  yf : ι × ι → ℝ
  yg : ι × ι → ℝ
  yf_bound : ∀ p : ι × ι, |yf p| ≤ B
  yg_bound : ∀ p : ι × ι, |yg p| ≤ B

/--
Finite two-step path certificate controlling `kappa`.

Each path increment is one of the certified edge-field sums, the list length is at
most the graph diameter, and its total controls `kappa`.
-/
structure TwoStepPathCertificate
    {ι : Type*} (edge : UnitBoundedTwoStepFields ι)
    (graphDiam : ℕ) (kappa : ℝ) where
  steps : List ℝ
  length_le : steps.length ≤ graphDiam
  steps_from_edge : ∀ x ∈ steps, ∃ p : ι × ι, x = (edge.yf + edge.yg) p
  kappa_le_abs_sum : kappa ≤ |steps.sum|

/--
Pointwise primal-mass proxy used to convert the orbit bound into an `X_gamma`
bound.
-/
structure GraphW1MassProxy
    (ι : Type*) [Fintype ι] [Nonempty ι]
    (Ψ : (ι → ℝ) → (ι → ℝ))
    (gamma bMass p lengthMin : ℝ) where
  xMass : ℕ → ℝ
  bMass_nonneg : 0 ≤ bMass
  pointwise :
    ∀ k : ℕ,
      xMass k ≤
        bMass * variationSeminorm ((Ψ^[k]) (0 : ι → ℝ)) / gamma +
          p * Real.exp (-lengthMin / gamma)

/--
Local little-o regime used by the paper statement `p=o(1/log(1/eps))`.

The asymptotic side condition is kept explicit in Lean rather than hidden in
prose: near `eps = 0`, the edge-count family `pOfEps` is dominated by
`1 / log(1/eps)`.
-/
def graphW1LittleOEdgeRegime (pOfEps : ℝ → ℝ) : Prop :=
  ∀ η : ℝ, 0 < η →
    ∃ δ : ℝ, 0 < δ ∧
      ∀ eps : ℝ, 0 < eps → eps < δ →
        |pOfEps eps| ≤ η * |1 / Real.log (1 / eps)|

/--
Operation-budget certificate for the graph-W1 complexity statement.

The main theorem states the final arithmetic operation bound.  This record names the
algorithmic ingredients that feed that bound: epsilon accuracy at one iterate, the iteration
budget, sparse per-sweep work, total operation accounting, and the local little-o edge-count
regime.  Keeping these fields in statement vocabulary makes the Comparator challenge less
dependent on a long list of anonymous scalar hypotheses.
-/
structure GraphW1OperationBudgetCertificate
    (w1Error : ℕ → ℝ)
    (eps p graphDiam logFactor iterationBudget perSweepOps operationCount : ℝ)
    (pOfEps : ℝ → ℝ) (k : ℕ) : Prop where
  eps_pos : 0 < eps
  edge_nonneg : 0 ≤ p
  accuracy : w1Error k ≤ eps
  iteration_index_le_budget : (k : ℝ) ≤ iterationBudget
  iteration_budget : iterationBudget ≤ logFactor * graphDiam ^ 3 / eps ^ 4
  per_sweep_ops : perSweepOps ≤ p
  operation_count : operationCount ≤ (k : ℝ) * perSweepOps
  edge_count_eval : p = pOfEps eps
  edge_count_littleO : graphW1LittleOEdgeRegime pOfEps

end GraphW1
end Applications
end KLProjection
end FlowSinkhorn
