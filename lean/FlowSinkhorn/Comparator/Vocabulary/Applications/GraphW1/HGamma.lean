import Mathlib
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
# Graph-W1 H_gamma statement vocabulary

This proof-free module contains the abstract entropic objective and feasible
minimizer predicate used to state the graph-W1 positive-cost mass bound.  The
proofs live in `HGamma.lean`.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace Applications
namespace GraphW1

open scoped BigOperators

noncomputable section

/--
Positive real-valued field on an index type.

This is used in paper-facing graph-W1 statements for objects denoted
`f,z : E -> R_{++}` in LaTeX.  Keeping positivity in the data type makes the
Lean theorem closer to that mathematical notation and avoids carrying separate
positivity hypotheses for every positive field.
-/
structure PositiveField (ι : Type*) where
  val : ι → ℝ
  pos : ∀ i, 0 < val i

instance {ι : Type*} : CoeFun (PositiveField ι) (fun _ => ι → ℝ) where
  coe f := f.val

/--
Finite minimum cost used in the graph-W1 positive-cost mass bound.

This proof-free, transparent wrapper names the finite coordinate minimum used by the paper.
The implementation proves that this value is positive and below every coordinate from strict
positivity of the finite cost vector.
-/
def graphW1CostMin
    {ι : Type*} [Fintype ι] [Nonempty ι] (C : ι → ℝ) : ℝ :=
  coordMin C

/--
Entropic objective used in the graph-W1 positive-cost mass bound.

The KL term is kept as an abstract functional here, because the lemma only uses
optimality of the full objective and nonnegativity of the KL value at the
minimizer.
-/
def entropicObjective
    {ι : Type*} [Fintype ι]
    (C : ι → ℝ) (gamma : ℝ) (KL : (ι → ℝ) → ℝ) (x : ι → ℝ) : ℝ :=
  (∑ i, C i * x i) + gamma * KL x

/--
Feasible minimizer predicate for the entropic graph-W1 objective.

This packages the exact paper phrase "`xγ⋆` solves the entropic penalized
problem": the point is feasible, and its entropic objective is no larger than
any feasible comparison point.
-/
def IsFeasibleEntropicMinimizer
    {ι : Type*} [Fintype ι]
    (Feasible : (ι → ℝ) → Prop)
    (C : ι → ℝ) (gamma : ℝ) (KL : (ι → ℝ) → ℝ) (xStar : ι → ℝ) : Prop :=
  Feasible xStar ∧
    ∀ y, Feasible y → entropicObjective C gamma KL xStar ≤ entropicObjective C gamma KL y

end

end GraphW1
end Applications
end KLProjection
end FlowSinkhorn
