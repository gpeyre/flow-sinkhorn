import Mathlib.Analysis.SpecialFunctions.Arsinh
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
# Graph-W1 closed-form statement vocabulary

This module contains proof-free definitions needed to state the graph-W1
closed-form propositions in the paper and in the Comparator challenge.  Keeping
these definitions separate from `ClosedForms.lean` lets a trusted challenge
mention the concrete maps without importing the theorem proofs about them.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace Applications
namespace GraphW1

open scoped BigOperators

/--
Helper scalar root used in Proposition `prop:explici-formula-proj`.
-/
noncomputable def graphW1_phi (t u : ℝ) : ℝ :=
  (Real.sqrt (t ^ 2 + 4 * u) - t) / 2

/--
The scalar scaling formula in Proposition `prop:graphw1-projection-closed-form`.
-/
noncomputable def graphW1_C1Scaling {ι : Type*}
    (bDiff hRow hCol : ι → ℝ) : ι → ℝ :=
  fun i => graphW1_phi (bDiff i / hRow i) (hCol i / hRow i)

/--
Left component of the displayed `C1` projection candidate `diag(s) h`.
-/
noncomputable def graphW1_projC1Left {ι : Type*}
    (bDiff hRow hCol : ι → ℝ)
    (h : ι → ι → ℝ) : ι → ι → ℝ :=
  fun i j => graphW1_C1Scaling bDiff hRow hCol i * h i j

/--
Right component of the displayed `C1` projection candidate `h diag(s)^{-1}`.
-/
noncomputable def graphW1_projC1Right {ι : Type*}
    (bDiff hRow hCol : ι → ℝ)
    (h : ι → ι → ℝ) : ι → ι → ℝ :=
  fun i j => h i j / graphW1_C1Scaling bDiff hRow hCol j

/--
Common component of the displayed `C2` projection candidate `sqrt(f ⊙ g)`.
-/
noncomputable def graphW1_projC2Common {ι : Type*}
    (f g : ι → ι → ℝ) : ι → ι → ℝ :=
  fun i j => Real.sqrt (f i j * g i j)

/--
Finite KL objective for one graph-flow array, used only as statement vocabulary for the
graph-W1 projection proposition.
-/
noncomputable def graphW1ArrayKL {ι : Type*} [Fintype ι]
    (x z : ι → ι → ℝ) : ℝ :=
  Finset.univ.sum fun i : ι =>
    Finset.univ.sum fun j : ι =>
      x i j * Real.log (x i j / z i j) - x i j + z i j

/--
Finite pair KL objective for two graph-flow arrays.
-/
noncomputable def graphW1PairKL {ι : Type*} [Fintype ι]
    (f g f0 g0 : ι → ι → ℝ) : ℝ :=
  graphW1ArrayKL f f0 + graphW1ArrayKL g g0

/--
The graph-W1 `C1` divergence constraint used in Proposition
`prop:graphw1-projection-closed-form`.
-/
def GraphW1C1Constraint {ι : Type*} [Fintype ι]
    (bDiff : ι → ℝ) (f g : ι → ι → ℝ) : Prop :=
  ∀ i : ι, -(∑ j : ι, f i j) + (∑ j : ι, g j i) = bDiff i

/--
Nonnegativity predicate for one graph-flow array.
-/
def GraphW1ArrayNonnegative {ι : Type*} (f : ι → ι → ℝ) : Prop :=
  ∀ i j : ι, 0 ≤ f i j

/--
Nonnegativity predicate for a pair of graph-flow arrays.
-/
def GraphW1PairNonnegative {ι : Type*} (f g : ι → ι → ℝ) : Prop :=
  GraphW1ArrayNonnegative f ∧ GraphW1ArrayNonnegative g

/--
Optimality predicate for the `C1` KL projection candidate over the nonnegative flow cone.
-/
def GraphW1C1ProjectionOptimality {ι : Type*} [Fintype ι]
    (bDiff : ι → ℝ) (h left right : ι → ι → ℝ) : Prop :=
  ∀ f g : ι → ι → ℝ,
    GraphW1PairNonnegative f g →
    GraphW1C1Constraint bDiff f g →
      graphW1PairKL left right h h ≤ graphW1PairKL f g h h

/--
Optimality predicate for the `C2` diagonal KL projection candidate over the nonnegative flow cone.
-/
def GraphW1C2ProjectionOptimality {ι : Type*} [Fintype ι]
    (f g common : ι → ι → ℝ) : Prop :=
  ∀ q : ι → ι → ℝ,
    GraphW1ArrayNonnegative q →
      graphW1PairKL common common f g ≤ graphW1PairKL q q f g

/--
Proof-free variational certificate for Proposition `prop:graphw1-projection-closed-form`.

The algebraic part of the paper-facing theorem derives the displayed formulas and the `C1`
constraint.  The remaining variational statement that the candidates are the KL minimizers is
made explicit here, so Comparator sees exactly which projection optimality facts are still supplied
by the concrete KKT/diagonal KL argument.  The nonnegativity fields keep the statement aligned with
the paper's flow cone rather than treating KL as an unconstrained real-array objective.
-/
structure GraphW1ProjectionVariationalCertificate {ι : Type*} [Fintype ι]
    (bDiff hRow hCol : ι → ℝ)
    (h f g : ι → ι → ℝ) : Prop where
  h_nonneg : GraphW1ArrayNonnegative h
  fg_nonneg : GraphW1PairNonnegative f g
  row_sum : ∀ i : ι, hRow i = ∑ j : ι, h i j
  col_sum : ∀ i : ι, hCol i = ∑ j : ι, h j i
  c1_optimality :
    GraphW1C1ProjectionOptimality bDiff h
      (graphW1_projC1Left bDiff hRow hCol h)
      (graphW1_projC1Right bDiff hRow hCol h)
  c2_optimality :
    GraphW1C2ProjectionOptimality f g (graphW1_projC2Common f g)

/--
Finite log-sum-exp operator used in the stable graph-W1 dual update.
-/
noncomputable def graphW1_Lgamma {ι : Type*} [Fintype ι]
    (gamma : ℝ) (s : ι → ℝ) : ℝ :=
  gamma * Real.log (Finset.univ.sum fun j : ι => Real.exp (s j / gamma))

/--
The `α⁺` field from Proposition `prop:graphw1-flow-sinkhorn-update`:
`α_i⁺(v) = L_γ(-w_{i,.} + v/2)`.
-/
noncomputable def graphW1_alphaPlus {ι : Type*} [Fintype ι]
    (w : ι → ι → ℝ) (gamma : ℝ) (v : ι → ℝ) : ι → ℝ :=
  fun i => graphW1_Lgamma gamma (fun j : ι => -w i j + v j / 2)

/--
The `α⁻` field from Proposition `prop:graphw1-flow-sinkhorn-update`:
`α_i⁻(v) = L_γ(-w_{i,.} - v/2)`.
-/
noncomputable def graphW1_alphaMinus {ι : Type*} [Fintype ι]
    (w : ι → ι → ℝ) (gamma : ℝ) (v : ι → ℝ) : ι → ℝ :=
  fun i => graphW1_Lgamma gamma (fun j : ι => -w i j - v j / 2)

/--
The `β` field from Proposition `prop:graphw1-flow-sinkhorn-update`.
-/
noncomputable def graphW1_beta {ι : Type*} [Fintype ι]
    (bDiff : ι → ℝ) (w : ι → ι → ℝ) (gamma : ℝ) (v : ι → ℝ) : ι → ℝ :=
  fun i =>
    (bDiff i / 2) *
      Real.exp (-(graphW1_alphaPlus w gamma v i + graphW1_alphaMinus w gamma v i) / (2 * gamma))

/--
Intermediate scalar used by the log-domain graph-W1 implementation.
-/
noncomputable def graphW1_mUpdate {ι : Type*} [Fintype ι]
    (bDiff : ι → ℝ) (w : ι → ι → ℝ) (gamma : ℝ) (v : ι → ℝ) : ι → ℝ :=
  fun i =>
    (graphW1_alphaMinus w gamma v i - graphW1_alphaPlus w gamma v i) / (2 * gamma) +
      Real.arsinh (graphW1_beta bDiff w gamma v i)

/--
Half-dual update used by the implementation, written directly from the paper's
log-sum-exp and `arsinh` fields.
-/
noncomputable def graphW1_hNextFromDual {ι : Type*} [Fintype ι]
    (bDiff : ι → ℝ) (w : ι → ι → ℝ) (gamma : ℝ) (v : ι → ℝ) : ι → ℝ :=
  fun i => v i / 4 - gamma / 2 * graphW1_mUpdate bDiff w gamma v i

/--
Concrete stable dual sweep map appearing in Proposition
`prop:graphw1-flow-sinkhorn-update`.
-/
noncomputable def graphW1_stableDualSweep {ι : Type*} [Fintype ι]
    (bDiff : ι → ℝ) (w : ι → ι → ℝ) (gamma : ℝ) : (ι → ℝ) → ι → ℝ :=
  fun v i => 2 * graphW1_hNextFromDual bDiff w gamma v i

/--
Proof-free certificate that the two graph-W1 block projection maps used in the paper compose to
the stable dual sweep stated in Proposition `prop:graphw1-flow-sinkhorn-update`.

The trusted Comparator challenge may mention this certificate without importing the proof-bearing
closed-form projection module.  The certificate is deliberately split at the intermediate
log-domain variable `m`: the second block must produce the Lean-defined `m` update, and the first
block must map that `m` update to the displayed stable dual step.  The proof side must instantiate
these two identities from the concrete `C1` and `C2` projection formulas.
-/
structure GraphW1StableProjectionSweepCertificate {ι : Type*} [Fintype ι]
    (Ψ₁ Ψ₂ : (ι → ℝ) → ι → ℝ)
    (bDiff : ι → ℝ) (w : ι → ι → ℝ) (gamma : ℝ) : Prop where
  second_block_eq_mUpdate :
    ∀ v : ι → ℝ, Ψ₂ v = graphW1_mUpdate bDiff w gamma v
  first_block_eq_from_mUpdate :
    ∀ v : ι → ℝ,
      Ψ₁ (graphW1_mUpdate bDiff w gamma v) =
        fun i : ι => v i / 2 - gamma * graphW1_mUpdate bDiff w gamma v i

/--
The graph-W1 second block map on an explicit finite edge set.
-/
noncomputable def graphW1_Psi2 {ι ε : Type*} (src dst : ε → ι) (v : ι → ℝ) : ε → ℝ :=
  fun e => (v (dst e) - v (src e)) / 2

end GraphW1
end Applications
end KLProjection
end FlowSinkhorn
