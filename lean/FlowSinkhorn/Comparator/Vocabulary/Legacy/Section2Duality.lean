import Mathlib

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
# Section-2 duality statement vocabulary

This proof-free module contains the definitions needed to state Proposition
`prop:dual-gamma-correct`: tilted kernels, primal reconstruction from dual
scores, dual objectives, and primal/dual optimality predicates.  The proof of
the proposition remains in `Section2Duality.lean`.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace Section2Duality

open scoped BigOperators

noncomputable section

variable {m d : Type*}

/-- Tilted Gibbs kernel `zC_i = z_i * exp(-C_i / γ)`. -/
def tiltedKernel (z C : d → ℝ) (gamma : ℝ) : d → ℝ :=
  fun i => z i * Real.exp (-C i / gamma)

/-- Primal variable reconstructed from dual scores (`score = Aᵀu` in the paper). -/
def primalFromDualScore (z C : d → ℝ) (gamma : ℝ) (score : d → ℝ) : d → ℝ :=
  fun i => z i * Real.exp ((score i - C i) / gamma)

/-- Dual objective written with `(z, C)` coordinates. -/
def dualObjective_from_zC
    [Fintype d]
    (z C : d → ℝ) (gamma Z : ℝ) (b : m → ℝ)
    (pairing : (m → ℝ) → (m → ℝ) → ℝ)
    (scoreMap : (m → ℝ) → (d → ℝ))
    (u : m → ℝ) : ℝ :=
  pairing b u + gamma * Z
    - gamma * ∑ i : d, z i * Real.exp ((scoreMap u i - C i) / gamma)

/-- Dual objective written with tilted kernel coordinates. -/
def dualObjective_from_kernel
    [Fintype d]
    (zC : d → ℝ) (gamma Z : ℝ) (b : m → ℝ)
    (pairing : (m → ℝ) → (m → ℝ) → ℝ)
    (scoreMap : (m → ℝ) → (d → ℝ))
    (u : m → ℝ) : ℝ :=
  pairing b u + gamma * Z
    - gamma * ∑ i : d, zC i * Real.exp (scoreMap u i / gamma)

/-- A feasible point whose primal objective is no larger than every feasible point. -/
def IsPrimalMinimizer
    (feasible : (d → ℝ) → Prop) (primalObjective : (d → ℝ) → ℝ)
    (x : d → ℝ) : Prop :=
  feasible x ∧ ∀ y : d → ℝ, feasible y → primalObjective x ≤ primalObjective y

/-- A primal minimizer which is the only primal minimizer. -/
def IsUniquePrimalMinimizer
    (feasible : (d → ℝ) → Prop) (primalObjective : (d → ℝ) → ℝ)
    (x : d → ℝ) : Prop :=
  IsPrimalMinimizer feasible primalObjective x ∧
    ∀ y : d → ℝ, IsPrimalMinimizer feasible primalObjective y → y = x

/-- A dual point whose dual objective is no smaller than every dual value. -/
def IsDualMaximizer
    (dualObjective : (m → ℝ) → ℝ) (u : m → ℝ) : Prop :=
  ∀ v : m → ℝ, dualObjective v ≤ dualObjective u

/--
Named primal-dual certificate for Proposition `prop:dual-gamma-correct`.

The paper proof obtains these fields from the finite-dimensional
KKT/Fenchel-duality argument: normalization and positivity of the reference
measure, weak duality, feasibility plus zero gap at the primal-from-dual point,
strict-convexity uniqueness of the primal minimizer, and the gradient identity
for the dual objective.  Keeping them in one structure makes the remaining
bridge explicit at the paper-facing theorem boundary.
-/
structure DualGammaPrimalDualCertificate
    [Fintype d]
    (A : (d → ℝ) →ₗ[ℝ] (m → ℝ))
    (z C : d → ℝ) (gamma Z : ℝ) (b : m → ℝ)
    (pairing : (m → ℝ) → (m → ℝ) → ℝ)
    (scoreMap : (m → ℝ) → (d → ℝ))
    (gradF : (m → ℝ) → (m → ℝ))
    (feasible : (d → ℝ) → Prop)
    (primalObjective : (d → ℝ) → ℝ)
    (uStar : m → ℝ) : Prop where
  mass_normalization : Z = ∑ i : d, z i
  reference_positive : ∀ i : d, 0 < z i
  gamma_positive : 0 < gamma
  weak_duality :
    ∀ (x : d → ℝ) (u : m → ℝ),
      feasible x →
        dualObjective_from_zC z C gamma Z b pairing scoreMap u ≤ primalObjective x
  primal_from_dual_feasible :
    feasible (primalFromDualScore z C gamma (scoreMap uStar))
  zero_gap :
    primalObjective (primalFromDualScore z C gamma (scoreMap uStar)) =
      dualObjective_from_zC z C gamma Z b pairing scoreMap uStar
  unique_by_value :
    ∀ x : d → ℝ,
      feasible x →
        primalObjective x =
          primalObjective (primalFromDualScore z C gamma (scoreMap uStar)) →
        x = primalFromDualScore z C gamma (scoreMap uStar)
  gradient_identity :
    gradF uStar =
      fun j => b j - A (primalFromDualScore z C gamma (scoreMap uStar)) j

end
end Section2Duality
end KLProjection
end FlowSinkhorn
