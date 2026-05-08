import Mathlib

/-!
# Section-2 duality layer

Lean-side home for Proposition `prop:dual-gamma-correct` from
`paper/sections/sec-iterative-kl-proj.tex`.

This module certifies the algebraic core used by the paper:
- primal-from-dual exponential map;
- equivalence of the two dual objective formulas (`z`-form and `zC`-form);
- stationarity equivalence `∇Fγ(u)=0 ↔ A x(u)=b` once the gradient identity is provided.
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

/--
Objective-form equivalence used in Proposition `prop:dual-gamma-correct`:
the two formulas of `F_γ` coincide when `zC_i = z_i * exp(-C_i/γ)`.
-/
theorem dualObjective_form_eq
    [Fintype d]
    (z C : d → ℝ) (gamma Z : ℝ) (b : m → ℝ)
    (pairing : (m → ℝ) → (m → ℝ) → ℝ)
    (scoreMap : (m → ℝ) → (d → ℝ))
    (u : m → ℝ) :
    dualObjective_from_zC z C gamma Z b pairing scoreMap u =
      dualObjective_from_kernel (tiltedKernel z C gamma) gamma Z b pairing scoreMap u := by
  unfold dualObjective_from_zC dualObjective_from_kernel tiltedKernel
  have hsum :
      (∑ i : d, z i * Real.exp ((scoreMap u i - C i) / gamma)) =
      ∑ i : d, z i * Real.exp (-C i / gamma) * Real.exp (scoreMap u i / gamma) := by
    apply Finset.sum_congr rfl
    intro i hi
    calc
      z i * Real.exp ((scoreMap u i - C i) / gamma)
          = z i * (Real.exp (scoreMap u i / gamma) * Real.exp (-C i / gamma)) := by
              congr 1
              have hsplit :
                  (scoreMap u i - C i) / gamma = scoreMap u i / gamma + (-C i / gamma) := by
                ring
              rw [hsplit, Real.exp_add]
      _ = z i * Real.exp (-C i / gamma) * Real.exp (scoreMap u i / gamma) := by ring
  rw [hsum]

/--
Stationarity equivalence from the gradient identity:
`∇Fγ(u) = b - A(x(u))` implies
`∇Fγ(u)=0 ↔ A(x(u))=b`.
-/
theorem stationarity_iff_constraints
    (A : (d → ℝ) →ₗ[ℝ] (m → ℝ))
    (z C : d → ℝ) (gamma : ℝ)
    (scoreMap : (m → ℝ) → (d → ℝ))
    (b : m → ℝ)
    (gradF : (m → ℝ) → (m → ℝ))
    (u : m → ℝ)
    (hgrad :
      gradF u = fun j => b j - A (primalFromDualScore z C gamma (scoreMap u)) j) :
    gradF u = 0 ↔ A (primalFromDualScore z C gamma (scoreMap u)) = b := by
  constructor
  · intro h0
    rw [hgrad] at h0
    funext j
    have hj := congrArg (fun f : m → ℝ => f j) h0
    have hj' : b j - A (primalFromDualScore z C gamma (scoreMap u)) j = 0 := by
      simpa using hj
    linarith
  · intro hAx
    rw [hgrad]
    funext j
    have hj := congrArg (fun f : m → ℝ => f j) hAx
    have hj' : A (primalFromDualScore z C gamma (scoreMap u)) j = b j := by
      simpa using hj
    have : b j - A (primalFromDualScore z C gamma (scoreMap u)) j = 0 := by
      linarith
    simpa using this

/--
Primal-dual link in explicit map form (paper equation `x^* = x(u^*)`).
-/
theorem primal_from_dual_explicit
    (z C : d → ℝ) (gamma : ℝ) (score : d → ℝ) :
    primalFromDualScore z C gamma score =
      fun i => z i * Real.exp ((score i - C i) / gamma) := rfl

/--
Single theorem packaging the certified algebraic core used for
Proposition `prop:dual-gamma-correct` in the paper.
-/
theorem dualGammaCorrect_core
    [Fintype d]
    (A : (d → ℝ) →ₗ[ℝ] (m → ℝ))
    (z C : d → ℝ) (gamma Z : ℝ) (b : m → ℝ)
    (pairing : (m → ℝ) → (m → ℝ) → ℝ)
    (scoreMap : (m → ℝ) → (d → ℝ))
    (gradF : (m → ℝ) → (m → ℝ))
    (u : m → ℝ)
    (hgrad :
      gradF u = fun j => b j - A (primalFromDualScore z C gamma (scoreMap u)) j) :
    dualObjective_from_zC z C gamma Z b pairing scoreMap u =
      dualObjective_from_kernel (tiltedKernel z C gamma) gamma Z b pairing scoreMap u
    ∧ primalFromDualScore z C gamma (scoreMap u) =
      (fun i => z i * Real.exp ((scoreMap u i - C i) / gamma))
    ∧ (gradF u = 0 ↔ A (primalFromDualScore z C gamma (scoreMap u)) = b) := by
  refine ⟨dualObjective_form_eq z C gamma Z b pairing scoreMap u, ?_, ?_⟩
  · simpa using primal_from_dual_explicit z C gamma (scoreMap u)
  · exact stationarity_iff_constraints A z C gamma scoreMap b gradF u hgrad

end

end Section2Duality
end KLProjection
end FlowSinkhorn
