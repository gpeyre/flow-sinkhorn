import FlowSinkhorn.KLProjection.Legacy.Section2DualityVocabulary
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

/--
Primal-dual certificate form of Proposition `prop:dual-gamma-correct`.

This theorem upgrades the algebraic core above to the paper-facing optimizer/value statement.
It does not assume the final conclusion.  Instead, it takes the standard finite-dimensional
certificate data used by a KKT/Fenchel proof:

* weak duality for all feasible primal points and all dual variables;
* one feasible primal-from-dual point with zero primal-dual gap;
* uniqueness of the primal optimum from equality of objective values.

From these inputs it proves equality of primal and dual values at the certified pair, uniqueness of
the primal minimizer, maximality of the dual point, the explicit exponential primal reconstruction,
and the stationarity/constraint equivalence.
-/
theorem dualGammaCorrect_primalDualCertificate
    [Fintype d]
    (A : (d → ℝ) →ₗ[ℝ] (m → ℝ))
    (z C : d → ℝ) (gamma Z : ℝ) (b : m → ℝ)
    (pairing : (m → ℝ) → (m → ℝ) → ℝ)
    (scoreMap : (m → ℝ) → (d → ℝ))
    (gradF : (m → ℝ) → (m → ℝ))
    (feasible : (d → ℝ) → Prop)
    (primalObjective : (d → ℝ) → ℝ)
    (uStar : m → ℝ)
    (hZ : Z = ∑ i : d, z i)
    (hz_pos : ∀ i : d, 0 < z i)
    (hgamma_pos : 0 < gamma)
    (hweak :
      ∀ (x : d → ℝ) (u : m → ℝ),
        feasible x →
          dualObjective_from_zC z C gamma Z b pairing scoreMap u ≤ primalObjective x)
    (hzero_feasible :
      feasible (primalFromDualScore z C gamma (scoreMap uStar)))
    (hzero_gap :
      primalObjective (primalFromDualScore z C gamma (scoreMap uStar)) =
        dualObjective_from_zC z C gamma Z b pairing scoreMap uStar)
    (hunique_value :
      ∀ x : d → ℝ,
        feasible x →
          primalObjective x =
            primalObjective (primalFromDualScore z C gamma (scoreMap uStar)) →
          x = primalFromDualScore z C gamma (scoreMap uStar))
    (hgrad :
      gradF uStar =
        fun j => b j - A (primalFromDualScore z C gamma (scoreMap uStar)) j) :
    let xStar := primalFromDualScore z C gamma (scoreMap uStar)
    Z = ∑ i : d, z i ∧
      (∀ i : d, 0 < z i) ∧
      0 < gamma ∧
      primalObjective xStar =
        dualObjective_from_zC z C gamma Z b pairing scoreMap uStar ∧
      IsUniquePrimalMinimizer feasible primalObjective xStar ∧
      IsDualMaximizer (dualObjective_from_zC z C gamma Z b pairing scoreMap) uStar ∧
      xStar = (fun i => z i * Real.exp ((scoreMap uStar i - C i) / gamma)) ∧
      (gradF uStar = 0 ↔ A xStar = b) ∧
      dualObjective_from_zC z C gamma Z b pairing scoreMap uStar =
        dualObjective_from_kernel (tiltedKernel z C gamma) gamma Z b pairing scoreMap uStar := by
  let xStar := primalFromDualScore z C gamma (scoreMap uStar)
  have hzero_feasible' : feasible xStar := by
    simpa [xStar] using hzero_feasible
  have hzero_gap' :
      primalObjective xStar =
        dualObjective_from_zC z C gamma Z b pairing scoreMap uStar := by
    simpa [xStar] using hzero_gap
  have hxMin : IsPrimalMinimizer feasible primalObjective xStar := by
    refine ⟨hzero_feasible', ?_⟩
    intro y hy
    have hwd := hweak y uStar hy
    linarith
  have hxUnique : IsUniquePrimalMinimizer feasible primalObjective xStar := by
    refine ⟨hxMin, ?_⟩
    intro y hyMin
    have hy_feasible : feasible y := hyMin.1
    have hy_le : primalObjective y ≤ primalObjective xStar := hyMin.2 xStar hzero_feasible'
    have hx_le : primalObjective xStar ≤ primalObjective y := hxMin.2 y hy_feasible
    have hvalue_eq : primalObjective y = primalObjective xStar := le_antisymm hy_le hx_le
    exact hunique_value y hy_feasible (by simpa [xStar] using hvalue_eq)
  have huMax : IsDualMaximizer (dualObjective_from_zC z C gamma Z b pairing scoreMap) uStar := by
    intro v
    have hwd := hweak xStar v hzero_feasible'
    linarith
  have hxFormula :
      xStar = (fun i => z i * Real.exp ((scoreMap uStar i - C i) / gamma)) := by
    rfl
  have hstat :
      gradF uStar = 0 ↔ A xStar = b := by
    simpa [xStar] using stationarity_iff_constraints A z C gamma scoreMap b gradF uStar hgrad
  refine ⟨hZ, hz_pos, hgamma_pos, hzero_gap', hxUnique, huMax, hxFormula, hstat, ?_⟩
  exact dualObjective_form_eq z C gamma Z b pairing scoreMap uStar

/--
Paper-facing structured-certificate version of Proposition
`prop:dual-gamma-correct`.

Compared with `dualGammaCorrect_primalDualCertificate`, this endpoint exposes a
single named certificate object instead of a long list of raw assumptions.  The
proof below unfolds that certificate and reuses the certified primal-dual
argument, so the challenge boundary records exactly which KKT/Fenchel facts
remain to be instantiated for the concrete KL projection problem.
-/
theorem dualGammaCorrect_from_primalDualCertificate
    [Fintype d]
    (A : (d → ℝ) →ₗ[ℝ] (m → ℝ))
    (z C : d → ℝ) (gamma Z : ℝ) (b : m → ℝ)
    (pairing : (m → ℝ) → (m → ℝ) → ℝ)
    (scoreMap : (m → ℝ) → (d → ℝ))
    (gradF : (m → ℝ) → (m → ℝ))
    (feasible : (d → ℝ) → Prop)
    (primalObjective : (d → ℝ) → ℝ)
    (uStar : m → ℝ)
    (hcert :
      DualGammaPrimalDualCertificate A z C gamma Z b pairing scoreMap gradF
        feasible primalObjective uStar) :
    let xStar := primalFromDualScore z C gamma (scoreMap uStar)
    Z = ∑ i : d, z i ∧
      (∀ i : d, 0 < z i) ∧
      0 < gamma ∧
      primalObjective xStar =
        dualObjective_from_zC z C gamma Z b pairing scoreMap uStar ∧
      IsUniquePrimalMinimizer feasible primalObjective xStar ∧
      IsDualMaximizer (dualObjective_from_zC z C gamma Z b pairing scoreMap) uStar ∧
      xStar = (fun i => z i * Real.exp ((scoreMap uStar i - C i) / gamma)) ∧
      (gradF uStar = 0 ↔ A xStar = b) ∧
      dualObjective_from_zC z C gamma Z b pairing scoreMap uStar =
        dualObjective_from_kernel (tiltedKernel z C gamma) gamma Z b pairing scoreMap uStar := by
  exact dualGammaCorrect_primalDualCertificate
    A z C gamma Z b pairing scoreMap gradF feasible primalObjective uStar
    hcert.mass_normalization
    hcert.reference_positive
    hcert.gamma_positive
    hcert.weak_duality
    hcert.primal_from_dual_feasible
    hcert.zero_gap
    hcert.unique_by_value
    hcert.gradient_identity

end

end Section2Duality
end KLProjection
end FlowSinkhorn
