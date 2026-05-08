import FlowSinkhorn.KLProjection.PrimalDualBounds.Blueprint
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# `H_γ` for balanced optimal transport

This module is reserved for Proposition `prop:Hgamma-ot` from
`papers/kl-projections/sections/sec-sinkhorn.tex`.

Intended theorem names:
- `ot_HGamma_bound`;
- `ot_dualPotential_bound`.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace Applications
namespace OT

variable {𝕜 E : Type*}
variable [NormedField 𝕜] [AddCommGroup E] [Module 𝕜 E]

/--
Paper-facing `H_γ` monotonicity package for balanced OT.

If an application proves `H_γ ≤ H̄_γ`, this theorem turns it into the corresponding bound on
the canonical orbit budget `κ (cost + γ H_γ)`.
-/
theorem ot_HGamma_bound
    {kappa cost gamma hGamma hGammaUpper : ℝ}
    (hkappa : 0 ≤ kappa)
    (hgamma : 0 ≤ gamma)
    (hH : hGamma ≤ hGammaUpper) :
    PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGammaUpper := by
  dsimp [PrimalDualBounds.hGammaKappaBudget]
  have hmul : gamma * hGamma ≤ gamma * hGammaUpper :=
    mul_le_mul_of_nonneg_left hH hgamma
  linarith [mul_le_mul_of_nonneg_left hmul hkappa]

/--
Paper-facing dual-potential bound for balanced OT.

This is the OT-namespaced endpoint for the fixed-point control assumption
`p u⋆ ≤ κ (cost + γ H_γ)`.
-/
theorem ot_dualPotential_bound
    (p : Seminorm 𝕜 E) {uStar : E}
    {kappa cost gamma hGamma : ℝ}
    (hbound : p uStar ≤ PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma) :
    p uStar ≤ PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma :=
  PrimalDualBounds.fixedPointBound_of_HGamma_kappa p hbound

/--
Transport a fixed-point estimate through an OT `H_γ` upper bound.

This is the compositional step used in applications: first control `p u⋆` with
`κ (cost + γ H_γ)`, then replace `H_γ` by any certified upper bound.
-/
theorem ot_dualPotential_bound_of_HGamma_upper
    (p : Seminorm 𝕜 E) {uStar : E}
    {kappa cost gamma hGamma hGammaUpper : ℝ}
    (hkappa : 0 ≤ kappa)
    (hgamma : 0 ≤ gamma)
    (hH : hGamma ≤ hGammaUpper)
    (hbound : p uStar ≤ PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma) :
    p uStar ≤ PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGammaUpper := by
  exact le_trans hbound (ot_HGamma_bound hkappa hgamma hH)

/--
OT iterate-orbit bound after substituting an `H_γ` upper bound in the budget.

This packages the common chain
`fixed-point control` + `H_γ` upper estimate + `non-expansive dynamics`.
-/
theorem ot_uniformIterateBound_of_HGamma_upper
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma hGammaUpper : ℝ}
    (hkappa : 0 ≤ kappa)
    (hgamma : 0 ≤ gamma)
    (hH : hGamma ≤ hGammaUpper)
    (hbound : p uStar ≤ PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤
      p u0 + 2 * PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGammaUpper := by
  exact PrimalDualBounds.uniformIterateBound_of_nonexpansive_of_HGamma_kappa
    p Psi hPsi hfix
    (ot_dualPotential_bound_of_HGamma_upper p hkappa hgamma hH hbound) k

/--
Non-negativity of the explicit OT `H_γ` formula from Proposition `prop:Hgamma-ot`.

For regularization `gamma > 0`, cost supremum `C_max ≥ 0`, and minimum marginal mass `min_b > 0`,
the formula `|log(min_b)| + 2 * C_max / gamma` is nonneg.
-/
theorem ot_HGamma_formula_nonneg
    {min_b C_max gamma : ℝ}
    (hgamma : 0 < gamma)
    (hC_max : 0 ≤ C_max)
    (_ : 0 < min_b) :
    0 ≤ |Real.log min_b| + 2 * C_max / gamma := by
  apply add_nonneg
  · exact abs_nonneg _
  · positivity

/--
Explicit OT `H_γ/κ` budget evaluation from Proposition `prop:Hgamma-ot`.

With `κ = 1` and `cost = C_max`, the budget formula
`κ * (cost + γ * H_γ)` evaluates at `H_γ = |log(min_b)| + 2 * C_max / γ` to
`C_max + γ * |log(min_b)| + 2 * C_max`.

The corresponding `U_max = 2 * budget = 6 * C_max + 2 * γ * |log(min_b)|`
(since `2 * (C_max + γ * |log(min_b)| + 2 * C_max) = 6 * C_max + 2γ * |log(min_b)|`).
NOTE: Corollary `cor:OT-XmaxUmax` in the paper originally stated `4 * C_max`; the correct value
is `6 * C_max`, as verified by this formalization (corrected in the paper).
-/
theorem ot_hGammaBudget_explicit_formula
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma)
    (_ : 0 < min_b)
    (_ : 0 ≤ C_max) :
    PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) =
      C_max + gamma * |Real.log min_b| + 2 * C_max := by
  have hg : gamma ≠ 0 := ne_of_gt hgamma
  simp only [PrimalDualBounds.hGammaKappaBudget, one_mul]
  field_simp [hg]
  ring

/--
Twice the explicit OT budget evaluates to `6 * C_max + 2 * γ * |log(min_b)|`.

This is the `U_max` constant from Corollary `cor:OT-XmaxUmax`.
-/
theorem ot_Umax_twoTimesHGammaBudget
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma)
    (hmin_b : 0 < min_b)
    (hC_max : 0 ≤ C_max) :
    2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) =
      6 * C_max + 2 * gamma * |Real.log min_b| := by
  rw [ot_hGammaBudget_explicit_formula hgamma hmin_b hC_max]
  ring

/--
Non-negativity of the explicit OT budget.

With `γ > 0`, `min_b > 0`, `C_max ≥ 0`, the canonical budget
`hGammaKappaBudget 1 C_max γ (|log(min_b)| + 2 * C_max / γ)` is non-negative.
-/
theorem ot_hGammaKappaBudget_nonneg
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max) :
    0 ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) := by
  rw [ot_hGammaBudget_explicit_formula hgamma hmin_b hC_max]
  have hab : 0 ≤ |Real.log min_b| := abs_nonneg _
  nlinarith

/--
Normalized explicit OT budget constant.

This re-expresses the explicit budget in the compact form
`3 * C_max + γ * |log(min_b)|`.
-/
theorem ot_hGammaBudget_explicit_formula_threeCmax
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma)
    (hmin_b : 0 < min_b)
    (hC_max : 0 ≤ C_max) :
    PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) =
      3 * C_max + gamma * |Real.log min_b| := by
  rw [ot_hGammaBudget_explicit_formula hgamma hmin_b hC_max]
  ring

/--
Concrete fixed-point bridge using the explicit OT `H_γ` formula.

This removes the abstract `hGammaKappaBudget` term and exposes the closed form
`C_max + γ * |log(min_b)| + 2 * C_max`, directly usable in complexity constants.
-/
theorem ot_fixedPointBound_explicit_of_HGamma_formula
    (p : Seminorm 𝕜 E) {uStar : E}
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbound : p uStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma)) :
    p uStar ≤ C_max + gamma * |Real.log min_b| + 2 * C_max := by
  rw [ot_hGammaBudget_explicit_formula hgamma hmin_b hC_max] at hbound
  exact hbound

/--
Concrete fixed-point bridge in normalized constant form.

This is the fixed-point estimate written with `3 * C_max + γ * |log(min_b)|`.
-/
theorem ot_fixedPointBound_explicit_threeCmax
    (p : Seminorm 𝕜 E) {uStar : E}
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbound : p uStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma)) :
    p uStar ≤ 3 * C_max + gamma * |Real.log min_b| := by
  rw [ot_hGammaBudget_explicit_formula_threeCmax hgamma hmin_b hC_max] at hbound
  exact hbound

/--
Concrete iterate-orbit bridge using the explicit OT `H_γ` formula.

From non-expansive dynamics and a fixed-point bound at the explicit OT budget,
this yields the iterate bound with the closed-form constant
`6 * C_max + 2 * γ * |log(min_b)|`.
-/
theorem ot_uniformIterateBound_explicit_of_HGamma_formula
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbound : p uStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma))
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ p u0 + (6 * C_max + 2 * gamma * |Real.log min_b|) := by
  have hiter := seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
    p Psi hPsi (uStar := uStar) (u0 := u0) hfix hbound k
  have hU := ot_Umax_twoTimesHGammaBudget hgamma hmin_b hC_max
  calc
    p ((Psi^[k]) u0)
      ≤ p u0 + 2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
          (|Real.log min_b| + 2 * C_max / gamma) := hiter
    _ = p u0 + (6 * C_max + 2 * gamma * |Real.log min_b|) := by rw [hU]

/--
Explicit iterate-orbit bound under an upper envelope on the OT closed-form constant.

This is convenient for complexity pipelines that maintain a coarse certified constant `U`
such that `6 * C_max + 2 * gamma * |log(min_b)| ≤ U`.
-/
theorem ot_uniformIterateBound_explicit_of_HGamma_formula_of_upperConstant
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {gamma min_b C_max U : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbound : p uStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma))
    (hU : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ U)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ p u0 + U := by
  have hiter := ot_uniformIterateBound_explicit_of_HGamma_formula
    p Psi hPsi (uStar := uStar) (u0 := u0) hfix
    hgamma hmin_b hC_max hbound k
  have hshift :
      p u0 + (6 * C_max + 2 * gamma * |Real.log min_b|) ≤ p u0 + U := by
    simpa [add_comm, add_left_comm, add_assoc] using
      add_le_add_left hU (p u0)
  exact hiter.trans hshift

/--
Concrete iterate-orbit bridge in normalized constant form.

This is the same explicit iterate control written as
`p u0 + 2 * (3 * C_max + γ * |log(min_b)|)`.
-/
theorem ot_uniformIterateBound_explicit_threeCmax
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbound : p uStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma))
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ p u0 + 2 * (3 * C_max + gamma * |Real.log min_b|) := by
  have hiter := ot_uniformIterateBound_explicit_of_HGamma_formula
    p Psi hPsi (uStar := uStar) (u0 := u0) hfix
    hgamma hmin_b hC_max hbound k
  calc
    p ((Psi^[k]) u0) ≤ p u0 + (6 * C_max + 2 * gamma * |Real.log min_b|) := hiter
    _ = p u0 + 2 * (3 * C_max + gamma * |Real.log min_b|) := by ring

/--
Normalized explicit iterate-orbit bound under an upper envelope on the threeCmax constant.
-/
theorem ot_uniformIterateBound_explicit_threeCmax_of_upperConstant
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {gamma min_b C_max U : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbound : p uStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma))
    (hU : 2 * (3 * C_max + gamma * |Real.log min_b|) ≤ U)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ p u0 + U := by
  have hiter := ot_uniformIterateBound_explicit_threeCmax
    p Psi hPsi (uStar := uStar) (u0 := u0) hfix
    hgamma hmin_b hC_max hbound k
  have hshift :
      p u0 + 2 * (3 * C_max + gamma * |Real.log min_b|) ≤ p u0 + U := by
    simpa [add_comm, add_left_comm, add_assoc] using
      add_le_add_left hU (p u0)
  exact hiter.trans hshift

/--
Explicit iterate-orbit bound with a bounded initial radius.

This is a complexity-friendly corollary: replace `p u0` by any certified upper bound `U0`.
-/
theorem ot_uniformIterateBound_explicit_of_HGamma_formula_of_baseBound
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {gamma min_b C_max U0 : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbound : p uStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma))
    (hU0 : p u0 ≤ U0)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ U0 + (6 * C_max + 2 * gamma * |Real.log min_b|) := by
  have hiter := ot_uniformIterateBound_explicit_of_HGamma_formula
    p Psi hPsi (uStar := uStar) (u0 := u0) hfix
    hgamma hmin_b hC_max hbound k
  have hbase :
      p u0 + (6 * C_max + 2 * gamma * |Real.log min_b|) ≤
        U0 + (6 * C_max + 2 * gamma * |Real.log min_b|) := by
    simpa [add_comm, add_left_comm, add_assoc] using
      add_le_add_right hU0 (6 * C_max + 2 * gamma * |Real.log min_b|)
  exact hiter.trans hbase

/--
Explicit iterate-orbit bound with bounded initial radius at successor index.

This is the `(k+1)` convenience form of
`ot_uniformIterateBound_explicit_of_HGamma_formula_of_baseBound`.
-/
theorem ot_uniformIterateBound_explicit_of_HGamma_formula_of_baseBound_succ
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {gamma min_b C_max U0 : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbound : p uStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma))
    (hU0 : p u0 ≤ U0)
    (k : ℕ) :
    p ((Psi^[k + 1]) u0) ≤ U0 + (6 * C_max + 2 * gamma * |Real.log min_b|) := by
  simpa [Nat.succ_eq_add_one] using
    ot_uniformIterateBound_explicit_of_HGamma_formula_of_baseBound
      p Psi hPsi (uStar := uStar) (u0 := u0) hfix
      hgamma hmin_b hC_max hbound hU0 (k + 1)

/--
Explicit iterate-orbit bound under a nonpositive initial seminorm.

This is a convenient corollary of
`ot_uniformIterateBound_explicit_of_HGamma_formula_of_baseBound` with `U0 = 0`.
-/
theorem ot_uniformIterateBound_explicit_of_HGamma_formula_of_nonpos_base
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbound : p uStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma))
    (hU0 : p u0 ≤ 0)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ 6 * C_max + 2 * gamma * |Real.log min_b| := by
  have hiter := ot_uniformIterateBound_explicit_of_HGamma_formula_of_baseBound
    p Psi hPsi (uStar := uStar) (u0 := u0) hfix
    (U0 := 0) hgamma hmin_b hC_max hbound hU0 k
  simpa [zero_add] using hiter

/--
Explicit iterate-orbit bound from a zero seminorm seed.

This is the direct specialization of
`ot_uniformIterateBound_explicit_of_HGamma_formula` to `p u0 = 0`.
-/
theorem ot_uniformIterateBound_explicit_zero_base
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbound : p uStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma))
    (hzero : p u0 = 0)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ 6 * C_max + 2 * gamma * |Real.log min_b| := by
  have hiter := ot_uniformIterateBound_explicit_of_HGamma_formula
    p Psi hPsi (uStar := uStar) (u0 := u0) hfix
    hgamma hmin_b hC_max hbound k
  rw [hzero, zero_add] at hiter
  exact hiter

/--
Zero-seed explicit iterate-orbit bound under an upper envelope on the closed-form constant.
-/
theorem ot_uniformIterateBound_explicit_zero_base_of_upperConstant
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {gamma min_b C_max U : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbound : p uStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma))
    (hzero : p u0 = 0)
    (hU : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ U)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ U := by
  have hiter := ot_uniformIterateBound_explicit_of_HGamma_formula_of_upperConstant
    p Psi hPsi (uStar := uStar) (u0 := u0) hfix
    hgamma hmin_b hC_max hbound hU k
  rw [hzero, zero_add] at hiter
  exact hiter

/--
Zero-seed explicit iterate-orbit bound in normalized constant form.

This rewrites the zero-base explicit constant as
`2 * (3 * C_max + gamma * |log(min_b)|)`.
-/
theorem ot_uniformIterateBound_explicit_zero_base_threeCmax
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbound : p uStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma))
    (hzero : p u0 = 0)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ 2 * (3 * C_max + gamma * |Real.log min_b|) := by
  have hiter := ot_uniformIterateBound_explicit_zero_base
    p Psi hPsi (uStar := uStar) (u0 := u0) hfix
    hgamma hmin_b hC_max hbound hzero k
  calc
    p ((Psi^[k]) u0) ≤ 6 * C_max + 2 * gamma * |Real.log min_b| := hiter
    _ = 2 * (3 * C_max + gamma * |Real.log min_b|) := by ring

/--
Zero-seed explicit iterate-orbit bound in normalized form at successor index.

This is the `(k+1)` convenience form of
`ot_uniformIterateBound_explicit_zero_base_threeCmax`.
-/
theorem ot_uniformIterateBound_explicit_zero_base_threeCmax_succ
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbound : p uStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma))
    (hzero : p u0 = 0)
    (k : ℕ) :
    p ((Psi^[k + 1]) u0) ≤ 2 * (3 * C_max + gamma * |Real.log min_b|) := by
  simpa [Nat.succ_eq_add_one] using
    ot_uniformIterateBound_explicit_zero_base_threeCmax
      p Psi hPsi (uStar := uStar) (u0 := u0) hfix
      hgamma hmin_b hC_max hbound hzero (k + 1)

/--
Zero-seed explicit iterate-orbit bound at successor index.
-/
theorem ot_uniformIterateBound_explicit_zero_base_succ
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbound : p uStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma))
    (hzero : p u0 = 0)
    (k : ℕ) :
    p ((Psi^[k + 1]) u0) ≤ 6 * C_max + 2 * gamma * |Real.log min_b| := by
  simpa [Nat.succ_eq_add_one] using
    ot_uniformIterateBound_explicit_zero_base
      p Psi hPsi (uStar := uStar) (u0 := u0) hfix
      hgamma hmin_b hC_max hbound hzero (k + 1)

/--
Successor-index zero-seed explicit iterate bound under an upper envelope constant.
-/
theorem ot_uniformIterateBound_explicit_zero_base_succ_of_upperConstant
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {gamma min_b C_max U : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbound : p uStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma))
    (hzero : p u0 = 0)
    (hU : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ U)
    (k : ℕ) :
    p ((Psi^[k + 1]) u0) ≤ U := by
  simpa [Nat.succ_eq_add_one] using
    ot_uniformIterateBound_explicit_zero_base_of_upperConstant
      p Psi hPsi (uStar := uStar) (u0 := u0) hfix
      hgamma hmin_b hC_max hbound hzero hU (k + 1)

/--
Normalized-constant version of the bounded-base explicit iterate bound.
-/
theorem ot_uniformIterateBound_explicit_threeCmax_of_baseBound
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {gamma min_b C_max U0 : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbound : p uStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma))
    (hU0 : p u0 ≤ U0)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ U0 + 2 * (3 * C_max + gamma * |Real.log min_b|) := by
  have hiter := ot_uniformIterateBound_explicit_threeCmax
    p Psi hPsi (uStar := uStar) (u0 := u0) hfix
    hgamma hmin_b hC_max hbound k
  have hbase :
      p u0 + 2 * (3 * C_max + gamma * |Real.log min_b|) ≤
        U0 + 2 * (3 * C_max + gamma * |Real.log min_b|) := by
    simpa [add_comm, add_left_comm, add_assoc] using
      add_le_add_right hU0 (2 * (3 * C_max + gamma * |Real.log min_b|))
  exact hiter.trans hbase

/--
Normalized-constant bounded-base explicit iterate bound at successor index.

This is the `(k+1)` convenience form of
`ot_uniformIterateBound_explicit_threeCmax_of_baseBound`.
-/
theorem ot_uniformIterateBound_explicit_threeCmax_of_baseBound_succ
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {gamma min_b C_max U0 : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbound : p uStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma))
    (hU0 : p u0 ≤ U0)
    (k : ℕ) :
    p ((Psi^[k + 1]) u0) ≤ U0 + 2 * (3 * C_max + gamma * |Real.log min_b|) := by
  simpa [Nat.succ_eq_add_one] using
    ot_uniformIterateBound_explicit_threeCmax_of_baseBound
      p Psi hPsi (uStar := uStar) (u0 := u0) hfix
      hgamma hmin_b hC_max hbound hU0 (k + 1)

/--
Normalized explicit iterate-orbit bound under a nonpositive initial seminorm.

This is the normalized-constant counterpart of
`ot_uniformIterateBound_explicit_of_HGamma_formula_of_nonpos_base`.
-/
theorem ot_uniformIterateBound_explicit_threeCmax_of_nonpos_base
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbound : p uStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma))
    (hU0 : p u0 ≤ 0)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ 2 * (3 * C_max + gamma * |Real.log min_b|) := by
  have hiter := ot_uniformIterateBound_explicit_threeCmax_of_baseBound
    p Psi hPsi (uStar := uStar) (u0 := u0) hfix
    (U0 := 0) hgamma hmin_b hC_max hbound hU0 k
  simpa [zero_add] using hiter

/--
Uniform iterate bound starting from a zero-variation seed.

If `Psi` is non-expansive for `variationSeminormAsSeminorm`, `alphaStar` is a fixed point
with `variationSeminorm alphaStar ≤ budget`, and `u0` has `variationSeminorm u0 = 0`, then
every iterate satisfies `variationSeminorm ((Psi^[k]) u0) ≤ 6 * C_max + 2 * γ * |log(min_b)|`.
-/
theorem ot_uniformIterateBound_from_zero
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbound : variationSeminorm alphaStar ≤
      PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (hu0 : variationSeminorm u0 = 0) (k : ℕ) :
    variationSeminorm ((Psi^[k]) u0) ≤ 6 * C_max + 2 * gamma * |Real.log min_b| := by
  have hiter := seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
    variationSeminormAsSeminorm Psi hPsi (uStar := alphaStar) (u0 := u0) hfix hbound k
  have hU := ot_Umax_twoTimesHGammaBudget hgamma hmin_b hC_max
  -- `variationSeminormAsSeminorm` is `Seminorm.of variationSeminorm ...`
  -- so the coercion is definitionally `variationSeminorm`.
  have hiter' : variationSeminorm ((Psi^[k]) u0) ≤
      variationSeminorm u0 +
        2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
          (|Real.log min_b| + 2 * C_max / gamma) := hiter
  linarith [hu0 ▸ hiter']

/--
Zero-seed iterate control in normalized constant form.

Under the same assumptions as `ot_uniformIterateBound_from_zero`, the iterate bound is
`2 * (3 * C_max + γ * |log(min_b)|)`.
-/
theorem ot_uniformIterateBound_from_zero_threeCmax
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbound : variationSeminorm alphaStar ≤
      PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (hu0 : variationSeminorm u0 = 0) (k : ℕ) :
    variationSeminorm ((Psi^[k]) u0) ≤
      2 * (3 * C_max + gamma * |Real.log min_b|) := by
  have hiter := ot_uniformIterateBound_from_zero
    Psi hPsi hfix hgamma hmin_b hC_max hbound hu0 k
  calc
    variationSeminorm ((Psi^[k]) u0) ≤ 6 * C_max + 2 * gamma * |Real.log min_b| := hiter
    _ = 2 * (3 * C_max + gamma * |Real.log min_b|) := by ring

/--
Zero-seed iterate control in normalized constant form at successor index.

This is the `(k+1)`-iterate form of `ot_uniformIterateBound_from_zero_threeCmax`.
-/
theorem ot_uniformIterateBound_from_zero_threeCmax_succ
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbound : variationSeminorm alphaStar ≤
      PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (hu0 : variationSeminorm u0 = 0) (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) u0) ≤
      2 * (3 * C_max + gamma * |Real.log min_b|) := by
  simpa [Nat.succ_eq_add_one] using
    ot_uniformIterateBound_from_zero_threeCmax
      Psi hPsi hfix hgamma hmin_b hC_max hbound hu0 (k + 1)

/--
Zero-seed iterate control at successor index.

This is the `(k+1)`-iterate form of `ot_uniformIterateBound_from_zero`.
-/
theorem ot_uniformIterateBound_from_zero_succ
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbound : variationSeminorm alphaStar ≤
      PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (hu0 : variationSeminorm u0 = 0) (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) u0) ≤ 6 * C_max + 2 * gamma * |Real.log min_b| := by
  simpa [Nat.succ_eq_add_one] using
    ot_uniformIterateBound_from_zero
      Psi hPsi hfix hgamma hmin_b hC_max hbound hu0 (k + 1)

/--
Explicit fixed-point-to-orbit bridge in normalized constant form.

If an application has already derived the concrete fixed-point estimate
`variationSeminorm alphaStar ≤ 3 * C_max + gamma * |log(min_b)|`, this theorem turns it into
the iterate-orbit budget with arbitrary base radius.
-/
theorem ot_uniformIterateBound_from_explicitFixedPoint_threeCmax
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) u0) ≤
      variationSeminorm u0 + 2 * (3 * C_max + gamma * |Real.log min_b|) := by
  simpa using
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
      variationSeminormAsSeminorm Psi hPsi (uStar := alphaStar) (u0 := u0) hfix hfixed k

/--
Zero-seed explicit fixed-point-to-orbit bridge in normalized constant form.

This is the `variationSeminorm u0 = 0` specialization of
`ot_uniformIterateBound_from_explicitFixedPoint_threeCmax`.
-/
theorem ot_uniformIterateBound_from_zero_of_explicitFixedPoint_threeCmax
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hu0 : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) u0) ≤ 2 * (3 * C_max + gamma * |Real.log min_b|) := by
  have hiter := ot_uniformIterateBound_from_explicitFixedPoint_threeCmax
    Psi hPsi (u0 := u0) hfix hfixed k
  rw [hu0, zero_add] at hiter
  exact hiter

/--
Zero-seed explicit fixed-point-to-orbit bridge at successor index.

This is the `(k+1)` form of
`ot_uniformIterateBound_from_zero_of_explicitFixedPoint_threeCmax`.
-/
theorem ot_uniformIterateBound_from_zero_of_explicitFixedPoint_threeCmax_succ
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hu0 : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) u0) ≤ 2 * (3 * C_max + gamma * |Real.log min_b|) := by
  simpa [Nat.succ_eq_add_one] using
    ot_uniformIterateBound_from_zero_of_explicitFixedPoint_threeCmax
      Psi hPsi hfix hfixed hu0 (k + 1)

/--
Zero-seed explicit fixed-point-to-orbit bridge under an upper-envelope constant.

This corollary is tailored for complexity statements that track a certified
coarse constant `U` with `2 * (3 * C_max + gamma * |log(min_b)|) ≤ U`.
-/
theorem ot_uniformIterateBound_from_zero_of_explicitFixedPoint_threeCmax_of_upperConstant
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max U : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hu0 : variationSeminorm u0 = 0)
    (hU : 2 * (3 * C_max + gamma * |Real.log min_b|) ≤ U)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) u0) ≤ U := by
  have hiter := ot_uniformIterateBound_from_zero_of_explicitFixedPoint_threeCmax
    Psi hPsi hfix hfixed hu0 k
  exact hiter.trans hU

/--
Explicit fixed-point-to-orbit bridge with a bounded initial radius.

This is the bounded-base counterpart of
`ot_uniformIterateBound_from_explicitFixedPoint_threeCmax`.
-/
theorem ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_of_baseBound
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max U0 : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU0 : variationSeminorm u0 ≤ U0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) u0) ≤
      U0 + 2 * (3 * C_max + gamma * |Real.log min_b|) := by
  have hiter := ot_uniformIterateBound_from_explicitFixedPoint_threeCmax
    Psi hPsi (u0 := u0) hfix hfixed k
  have hbase :
      variationSeminorm u0 + 2 * (3 * C_max + gamma * |Real.log min_b|) ≤
        U0 + 2 * (3 * C_max + gamma * |Real.log min_b|) := by
    simpa [add_comm, add_left_comm, add_assoc] using
      add_le_add_right hU0 (2 * (3 * C_max + gamma * |Real.log min_b|))
  exact hiter.trans hbase

/--
Bounded-base explicit fixed-point-to-orbit bridge at successor index.

This is the `(k+1)` form of
`ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_of_baseBound`.
-/
theorem ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_of_baseBound_succ
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max U0 : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU0 : variationSeminorm u0 ≤ U0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) u0) ≤
      U0 + 2 * (3 * C_max + gamma * |Real.log min_b|) := by
  simpa [Nat.succ_eq_add_one] using
    ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_of_baseBound
      Psi hPsi hfix hfixed hU0 (k + 1)

/--
Explicit fixed-point-to-orbit bridge under an upper-envelope constant.

This upgrades the explicit threeCmax iterate increment to any `U` such that
`2 * (3 * C_max + gamma * |log(min_b)|) ≤ U`.
-/
theorem ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_of_upperConstant
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max U : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 2 * (3 * C_max + gamma * |Real.log min_b|) ≤ U)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) u0) ≤ variationSeminorm u0 + U := by
  have hiter := ot_uniformIterateBound_from_explicitFixedPoint_threeCmax
    Psi hPsi (u0 := u0) hfix hfixed k
  have hshift :
      variationSeminorm u0 + 2 * (3 * C_max + gamma * |Real.log min_b|) ≤
        variationSeminorm u0 + U := by
    simpa [add_comm, add_left_comm, add_assoc] using
      add_le_add_left hU (variationSeminorm u0)
  exact hiter.trans hshift

/--
Successor-index zero-seed explicit fixed-point-to-orbit bridge under an upper-envelope constant.
-/
theorem ot_uniformIterateBound_from_zero_of_explicitFixedPoint_threeCmax_of_upperConstant_succ
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max U : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hu0 : variationSeminorm u0 = 0)
    (hU : 2 * (3 * C_max + gamma * |Real.log min_b|) ≤ U)
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) u0) ≤ U := by
  simpa [Nat.succ_eq_add_one] using
    ot_uniformIterateBound_from_zero_of_explicitFixedPoint_threeCmax_of_upperConstant
      Psi hPsi hfix hfixed hu0 hU (k + 1)

/--
Successor-index explicit fixed-point-to-orbit bridge under an upper-envelope constant.
-/
theorem ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_upperConstant_succ
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max U : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 2 * (3 * C_max + gamma * |Real.log min_b|) ≤ U)
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) u0) ≤ variationSeminorm u0 + U := by
  simpa [Nat.succ_eq_add_one] using
    ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_of_upperConstant
      Psi hPsi hfix hfixed hU (k + 1)

/--
Bounded-base explicit fixed-point-to-orbit bridge under an upper-envelope constant.
-/
theorem ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_upperConstant_baseBound
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max U U0 : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 2 * (3 * C_max + gamma * |Real.log min_b|) ≤ U)
    (hU0 : variationSeminorm u0 ≤ U0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) u0) ≤ U0 + U := by
  have hiter := ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_of_upperConstant
    Psi hPsi (u0 := u0) hfix hfixed hU k
  have hbase : variationSeminorm u0 + U ≤ U0 + U := by
    simpa [add_comm, add_left_comm, add_assoc] using add_le_add_right hU0 U
  exact hiter.trans hbase

/--
Successor-index bounded-base explicit fixed-point-to-orbit bridge under an upper-envelope constant.
-/
theorem ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_upperConstant_baseBound_succ
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max U U0 : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 2 * (3 * C_max + gamma * |Real.log min_b|) ≤ U)
    (hU0 : variationSeminorm u0 ≤ U0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) u0) ≤ U0 + U := by
  simpa [Nat.succ_eq_add_one] using
    ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_upperConstant_baseBound
      Psi hPsi hfix hfixed hU hU0 (k + 1)

/--
Nonpositive-base explicit fixed-point-to-orbit bridge under an upper-envelope constant.
-/
theorem ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_upperConstant_nonpos_base
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max U : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 2 * (3 * C_max + gamma * |Real.log min_b|) ≤ U)
    (hU0 : variationSeminorm u0 ≤ 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) u0) ≤ U := by
  have hiter := ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_upperConstant_baseBound
    Psi hPsi (u0 := u0) hfix (U0 := 0) hfixed hU hU0 k
  simpa [zero_add] using hiter

/--
Index-`0` nonpositive-base explicit fixed-point-to-orbit bridge
under an upper-envelope constant.
-/
theorem ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_upperConstant_nonpos_base_zeroIndex
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max U : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 2 * (3 * C_max + gamma * |Real.log min_b|) ≤ U)
    (hU0 : variationSeminorm u0 ≤ 0) :
    variationSeminorm ((Psi^[0]) u0) ≤ U :=
  ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_upperConstant_nonpos_base
    (Psi := Psi) hPsi (u0 := u0) hfix hfixed hU hU0 0

/--
Index-`0` explicit fixed-point-to-orbit bridge under an upper-envelope constant.

This is the base-index specialization of
`ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_of_upperConstant`.
-/
theorem ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_of_upperConstant_zeroIndex
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max U : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 2 * (3 * C_max + gamma * |Real.log min_b|) ≤ U) :
    variationSeminorm ((Psi^[0]) u0) ≤ variationSeminorm u0 + U := by
  exact
    ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_of_upperConstant
      (Psi := Psi) hPsi (u0 := u0) hfix hfixed hU 0

/--
Ceiling-index explicit fixed-point-to-orbit bridge under an upper-envelope constant.
-/
theorem ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_of_upperConstant_ceilIndex
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max U : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 2 * (3 * C_max + gamma * |Real.log min_b|) ≤ U)
    (r : ℝ) :
    variationSeminorm ((Psi^[Nat.ceil r]) u0) ≤ variationSeminorm u0 + U :=
  ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_of_upperConstant
    (Psi := Psi) hPsi (u0 := u0) hfix hfixed hU (Nat.ceil r)

/--
Successor-index ceiling-index explicit fixed-point-to-orbit bridge
under an upper-envelope constant.
-/
theorem ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_of_upperConstant_ceilIndex_succ
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max U : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 2 * (3 * C_max + gamma * |Real.log min_b|) ≤ U)
    (r : ℝ) :
    variationSeminorm ((Psi^[Nat.ceil r + 1]) u0) ≤ variationSeminorm u0 + U := by
  simpa [Nat.succ_eq_add_one] using
    ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_upperConstant_succ
      (Psi := Psi) hPsi (u0 := u0) hfix hfixed hU (Nat.ceil r)

/--
Index-`0` bounded-base explicit fixed-point-to-orbit bridge
under an upper-envelope constant.

This is the base-index specialization of
`ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_upperConstant_baseBound`.
-/
theorem ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_upperConstant_baseBound_zeroIndex
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max U U0 : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 2 * (3 * C_max + gamma * |Real.log min_b|) ≤ U)
    (hU0 : variationSeminorm u0 ≤ U0) :
    variationSeminorm ((Psi^[0]) u0) ≤ U0 + U := by
  exact
    ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_upperConstant_baseBound
      (Psi := Psi) hPsi (u0 := u0) hfix hfixed hU hU0 0

/--
Successor-index nonpositive-base explicit fixed-point-to-orbit bridge
under an upper-envelope constant.

This is the `(k+1)` form of
`ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_upperConstant_nonpos_base`.
-/
theorem ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_upperConstant_nonpos_base_succ
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max U : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 2 * (3 * C_max + gamma * |Real.log min_b|) ≤ U)
    (hU0 : variationSeminorm u0 ≤ 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) u0) ≤ U := by
  simpa [Nat.succ_eq_add_one] using
    ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_upperConstant_nonpos_base
      (Psi := Psi) hPsi hfix hfixed hU hU0 (k + 1)

/--
Ceiling-index nonpositive-base explicit fixed-point-to-orbit bridge
under an upper-envelope constant.
-/
theorem ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_upperConstant_nonpos_base_ceilIndex
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max U : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 2 * (3 * C_max + gamma * |Real.log min_b|) ≤ U)
    (hU0 : variationSeminorm u0 ≤ 0)
    (r : ℝ) :
    variationSeminorm ((Psi^[Nat.ceil r]) u0) ≤ U :=
  ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_upperConstant_nonpos_base
    (Psi := Psi) hPsi (u0 := u0) hfix hfixed hU hU0 (Nat.ceil r)

/--
Index-`0` zero-seed explicit fixed-point-to-orbit bridge under an upper-envelope constant.
-/
theorem ot_uniformIterateBound_from_zero_of_explicitFixedPoint_threeCmax_of_upperConstant_zeroIndex
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max U : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hu0 : variationSeminorm u0 = 0)
    (hU : 2 * (3 * C_max + gamma * |Real.log min_b|) ≤ U) :
    variationSeminorm ((Psi^[0]) u0) ≤ U :=
  ot_uniformIterateBound_from_zero_of_explicitFixedPoint_threeCmax_of_upperConstant
    (Psi := Psi) hPsi (u0 := u0) hfix hfixed hu0 hU 0

/--
Ceiling-index zero-seed explicit fixed-point-to-orbit bridge under an upper-envelope constant.
-/
theorem ot_uniformIterateBound_from_zero_of_explicitFixedPoint_threeCmax_of_upperConstant_ceilIndex
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max U : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hu0 : variationSeminorm u0 = 0)
    (hU : 2 * (3 * C_max + gamma * |Real.log min_b|) ≤ U)
    (r : ℝ) :
    variationSeminorm ((Psi^[Nat.ceil r]) u0) ≤ U :=
  ot_uniformIterateBound_from_zero_of_explicitFixedPoint_threeCmax_of_upperConstant
    (Psi := Psi) hPsi (u0 := u0) hfix hfixed hu0 hU (Nat.ceil r)

/--
Successor-index ceiling-index zero-seed explicit fixed-point-to-orbit bridge
under an upper-envelope constant.
-/
theorem ot_uniformIterateBound_from_zero_of_explicitFixedPoint_threeCmax_of_upperConstant_ceilIndex_succ
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max U : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hu0 : variationSeminorm u0 = 0)
    (hU : 2 * (3 * C_max + gamma * |Real.log min_b|) ≤ U)
    (r : ℝ) :
    variationSeminorm ((Psi^[Nat.ceil r + 1]) u0) ≤ U := by
  simpa [Nat.succ_eq_add_one] using
    ot_uniformIterateBound_from_zero_of_explicitFixedPoint_threeCmax_of_upperConstant_succ
      (Psi := Psi) hPsi (u0 := u0) hfix hfixed hu0 hU (Nat.ceil r)

/--
Ceiling-index bounded-base explicit fixed-point-to-orbit bridge under an upper-envelope constant.
-/
theorem ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_upperConstant_baseBound_ceilIndex
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max U U0 : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 2 * (3 * C_max + gamma * |Real.log min_b|) ≤ U)
    (hU0 : variationSeminorm u0 ≤ U0)
    (r : ℝ) :
    variationSeminorm ((Psi^[Nat.ceil r]) u0) ≤ U0 + U :=
  ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_upperConstant_baseBound
    (Psi := Psi) hPsi (u0 := u0) hfix hfixed hU hU0 (Nat.ceil r)

/--
Successor-index ceiling-index bounded-base explicit fixed-point-to-orbit bridge
under an upper-envelope constant.
-/
theorem ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_upperConstant_baseBound_ceilIndex_succ
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max U U0 : ℝ}
    (hfixed : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 2 * (3 * C_max + gamma * |Real.log min_b|) ≤ U)
    (hU0 : variationSeminorm u0 ≤ U0)
    (r : ℝ) :
    variationSeminorm ((Psi^[Nat.ceil r + 1]) u0) ≤ U0 + U := by
  simpa [Nat.succ_eq_add_one] using
    ot_uniformIterateBound_from_explicitFixedPoint_threeCmax_upperConstant_baseBound_succ
      (Psi := Psi) hPsi (u0 := u0) hfix hfixed hU hU0 (Nat.ceil r)

end OT
end Applications
end KLProjection
end FlowSinkhorn
