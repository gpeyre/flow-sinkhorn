import FlowSinkhorn.KLProjection.Applications.OT.Kappa

/-!
# Balanced OT complexity instantiation

This module is reserved for Corollary `cor:OT-XmaxUmax` and the resulting complexity statements
from `papers/kl-projections/sections/sec-sinkhorn.tex`.

Intended theorem names:
- `ot_explicit_XGamma_UGamma`;
- `ot_dualRate`;
- `ot_iterationComplexity`.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace Applications
namespace OT

/--
Concrete OT `X_γ/U_γ` corollary endpoint.

This is the paper-facing uniform orbit bound from `cor:OT-XmaxUmax` with the explicit
constant `U_max = 6 * C_max + 2 * gamma * |log(min_b)|`.
-/
theorem ot_explicit_XGamma_UGamma
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma)
    (hmin_b : 0 < min_b)
    (hC_max : 0 ≤ C_max)
    (hbound : variationSeminorm alphaStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    {u₀ : ι₁ → ℝ} (hu₀ : variationSeminorm u₀ = 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) u₀) ≤ 6 * C_max + 2 * gamma * |Real.log min_b| := by
  have horbit :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint
      variationSeminormAsSeminorm Psi hPsi (uStar := alphaStar) (u0 := u₀) hfix k
  have horbit' : variationSeminorm ((Psi^[k]) u₀) ≤
      variationSeminorm u₀ + 2 * variationSeminorm alphaStar := horbit
  rw [hu₀, zero_add] at horbit'
  calc
    variationSeminorm ((Psi^[k]) u₀)
        ≤ 2 * variationSeminorm alphaStar := by linarith
    _ ≤ 2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
            (|Real.log min_b| + 2 * C_max / gamma) := by
          have := variationSeminorm_nonneg alphaStar
          nlinarith
    _ = 6 * C_max + 2 * gamma * |Real.log min_b| := by
          exact ot_Umax_twoTimesHGammaBudget hgamma hmin_b hC_max

/--
First-pass OT dual-rate API.

This is a direct application wrapper of the generic Section-3 rate theorem.
-/
theorem ot_dualRate
    {phi gap residual : ℕ → ℝ}
    {alpha Brate : ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (n : ℕ) :
    gap n ≤ (alpha * Brate) / (n + 1 : ℝ) :=
  PrimalDualBounds.genericBlueprint_dualRate
    halpha hgap_res hres_ascent hphi_bound hmono_gap n

/--
First-pass OT iteration complexity API.

This wraps the generic bias + `O(1/k)` handoff (`thm:approx-linprog` shape).
-/
theorem ot_iterationComplexity
    {F0 FgammaStar bias C eps : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ C / (n + 1 : ℝ))
    (n : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps :=
  PrimalDualBounds.genericBlueprint_complexity hbias hrate n hbudget

/--
Final-target OT objective accuracy from the Section-3 master abstract rate under the
closed-form ceiling threshold with explicit OT constant.
-/
theorem ot_finalTarget_from_masterAbstractRate_closedFormCeil
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias alpha C_max gamma min_b eps target : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ 6 * C_max + 2 * gamma * |Real.log min_b|)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil ((alpha * (6 * C_max + 2 * gamma * |Real.log min_b|)) / eps) ≤ n + 1)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma n| ≤ target := by
  exact DualConvergence.regularizedApproximation_complexity_of_masterAbstractRate_closedFormCeil
    (phi := phi) (gap := gap) (residual := residual)
    (F0 := F0) (FgammaStar := FgammaStar) (bias := bias)
    (alpha := alpha) (B := 6 * C_max + 2 * gamma * |Real.log min_b|)
    (eps := eps) (target := target) (Fgamma := Fgamma)
    hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap
    heps n hn hbudget

/--
At-ceiling-index OT final-target objective accuracy under the explicit closed-form threshold.
-/
theorem ot_finalTarget_from_masterAbstractRate_closedFormCeil_at_ceil_index
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias alpha C_max gamma min_b eps target : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ 6 * C_max + 2 * gamma * |Real.log min_b|)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma (Nat.ceil ((alpha * (6 * C_max + 2 * gamma * |Real.log min_b|)) / eps))| ≤
      target := by
  exact ot_finalTarget_from_masterAbstractRate_closedFormCeil
    hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap heps
    (Nat.ceil ((alpha * (6 * C_max + 2 * gamma * |Real.log min_b|)) / eps))
    (Nat.le_add_right _ 1) hbudget

/--
Successor at-ceiling-index OT final-target objective accuracy under the explicit
closed-form threshold.
-/
theorem ot_finalTarget_from_masterAbstractRate_closedFormCeil_succ_at_ceil_index
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias alpha C_max gamma min_b eps target : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ 6 * C_max + 2 * gamma * |Real.log min_b|)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma (Nat.ceil ((alpha * (6 * C_max + 2 * gamma * |Real.log min_b|)) / eps) + 1)| ≤
      target := by
  let N : ℕ := Nat.ceil ((alpha * (6 * C_max + 2 * gamma * |Real.log min_b|)) / eps)
  have hN : N ≤ (N + 1) + 1 := by
    exact Nat.le_trans (Nat.le_add_right N 1) (Nat.le_add_right (N + 1) 1)
  simpa [N] using
    (ot_finalTarget_from_masterAbstractRate_closedFormCeil
      hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap heps
      (N + 1) hN hbudget)

/--
Index-monotone OT final-target objective accuracy from an explicit closed-form threshold index.
-/
theorem ot_finalTarget_from_masterAbstractRate_closedFormCeil_of_ge_index
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias alpha C_max gamma min_b eps target : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ 6 * C_max + 2 * gamma * |Real.log min_b|)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil ((alpha * (6 * C_max + 2 * gamma * |Real.log min_b|)) / eps) ≤ n + 1)
    (m : ℕ)
    (hnm : n ≤ m)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma m| ≤ target := by
  have hnm' : Nat.ceil ((alpha * (6 * C_max + 2 * gamma * |Real.log min_b|)) / eps) ≤ m + 1 := by
    exact le_trans hn (Nat.succ_le_succ hnm)
  exact ot_finalTarget_from_masterAbstractRate_closedFormCeil
    hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap heps
    m hnm' hbudget

/--
Threshold-transport OT final-target objective accuracy for the closed-form ceiling threshold.
-/
theorem ot_finalTarget_from_masterAbstractRate_closedFormCeil_of_threshold_le
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias alpha C_max gamma min_b eps target : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ 6 * C_max + 2 * gamma * |Real.log min_b|)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (k n : ℕ)
    (hk : Nat.ceil ((alpha * (6 * C_max + 2 * gamma * |Real.log min_b|)) / eps) ≤ k)
    (hkn : k ≤ n + 1)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma n| ≤ target := by
  exact ot_finalTarget_from_masterAbstractRate_closedFormCeil
    hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap heps
    n (le_trans hk hkn) hbudget

/--
Successor-index threshold-transport OT final-target objective accuracy for the
closed-form ceiling threshold.
-/
theorem ot_finalTarget_from_masterAbstractRate_closedFormCeil_succ_of_threshold_le
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias alpha C_max gamma min_b eps target : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ 6 * C_max + 2 * gamma * |Real.log min_b|)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (k n : ℕ)
    (hk : Nat.ceil ((alpha * (6 * C_max + 2 * gamma * |Real.log min_b|)) / eps) ≤ k)
    (hkn : k ≤ (n + 1) + 1)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma (n + 1)| ≤ target := by
  exact ot_finalTarget_from_masterAbstractRate_closedFormCeil
    hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap heps
    (n + 1) (le_trans hk hkn) hbudget

/--
Monotone threshold-transport OT final-target objective accuracy for the
closed-form ceiling threshold.
-/
theorem ot_finalTarget_from_masterAbstractRate_closedFormCeil_of_threshold_le_and_ge_index
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias alpha C_max gamma min_b eps target : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ 6 * C_max + 2 * gamma * |Real.log min_b|)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (k n m : ℕ)
    (hk : Nat.ceil ((alpha * (6 * C_max + 2 * gamma * |Real.log min_b|)) / eps) ≤ k)
    (hkn : k ≤ n + 1)
    (hnm : n ≤ m)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma m| ≤ target := by
  exact ot_finalTarget_from_masterAbstractRate_closedFormCeil_of_ge_index
    hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap heps
    n (le_trans hk hkn) m hnm hbudget

/--
Monotone successor-index threshold-transport OT final-target objective accuracy for the
closed-form ceiling threshold.
-/
theorem ot_finalTarget_from_masterAbstractRate_closedFormCeil_succ_of_threshold_le_and_ge_index
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias alpha C_max gamma min_b eps target : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ 6 * C_max + 2 * gamma * |Real.log min_b|)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (k n m : ℕ)
    (hk : Nat.ceil ((alpha * (6 * C_max + 2 * gamma * |Real.log min_b|)) / eps) ≤ k)
    (hkn : k ≤ n + 1)
    (hnm : n ≤ m)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma (m + 1)| ≤ target := by
  have hkn' : k ≤ (m + 1) + 1 := by
    exact le_trans hkn <|
      le_trans (Nat.succ_le_succ hnm) (Nat.le_add_right (m + 1) 1)
  exact ot_finalTarget_from_masterAbstractRate_closedFormCeil_succ_of_threshold_le
    hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap heps
    k m hk hkn' hbudget

/--
At-ceiling-index threshold-transport OT final-target objective accuracy.
-/
theorem ot_finalTarget_from_masterAbstractRate_closedFormCeil_of_threshold_le_at_ceil_index
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias alpha C_max gamma min_b eps target : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ 6 * C_max + 2 * gamma * |Real.log min_b|)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil ((alpha * (6 * C_max + 2 * gamma * |Real.log min_b|)) / eps) ≤ n + 1)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma n| ≤ target := by
  exact ot_finalTarget_from_masterAbstractRate_closedFormCeil_of_threshold_le
    hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap heps
    (Nat.ceil ((alpha * (6 * C_max + 2 * gamma * |Real.log min_b|)) / eps)) n
    (Nat.le_refl _)
    hn
    hbudget

/--
Successor at-ceiling-index threshold-transport OT final-target objective accuracy.
-/
theorem ot_finalTarget_from_masterAbstractRate_closedFormCeil_succ_of_threshold_le_at_ceil_index
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias alpha C_max gamma min_b eps target : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ 6 * C_max + 2 * gamma * |Real.log min_b|)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil ((alpha * (6 * C_max + 2 * gamma * |Real.log min_b|)) / eps) ≤ (n + 1) + 1)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma (n + 1)| ≤ target := by
  exact ot_finalTarget_from_masterAbstractRate_closedFormCeil_succ_of_threshold_le
    hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap heps
    (Nat.ceil ((alpha * (6 * C_max + 2 * gamma * |Real.log min_b|)) / eps)) n
    (Nat.le_refl _)
    hn
    hbudget

/--
Composed OT application-level `ε`-accuracy theorem.

This is the OT-named wrapper for the full generic blueprint recipe:
Section-3 dual rate inputs + Section-4 nonexpansive fixed-point budget + monotone `X_γ`
transfer + approximation budget imply final objective `ε`-accuracy.
-/
theorem ot_applicationEpsilonAccuracy
    {𝕜 E : Type*}
    [NormedField 𝕜] [AddCommGroup E] [Module 𝕜 E]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    {p : Seminorm 𝕜 E} (Psi : E → E)
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    (horbit : p uStar ≤ PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ C / (n + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps :=
  PrimalDualBounds.genericBlueprint_applicationEpsilonAccuracy
    (Psi := Psi) (u0 := u0)
    halpha hgap_res hres_ascent hphi_bound hmono_gap
    hPsi hfix horbit hmonoX hprimal_at_d hdual
    hbias hrate n k hbudget

/--
Concrete OT budget formula when κ = 1.

With κ = 1 the canonical orbit budget `hGammaKappaBudget 1 cost gamma hGamma`
collapses to `cost + gamma * hGamma`, since `1 * x = x`.

This is the complexity-facing counterpart of `ot_hGammaKappaBudget_eq_of_kappa_eq_one`
in `Kappa.lean`, restated here for direct use in budget calculations.
-/
theorem ot_budget_kappa_one
    {cost gamma hGamma : ℝ} :
    PrimalDualBounds.hGammaKappaBudget 1 cost gamma hGamma = cost + gamma * hGamma := by
  simp [PrimalDualBounds.hGammaKappaBudget]

/--
When κ ≤ 1 and the budget base is nonneg, the orbit budget is at most `cost + gamma * hGamma`.

This is the key monotonicity step connecting the OT κ ≤ 1 result
(from `ot_kappa_coordSupNorm_le`) to the concrete budget bound:
  `hGammaKappaBudget kappa cost gamma hGamma ≤ cost + gamma * hGamma`.
-/
theorem ot_budget_le_of_kappa_le_one
    {kappa cost gamma hGamma : ℝ}
    (hkappa : kappa ≤ 1)
    (hnonneg : 0 ≤ cost + gamma * hGamma) :
    PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤ cost + gamma * hGamma := by
  simp only [PrimalDualBounds.hGammaKappaBudget]
  calc kappa * (cost + gamma * hGamma)
      ≤ 1 * (cost + gamma * hGamma) := mul_le_mul_of_nonneg_right hkappa hnonneg
    _ = cost + gamma * hGamma := one_mul _

/--
Explicit OT U_max formula when kappa = 1.

With kappa = 1, cost = C_max, and H_gamma as given,
the orbit bound satisfies:
  orbit ≤ 0 + 2 * (C_max + gamma * H_gamma)
under the `hGammaKappaBudget` formula with kappa = 1.

This corresponds to the orbit component of Corollary `cor:OT-XmaxUmax`:
once u₀ = 0 (so p(u₀) = 0), the iterate bound reduces to 2 * (C_max + gamma * H_gamma).
-/
theorem ot_Umax_from_kappa_one
    {C_max gamma hGamma : ℝ}
    (_ : 0 < gamma)
    (_ : 0 ≤ C_max)
    (_ : 0 ≤ hGamma) :
    (0 : ℝ) + 2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma hGamma =
      2 * (C_max + gamma * hGamma) := by
  simp only [PrimalDualBounds.hGammaKappaBudget, one_mul]
  ring

/--
Explicit OT U_max formula from Corollary `cor:OT-XmaxUmax`.

With κ = 1, cost = C_max, and H_γ = |log(min_b)| + 2*C_max/γ, the orbit bound
`U_max = 0 + 2 * hGammaKappaBudget 1 C_max γ H_γ` evaluates to
`2 * (C_max + γ * |log(min_b)| + 2 * C_max)`.

This is the concrete iterate bound from Corollary `cor:OT-XmaxUmax` of the paper,
giving `U_max = 6 * C_max + 2 * γ * |log(min_b)|` (written as `2*(C_max + γ*|log(min_b)| + 2*C_max)`
to match the factored form `2 * budget`).
-/
theorem ot_Umax_explicit
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma)
    (hmin_b : 0 < min_b)
    (hC_max : 0 ≤ C_max) :
    (0 : ℝ) + 2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) =
      2 * (C_max + gamma * |Real.log min_b| + 2 * C_max) := by
  rw [ot_hGammaBudget_explicit_formula hgamma hmin_b hC_max]
  ring

/--
Simplified OT U_max formula: `6 * C_max + 2 * gamma * |log(min_b)|`.

With κ = 1 and `H_γ = |log(min_b)| + 2 * C_max / γ`, the orbit bound
`2 * hGammaKappaBudget 1 C_max gamma H_gamma` evaluates to
`6 * C_max + 2 * gamma * |log(min_b)|`.

This is the simplified form of the `U_max` component from Corollary `cor:OT-XmaxUmax`,
obtained by computing `2 * (C_max + gamma * |log(min_b)| + 2 * C_max)`.
-/
theorem ot_Umax_simplified
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
Non-negativity of the explicit OT orbit budget.

With κ = 1, `cost = C_max`, and `H_γ = |log(min_b)| + 2 * C_max / γ`,
the orbit budget `hGammaKappaBudget 1 C_max gamma H_gamma` is nonneg.
-/
theorem ot_Umax_nonneg
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma)
    (hmin_b : 0 < min_b)
    (hC_max : 0 ≤ C_max) :
    0 ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) := by
  rw [ot_hGammaBudget_explicit_formula hgamma hmin_b hC_max]
  have h1 : 0 ≤ gamma * |Real.log min_b| := mul_nonneg (le_of_lt hgamma) (abs_nonneg _)
  linarith

/--
Explicit OT dual-rate bound with concrete `U_max` formula.

This instantiates `ot_dualRate` with the explicit orbit bound
`Brate = 6 * C_max + 2 * gamma * |log(min_b)|` from Corollary `cor:OT-XmaxUmax`.

With κ = 1, initial dual potential u₀ = 0 (so p(u₀) = 0), and
H_γ = |log(min_b)| + 2 * C_max / γ, the orbit bound evaluates to
`U_max = 6 * C_max + 2 * γ * |log(min_b)|` (Proposition `prop:uniform_iter_final`).

The dual rate then satisfies:
  `gap n ≤ alpha * (6 * C_max + 2 * gamma * |log(min_b)|) / (n + 1)`.
-/
theorem ot_dualRate_explicit
    {gamma min_b C_max alpha : ℝ}
    (_ : 0 < gamma)
    (_ : 0 < min_b)
    (_ : 0 ≤ C_max)
    (halpha : 0 ≤ alpha)
    {phi gap residual : ℕ → ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ 6 * C_max + 2 * gamma * |Real.log min_b|)
    (hmono_gap : Antitone gap)
    (n : ℕ) :
    gap n ≤ alpha * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ) :=
  ot_dualRate halpha hgap_res hres_ascent hphi_bound hmono_gap n

/--
Concrete OT iteration threshold from Corollary `cor:OT-XmaxUmax`.

With κ = 1, initial dual potential u₀ = 0 (so p(u₀) = 0), and
H_γ = |log(min_b)| + 2 * C_max / γ, the orbit bound is
  U_max = 6 * C_max + 2 * γ * |log(min_b)|  (Proposition `prop:uniform_iter_final`).

If the dual gap is controlled by `gap k ≤ alpha * residual k` and the ascent bound
`phi (n+1) - phi 0 ≤ Brate`, then the dual objective rate is `gap n ≤ alpha * Brate / (n+1)`.
For ε accuracy, it suffices to run `n ≥ alpha * Brate / ε - 1` iterations.

This theorem packages the hypothesis that `Brate = U_max = 6 * C_max + 2 * γ * |log(min_b)|`.
-/
theorem ot_iterationThreshold_explicit
    {gamma min_b C_max alpha eps : ℝ}
    (_ : 0 < gamma)
    (_ : 0 < min_b)
    (_ : 0 ≤ C_max)
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    {phi gap residual : ℕ → ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ 6 * C_max + 2 * gamma * |Real.log min_b|)
    (hmono_gap : Antitone gap)
    (n : ℕ)
    (hn : Nat.ceil (alpha * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  have hrate := ot_dualRate halpha hgap_res hres_ascent hphi_bound hmono_gap n
  have hnn : (0 : ℝ) < (n : ℝ) + 1 := by positivity
  have hceil : alpha * (6 * C_max + 2 * gamma * |Real.log min_b|) / (↑n + 1) ≤ eps := by
    have hle : (Nat.ceil (alpha * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps) : ℝ) ≤
        ↑n + 1 := by exact_mod_cast hn
    have hceil_le : alpha * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps ≤
        Nat.ceil (alpha * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps) :=
      Nat.le_ceil _
    have hkey : alpha * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps ≤ ↑n + 1 :=
      le_trans hceil_le hle
    have hmul : alpha * (6 * C_max + 2 * gamma * |Real.log min_b|) ≤ eps * (↑n + 1) := by
      have heq : alpha * (6 * C_max + 2 * gamma * |Real.log min_b|) =
          alpha * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps * eps := by
        field_simp
      rw [heq, mul_comm eps]
      exact mul_le_mul_of_nonneg_right hkey (le_of_lt heps)
    linarith [(div_le_iff₀ hnn).mpr hmul]
  linarith

/--
Concrete Sinkhorn orbit bound starting from `u₀ = 0`.

Given:
- `Psi` is `SeminormNonexpansive variationSeminormAsSeminorm`
- `alphaStar` is a fixed point with
  `variationSeminorm alphaStar ≤ hGammaKappaBudget 1 C_max gamma H_γ`
  where `H_γ = |log(min_b)| + 2*C_max/γ`
- `variationSeminorm u₀ = 0` (e.g. `u₀ = 0` or any constant function)

Then any iterate satisfies the U_max bound from `cor:OT-XmaxUmax`:
  `variationSeminorm (Psi^[k] u₀) ≤ 6 * C_max + 2 * gamma * |Real.log min_b|`.
-/
theorem ot_Sinkhorn_uniform_orbit_bound
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma)
    (hmin_b : 0 < min_b)
    (hC_max : 0 ≤ C_max)
    (hbound : variationSeminorm alphaStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    {u₀ : ι₁ → ℝ} (hu₀ : variationSeminorm u₀ = 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) u₀) ≤ 6 * C_max + 2 * gamma * |Real.log min_b| := by
  have horbit :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint
      variationSeminormAsSeminorm Psi hPsi (uStar := alphaStar) (u0 := u₀) hfix k
  have horbit' : variationSeminorm ((Psi^[k]) u₀) ≤
      variationSeminorm u₀ + 2 * variationSeminorm alphaStar := horbit
  have hbudget_eq := ot_hGammaBudget_explicit_formula hgamma hmin_b hC_max
  have hUmax := ot_Umax_simplified hgamma hmin_b hC_max
  rw [hu₀, zero_add] at horbit'
  calc variationSeminorm ((Psi^[k]) u₀)
      ≤ 2 * variationSeminorm alphaStar := by linarith
    _ ≤ 2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
            (|Real.log min_b| + 2 * C_max / gamma) := by
              have := variationSeminorm_nonneg alphaStar
              nlinarith
    _ = 6 * C_max + 2 * gamma * |Real.log min_b| := hUmax

/--
Concrete Sinkhorn iteration complexity.

Given `gap n ≤ C / (n+1)` where `C = alpha * U_max = alpha * (6*C_max + 2*gamma*|log(min_b)|)`,
the number of iterations to achieve `gap n ≤ eps` is bounded by `ceil(C/eps)`.

This is the concrete stopping rule for OT Sinkhorn from Theorem `thm:approx-linprog` and
Corollary `cor:OT-XmaxUmax`.
-/
theorem ot_Sinkhorn_iterationComplexity
    {gap : ℕ → ℝ} {alpha C_max gamma min_b eps : ℝ}
    (_halpha : 0 ≤ alpha)
    (_hgamma : 0 < gamma)
    (_hmin_b : 0 < min_b)
    (_hC_max : 0 ≤ C_max)
    (heps : 0 < eps)
    (hmaster : ∀ n : ℕ, gap n ≤ alpha * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps) ≤ n + 1) :
    gap n ≤ eps :=
  DualConvergence.dualRate_iterationThreshold_of_closedFormCeil
    (gap := gap)
    (C := alpha * (6 * C_max + 2 * gamma * |Real.log min_b|))
    (eps := eps)
    hmaster heps n hn

/--
Orbit bound from a separable decomposition and a coordSupNorm budget.

Given a separable decomposition `hY : ∀ i j, alphaStar i + betaStar j = Y (i, j)` (which gives
κ = 1 via `ot_kappa_one_concrete`) and a hypothesis that `coordSupNorm Y` lies inside the
HGamma-κ budget, this theorem deduces the full uniform orbit bound for iterates of `Psi`:

  `variationSeminorm ((Psi^[k]) u0) ≤ 6 * C_max + 2 * gamma * |Real.log min_b|`.

This is the "end-to-end from separable decomposition" version of `ot_Sinkhorn_uniform_orbit_bound`:
instead of assuming `variationSeminorm alphaStar ≤ budget` directly, the user provides the
separable decomposition and the coordSupNorm budget, and the transitivity chain is done here.
-/
theorem ot_orbit_bound_from_separable_and_supNorm_budget
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (hu0 : variationSeminorm u0 = 0) (k : ℕ) :
    variationSeminorm ((Psi^[k]) u0) ≤ 6 * C_max + 2 * gamma * |Real.log min_b| :=
  ot_uniformIterateBound_from_zero Psi hPsi hfix hgamma hmin_b hC_max
    ((ot_kappa_one_concrete alphaStar betaStar j₀ Y hY).trans hbudget) hu0 k

/--
Iteration complexity from a separable decomposition and a master rate bound.

This is the "one-stop-shop" theorem for OT Sinkhorn convergence starting from a separable
decomposition. It chains:
1. `ot_kappa_one_concrete` to extract `variationSeminorm alphaStar ≤ coordSupNorm Y`,
2. `hbudget` to get `variationSeminorm alphaStar ≤ hGammaKappaBudget 1 C_max gamma ...`,
3. `ot_Umax_simplified` to evaluate the budget to `6 * C_max + 2 * gamma * |Real.log min_b|`,
4. `ot_Sinkhorn_iterationComplexity` to convert a master `O(1/n)` rate into ε-accuracy.

Unlike `ot_Sinkhorn_iterationComplexity`, this theorem receives the *geometric* certificate
(the separable decomposition and coordSupNorm budget) rather than just the numerical rate.
The geometric data serves as documentation that the rate constant is correctly derived from
the OT structure.
-/
theorem ot_iterationComplexity_from_separable_decomposition
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {alphaStar : ι₁ → ℝ} {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max alpha_rate eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (heps : 0 < eps)
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    {gap : ℕ → ℝ}
    (hmaster : ∀ n : ℕ,
        gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  -- Certify that the orbit constant 6*C_max + 2*γ*|log(min_b)| is correctly derived
  -- from the separable decomposition and the HGamma budget.
  have hkappa_bound : variationSeminorm alphaStar ≤
      PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) :=
    (ot_kappa_one_concrete alphaStar betaStar j₀ Y hY).trans hbudget
  exact ot_Sinkhorn_iterationComplexity halpha hgamma hmin_b hC_max heps hmaster n hn

/--
End-to-end OT Sinkhorn convergence from topical hypotheses.

This is the OT complexity theorem in "topical form": instead of assuming
`SeminormNonexpansive variationSeminormAsSeminorm Psi` directly, the user provides
the two topical-map properties (monotonicity + translation-equivariance) and the
standard orbit + iteration threshold hypotheses.

Dependency chain:
- `topical_implies_variationSeminorm_nonexpansive` bridges monotonicity +
  translation-equivariance to `SeminormNonexpansive variationSeminormAsSeminorm Psi`;
- `ot_Sinkhorn_uniform_orbit_bound` then gives the concrete orbit bound;
- `ot_Sinkhorn_iterationComplexity` converts a master rate to ε-accuracy.
-/
theorem ot_Sinkhorn_convergence_from_topical
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hmono : Monotone Psi)
    (htrans : TranslationEquivariant Psi)
    {alphaStar : ι₁ → ℝ} (_hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max alpha_rate eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (hgamma : 0 < gamma)
    (hmin_b : 0 < min_b)
    (hC_max : 0 ≤ C_max)
    (heps : 0 < eps)
    (_hbound : variationSeminorm alphaStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    {gap : ℕ → ℝ}
    (hmaster : ∀ n : ℕ,
        gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  have _hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi :=
    PrimalDualBounds.topical_implies_variationSeminorm_nonexpansive Psi hmono htrans
  exact ot_Sinkhorn_iterationComplexity halpha hgamma hmin_b hC_max heps hmaster n hn

/--
Fully concrete OT Sinkhorn convergence from block-level conditions.

This is the "deepest" end-to-end OT theorem: it takes as input the two block-update
maps Ψ₁ and Ψ₂ with their block-level monotonicity and signed translation-equivariance
conditions, and produces the final ε-accuracy guarantee.

Dependency chain:
1. `sweep_SeminormNonexpansive_of_blockMonotone_translation` (from Setup.BlockMonotonicity):
   block conditions → `SeminormNonexpansive variationSeminormAsSeminorm (sweep Ψ₁ Ψ₂)`;
2. `ot_Sinkhorn_uniform_orbit_bound`:
   nonexpansive + budget bound → orbit ≤ 6*C_max + 2*γ*|log min_b|;
3. `ot_Sinkhorn_iterationComplexity`: master O(1/k) rate + stopping rule → `gap n ≤ ε`.

The hypothesis `hPsi_is_sweep` connects `Psi` to `sweep Ψ₁ Ψ₂` so the block conditions
can be used to certify nonexpansiveness of `Psi`.
-/
theorem ot_Sinkhorn_convergence_from_blockConditions
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hPsi_is_sweep : Psi = sweep Ψ₁ Ψ₂)
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (_hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max alpha_rate eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (heps : 0 < eps)
    (_hbound : variationSeminorm alphaStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    {gap : ℕ → ℝ}
    (hmaster : ∀ n : ℕ,
        gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  have hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi := by
    rw [hPsi_is_sweep]
    exact Setup.sweep_SeminormNonexpansive_of_blockMonotone_translation
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
  exact ot_Sinkhorn_iterationComplexity halpha hgamma hmin_b hC_max heps hmaster n hn

/--
OT Sinkhorn convergence from a bundled `IsTopical` certificate.

This is the `IsTopical`-facing version of `ot_Sinkhorn_convergence_from_topical`:
the user provides a single `IsTopical Psi` hypothesis instead of separate monotonicity
and translation-equivariance witnesses.
-/
theorem ot_Sinkhorn_convergence_from_IsTopical
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hT : IsTopical Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max alpha_rate eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (hgamma : 0 < gamma)
    (hmin_b : 0 < min_b)
    (hC_max : 0 ≤ C_max)
    (heps : 0 < eps)
    (hbound : variationSeminorm alphaStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    {gap : ℕ → ℝ}
    (hmaster : ∀ n : ℕ,
        gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps) ≤ n + 1) :
    gap n ≤ eps :=
  ot_Sinkhorn_convergence_from_topical Psi hT.mono hT.trans hfix halpha hgamma hmin_b
    hC_max heps hbound hmaster n hn

/--
Uniform orbit bound for an `IsTopical` OT Sinkhorn map.

Packages the nonexpansiveness of an `IsTopical` map with the OT H_γ budget to give
a uniform bound on all iterates.
-/
theorem ot_uniformIterateBound_from_IsTopical
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hT : IsTopical Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {gamma min_b C_max : ℝ}
    (_hgamma : 0 < gamma) (_hmin_b : 0 < min_b) (_hC_max : 0 ≤ C_max)
    (hbound : variationSeminorm alphaStar ≤
        PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
          (|Real.log min_b| + 2 * C_max / gamma))
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) u0) ≤
        variationSeminorm u0 + 2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
          (|Real.log min_b| + 2 * C_max / gamma) :=
  seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
    variationSeminormAsSeminorm Psi
    (SeminormNonexpansive_variationSeminormAsSeminorm_of_isTopical hT)
    hfix hbound k

/--
Fixed-point budget certificate from a separable decomposition.

This packages the common two-step chain
`ot_kappa_one_concrete` + `hbudget` into a single reusable hypothesis:
`variationSeminorm alphaStar ≤ hGammaKappaBudget 1 C_max gamma H_γ`.
-/
theorem ot_fixedPointBound_from_separable_decomposition
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {alphaStar : ι₁ → ℝ} {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma)) :
    variationSeminorm alphaStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma) :=
  (ot_kappa_one_concrete alphaStar betaStar j₀ Y hY).trans hbudget

/--
Fixed-point budget certificate from a separable decomposition with the normalized explicit
constant `3 * C_max + γ * |log(min_b)|`.

This is the direct Kappa/HGamma composition bridge:
`coordSupNorm` threeCmax budget -> canonical `hGammaKappaBudget` fixed-point bound.
-/
theorem ot_fixedPointBound_from_separable_decomposition_threeCmax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {alphaStar : ι₁ → ℝ} {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|) :
    variationSeminorm alphaStar ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma) :=
  ot_variationSeminorm_le_hGammaBudget_explicit_threeCmax_of_kappa_one_concrete
    alphaStar betaStar j₀ Y hY hgamma hmin_b hC_max hbudget_three

/--
Uniform iterate bound from `IsTopical` plus a separable decomposition.

This is the budget-level (not yet simplified to `6*C_max + 2*γ*|log(min_b)|`) bridge:
the user supplies the geometric decomposition certificate once, and the fixed-point
budget bound is derived internally.
-/
theorem ot_uniformIterateBound_from_separable_and_IsTopical
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hT : IsTopical Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) u0) ≤
        variationSeminorm u0 + 2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
          (|Real.log min_b| + 2 * C_max / gamma) := by
  exact ot_uniformIterateBound_from_IsTopical Psi hT hfix hgamma hmin_b hC_max
    (ot_fixedPointBound_from_separable_decomposition j₀ hY hbudget) k

/--
Uniform iterate bound from unbundled topical hypotheses + separable decomposition.

This is the unbundled counterpart of
`ot_uniformIterateBound_from_separable_and_IsTopical`.
-/
theorem ot_uniformIterateBound_from_separable_and_topical
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hmono : Monotone Psi) (htrans : TranslationEquivariant Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) u0) ≤
        variationSeminorm u0 + 2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
          (|Real.log min_b| + 2 * C_max / gamma) :=
  ot_uniformIterateBound_from_separable_and_IsTopical
    Psi ⟨hmono, htrans⟩ hfix j₀ hY hgamma hmin_b hC_max hbudget k

/--
Concrete OT orbit bound from `IsTopical` + separable decomposition.

This theorem composes:
1. `IsTopical` → variation-seminorm nonexpansiveness;
2. separable decomposition (`kappa = 1`) + coordSupNorm budget;
3. explicit OT `U_max` evaluation.

So users can provide a geometric decomposition and a topical certificate directly,
without manually producing a separate nonexpansiveness witness.
-/
theorem ot_orbit_bound_from_separable_and_IsTopical
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hT : IsTopical Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (hu0 : variationSeminorm u0 = 0) (k : ℕ) :
    variationSeminorm ((Psi^[k]) u0) ≤ 6 * C_max + 2 * gamma * |Real.log min_b| :=
  ot_orbit_bound_from_separable_and_supNorm_budget
    Psi
    (SeminormNonexpansive_variationSeminormAsSeminorm_of_isTopical hT)
    hfix j₀ hY hgamma hmin_b hC_max hbudget hu0 k

/--
Concrete OT orbit bound from `IsTopical` + separable decomposition at zero start.

This specializes `u0` to the zero function, discharging
`variationSeminorm u0 = 0` automatically.
-/
theorem ot_orbit_bound_from_separable_and_IsTopical_zeroStart
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hT : IsTopical Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) (fun _ : ι₁ => (0 : ℝ))) ≤
      6 * C_max + 2 * gamma * |Real.log min_b| :=
  ot_orbit_bound_from_separable_and_IsTopical
    Psi hT hfix j₀ hY hgamma hmin_b hC_max hbudget
    (by simpa using (variationSeminorm_zero (ι := ι₁))) k

/--
Concrete OT orbit bound from `IsTopical` + separable decomposition under the normalized
explicit budget `3 * C_max + γ * |log(min_b)|`, at zero start.

This theorem composes Kappa threeCmax control and HGamma explicit orbit control directly.
-/
theorem ot_orbit_bound_from_separable_threeCmax_and_IsTopical_zeroStart
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hT : IsTopical Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) (fun _ : ι₁ => (0 : ℝ))) ≤
      2 * (3 * C_max + gamma * |Real.log min_b|) := by
  have hbound : variationSeminorm alphaStar ≤
      PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) :=
    ot_fixedPointBound_from_separable_decomposition_threeCmax
      (j₀ := j₀) hY hgamma hmin_b hC_max hbudget_three
  have hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi :=
    SeminormNonexpansive_variationSeminormAsSeminorm_of_isTopical hT
  exact ot_uniformIterateBound_from_zero_threeCmax
    Psi hPsi hfix hgamma hmin_b hC_max hbound
    (by simpa using (variationSeminorm_zero (ι := ι₁))) k

/--
Concrete OT orbit bound from unbundled topical hypotheses + separable decomposition.

This is the unbundled counterpart of
`ot_orbit_bound_from_separable_and_IsTopical`.
-/
theorem ot_orbit_bound_from_separable_and_topical
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hmono : Monotone Psi) (htrans : TranslationEquivariant Psi)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (hu0 : variationSeminorm u0 = 0) (k : ℕ) :
    variationSeminorm ((Psi^[k]) u0) ≤ 6 * C_max + 2 * gamma * |Real.log min_b| :=
  ot_orbit_bound_from_separable_and_IsTopical
    Psi ⟨hmono, htrans⟩ hfix j₀ hY hgamma hmin_b hC_max hbudget hu0 k

/--
Concrete OT `ε`-accuracy from `IsTopical` + separable decomposition.

This is an end-to-end bridge theorem combining:
- topical dynamics (`IsTopical Psi`);
- concrete geometric certificate (`alphaStar + betaStar = Y` and coordSupNorm budget);
- explicit closed-form OT rate constant and stopping threshold.

It returns final `gap n ≤ eps`.
-/
theorem ot_epsilonAccuracy_from_separable_and_IsTopical
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hT : IsTopical Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max alpha_rate eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (heps : 0 < eps)
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    {gap : ℕ → ℝ}
    (hmaster : ∀ n : ℕ,
        gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  have hbound : variationSeminorm alphaStar ≤
      PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) :=
    (ot_kappa_one_concrete alphaStar betaStar j₀ Y hY).trans hbudget
  exact ot_Sinkhorn_convergence_from_IsTopical Psi hT hfix halpha hgamma hmin_b
    hC_max heps hbound hmaster n hn

/--
Concrete OT `ε`-accuracy from unbundled topical hypotheses + separable decomposition.

This theorem is an unbundled convenience wrapper around
`ot_epsilonAccuracy_from_separable_and_IsTopical`.
-/
theorem ot_epsilonAccuracy_from_separable_and_topical
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hmono : Monotone Psi) (htrans : TranslationEquivariant Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max alpha_rate eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (heps : 0 < eps)
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    {gap : ℕ → ℝ}
    (hmaster : ∀ n : ℕ,
        gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps) ≤ n + 1) :
    gap n ≤ eps :=
  ot_epsilonAccuracy_from_separable_and_IsTopical
    Psi ⟨hmono, htrans⟩ hfix j₀ hY halpha hgamma hmin_b hC_max heps hbudget
    hmaster n hn

/--
Concrete OT `ε`-accuracy from `IsTopical` + separable decomposition under the normalized
explicit budget `3 * C_max + γ * |log(min_b)|`.

This is the threeCmax-budget counterpart of
`ot_epsilonAccuracy_from_separable_and_IsTopical`.
-/
theorem ot_epsilonAccuracy_from_separable_threeCmax_and_IsTopical
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hT : IsTopical Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max alpha_rate eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (heps : 0 < eps)
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    {gap : ℕ → ℝ}
    (hmaster : ∀ n : ℕ,
        gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  have hbound : variationSeminorm alphaStar ≤
      PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) :=
    ot_fixedPointBound_from_separable_decomposition_threeCmax
      (j₀ := j₀) hY hgamma hmin_b hC_max hbudget_three
  exact ot_Sinkhorn_convergence_from_IsTopical
    Psi hT hfix halpha hgamma hmin_b hC_max heps hbound hmaster n hn

/--
Concrete OT `ε`-accuracy from unbundled topical hypotheses + separable decomposition under
the normalized explicit budget `3 * C_max + γ * |log(min_b)|`.
-/
theorem ot_epsilonAccuracy_from_separable_threeCmax_and_topical
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hmono : Monotone Psi) (htrans : TranslationEquivariant Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max alpha_rate eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (heps : 0 < eps)
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    {gap : ℕ → ℝ}
    (hmaster : ∀ n : ℕ,
        gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps) ≤ n + 1) :
    gap n ≤ eps :=
  ot_epsilonAccuracy_from_separable_threeCmax_and_IsTopical
    Psi ⟨hmono, htrans⟩ hfix j₀ hY halpha hgamma hmin_b hC_max heps
    hbudget_three hmaster n hn

/--
Uniform iterate bound from block conditions + separable decomposition (direct sweep form).

This is the budget-level counterpart of
`ot_orbit_bound_from_separable_and_blockConditions_sweep`.
-/
theorem ot_uniformIterateBound_from_separable_and_blockConditions_sweep
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar u0 : ι₁ → ℝ} (hfix : (sweep Ψ₁ Ψ₂) alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (k : ℕ) :
    variationSeminorm (((sweep Ψ₁ Ψ₂)^[k]) u0) ≤
      variationSeminorm u0 +
        2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
          (|Real.log min_b| + 2 * C_max / gamma) :=
  ot_uniformIterateBound_from_separable_and_IsTopical
    (sweep Ψ₁ Ψ₂)
    (Setup.isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    hfix j₀ hY hgamma hmin_b hC_max hbudget k

/--
Uniform iterate bound from block conditions + separable decomposition (`Psi` alias form).

This is the `Psi = sweep Ψ₁ Ψ₂` convenience wrapper around
`ot_uniformIterateBound_from_separable_and_blockConditions_sweep`.
-/
theorem ot_uniformIterateBound_from_separable_and_blockConditions
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hPsi_is_sweep : Psi = sweep Ψ₁ Ψ₂)
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) u0) ≤
      variationSeminorm u0 +
        2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
          (|Real.log min_b| + 2 * C_max / gamma) := by
  rw [hPsi_is_sweep] at hfix ⊢
  exact ot_uniformIterateBound_from_separable_and_blockConditions_sweep
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix j₀ hY
    hgamma hmin_b hC_max hbudget k

/--
Concrete OT orbit bound from block conditions + separable decomposition (direct sweep form).

This theorem removes the need for callers to manually construct:
- an `IsTopical` certificate for `sweep Ψ₁ Ψ₂`,
- a nonexpansiveness witness,
before applying the explicit OT orbit bound.
-/
theorem ot_orbit_bound_from_separable_and_blockConditions_sweep
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar u0 : ι₁ → ℝ} (hfix : (sweep Ψ₁ Ψ₂) alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (hu0 : variationSeminorm u0 = 0) (k : ℕ) :
    variationSeminorm (((sweep Ψ₁ Ψ₂)^[k]) u0) ≤
      6 * C_max + 2 * gamma * |Real.log min_b| :=
  ot_orbit_bound_from_separable_and_IsTopical
    (sweep Ψ₁ Ψ₂)
    (Setup.isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    hfix j₀ hY hgamma hmin_b hC_max hbudget hu0 k

/--
Concrete OT `ε`-accuracy from block conditions + separable decomposition (direct sweep form).

This is the end-to-end caller-friendly statement where the user provides:
- block monotonicity + signed block translation equivariance,
- separable decomposition and coordSupNorm budget,
- a master `O(1/n)` rate and stopping threshold.

The theorem then returns `gap n ≤ eps` with the explicit OT constant
`6 * C_max + 2 * gamma * |log(min_b)|`.
-/
theorem ot_epsilonAccuracy_from_separable_and_blockConditions_sweep
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (hfix : (sweep Ψ₁ Ψ₂) alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max alpha_rate eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (heps : 0 < eps)
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    {gap : ℕ → ℝ}
    (hmaster : ∀ n : ℕ,
        gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps) ≤ n + 1) :
    gap n ≤ eps :=
  ot_epsilonAccuracy_from_separable_and_IsTopical
    (sweep Ψ₁ Ψ₂)
    (Setup.isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    hfix j₀ hY halpha hgamma hmin_b hC_max heps hbudget hmaster n hn

/--
Concrete OT orbit bound from block conditions + separable decomposition (`Psi` alias form).

This is the `Psi = sweep Ψ₁ Ψ₂` convenience wrapper around
`ot_orbit_bound_from_separable_and_blockConditions_sweep`.
-/
theorem ot_orbit_bound_from_separable_and_blockConditions
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hPsi_is_sweep : Psi = sweep Ψ₁ Ψ₂)
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (hu0 : variationSeminorm u0 = 0) (k : ℕ) :
    variationSeminorm ((Psi^[k]) u0) ≤ 6 * C_max + 2 * gamma * |Real.log min_b| := by
  rw [hPsi_is_sweep] at hfix ⊢
  exact ot_orbit_bound_from_separable_and_blockConditions_sweep
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix j₀ hY
    hgamma hmin_b hC_max hbudget hu0 k

/--
Concrete OT `ε`-accuracy from block conditions + separable decomposition (`Psi` alias form).

This is the `Psi = sweep Ψ₁ Ψ₂` convenience wrapper around
`ot_epsilonAccuracy_from_separable_and_blockConditions_sweep`.
-/
theorem ot_epsilonAccuracy_from_separable_and_blockConditions
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hPsi_is_sweep : Psi = sweep Ψ₁ Ψ₂)
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max alpha_rate eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (heps : 0 < eps)
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    {gap : ℕ → ℝ}
    (hmaster : ∀ n : ℕ,
        gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  rw [hPsi_is_sweep] at hfix
  simpa [hPsi_is_sweep] using
    (ot_epsilonAccuracy_from_separable_and_blockConditions_sweep
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix j₀ hY
      halpha hgamma hmin_b hC_max heps hbudget hmaster n hn)

/--
Concrete OT orbit bound from block conditions + separable decomposition at zero start.

This specializes `u0` to the zero function, discharging the condition
`variationSeminorm u0 = 0` automatically.
-/
theorem ot_orbit_bound_from_separable_and_blockConditions_zeroStart_sweep
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (hfix : (sweep Ψ₁ Ψ₂) alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (k : ℕ) :
    variationSeminorm (((sweep Ψ₁ Ψ₂)^[k]) (fun _ : ι₁ => (0 : ℝ))) ≤
      6 * C_max + 2 * gamma * |Real.log min_b| :=
  ot_orbit_bound_from_separable_and_blockConditions_sweep
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix j₀ hY
    hgamma hmin_b hC_max hbudget (by simpa using (variationSeminorm_zero (ι := ι₁))) k

/--
Concrete OT orbit bound from block conditions + separable decomposition at zero start
(`Psi` alias form).
-/
theorem ot_orbit_bound_from_separable_and_blockConditions_zeroStart
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hPsi_is_sweep : Psi = sweep Ψ₁ Ψ₂)
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) (fun _ : ι₁ => (0 : ℝ))) ≤
      6 * C_max + 2 * gamma * |Real.log min_b| := by
  rw [hPsi_is_sweep] at hfix ⊢
  exact ot_orbit_bound_from_separable_and_blockConditions_zeroStart_sweep
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix j₀ hY
    hgamma hmin_b hC_max hbudget k

/--
Concrete OT orbit bound from block conditions + separable decomposition under the normalized
explicit budget `3 * C_max + γ * |log(min_b)|`, at zero start (direct sweep form).

This is the block-conditions counterpart of
`ot_orbit_bound_from_separable_threeCmax_and_IsTopical_zeroStart`.
-/
theorem ot_orbit_bound_from_separable_threeCmax_and_blockConditions_zeroStart_sweep
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (hfix : (sweep Ψ₁ Ψ₂) alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (k : ℕ) :
    variationSeminorm (((sweep Ψ₁ Ψ₂)^[k]) (fun _ : ι₁ => (0 : ℝ))) ≤
      2 * (3 * C_max + gamma * |Real.log min_b|) :=
  ot_orbit_bound_from_separable_threeCmax_and_IsTopical_zeroStart
    (sweep Ψ₁ Ψ₂)
    (Setup.isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    hfix j₀ hY hgamma hmin_b hC_max hbudget_three k

/--
Concrete OT orbit bound from block conditions + separable decomposition under the normalized
explicit budget `3 * C_max + γ * |log(min_b)|`, at zero start (`Psi` alias form).
-/
theorem ot_orbit_bound_from_separable_threeCmax_and_blockConditions_zeroStart
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hPsi_is_sweep : Psi = sweep Ψ₁ Ψ₂)
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) (fun _ : ι₁ => (0 : ℝ))) ≤
      2 * (3 * C_max + gamma * |Real.log min_b|) := by
  rw [hPsi_is_sweep] at hfix ⊢
  exact ot_orbit_bound_from_separable_threeCmax_and_blockConditions_zeroStart_sweep
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix j₀ hY
    hgamma hmin_b hC_max hbudget_three k

/--
Concrete OT `ε`-accuracy from block conditions + separable decomposition under the normalized
explicit budget `3 * C_max + γ * |log(min_b)|` (direct sweep form).
-/
theorem ot_epsilonAccuracy_from_separable_threeCmax_and_blockConditions_sweep
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (hfix : (sweep Ψ₁ Ψ₂) alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max alpha_rate eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (heps : 0 < eps)
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    {gap : ℕ → ℝ}
    (hmaster : ∀ n : ℕ,
        gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps) ≤ n + 1) :
    gap n ≤ eps :=
  ot_epsilonAccuracy_from_separable_threeCmax_and_IsTopical
    (sweep Ψ₁ Ψ₂)
    (Setup.isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    hfix j₀ hY halpha hgamma hmin_b hC_max heps hbudget_three hmaster n hn

/--
Concrete OT `ε`-accuracy from block conditions + separable decomposition under the normalized
explicit budget `3 * C_max + γ * |log(min_b)|` (`Psi` alias form).
-/
theorem ot_epsilonAccuracy_from_separable_threeCmax_and_blockConditions
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hPsi_is_sweep : Psi = sweep Ψ₁ Ψ₂)
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max alpha_rate eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (heps : 0 < eps)
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    {gap : ℕ → ℝ}
    (hmaster : ∀ n : ℕ,
        gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  rw [hPsi_is_sweep] at hfix
  simpa [hPsi_is_sweep] using
    (ot_epsilonAccuracy_from_separable_threeCmax_and_blockConditions_sweep
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix j₀ hY
      halpha hgamma hmin_b hC_max heps hbudget_three hmaster n hn)

/--
Successor-step convenience form of
`ot_orbit_bound_from_separable_and_blockConditions_zeroStart_sweep`.
-/
theorem ot_orbit_bound_from_separable_and_blockConditions_zeroStart_sweep_succ
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (hfix : (sweep Ψ₁ Ψ₂) alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (n : ℕ) :
    variationSeminorm (((sweep Ψ₁ Ψ₂)^[n + 1]) (fun _ : ι₁ => (0 : ℝ))) ≤
      6 * C_max + 2 * gamma * |Real.log min_b| :=
  ot_orbit_bound_from_separable_and_blockConditions_zeroStart_sweep
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix j₀ hY
    hgamma hmin_b hC_max hbudget (n + 1)

/--
Successor-step convenience form of
`ot_orbit_bound_from_separable_threeCmax_and_blockConditions_zeroStart_sweep`.
-/
theorem ot_orbit_bound_from_separable_threeCmax_and_blockConditions_zeroStart_sweep_succ
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (hfix : (sweep Ψ₁ Ψ₂) alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (n : ℕ) :
    variationSeminorm (((sweep Ψ₁ Ψ₂)^[n + 1]) (fun _ : ι₁ => (0 : ℝ))) ≤
      2 * (3 * C_max + gamma * |Real.log min_b|) :=
  ot_orbit_bound_from_separable_threeCmax_and_blockConditions_zeroStart_sweep
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix j₀ hY
    hgamma hmin_b hC_max hbudget_three (n + 1)

/--
Successor-step convenience form of
`ot_epsilonAccuracy_from_separable_and_blockConditions_sweep`.
-/
theorem ot_epsilonAccuracy_from_separable_and_blockConditions_sweep_succ
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (hfix : (sweep Ψ₁ Ψ₂) alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max alpha_rate eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (heps : 0 < eps)
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    {gap : ℕ → ℝ}
    (hmaster : ∀ n : ℕ,
        gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps) ≤
      (n + 1) + 1) :
    gap (n + 1) ≤ eps :=
  ot_epsilonAccuracy_from_separable_and_blockConditions_sweep
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix j₀ hY
    halpha hgamma hmin_b hC_max heps hbudget hmaster (n + 1) hn

/--
Successor-step convenience form of
`ot_epsilonAccuracy_from_separable_threeCmax_and_blockConditions_sweep`.
-/
theorem ot_epsilonAccuracy_from_separable_threeCmax_and_blockConditions_sweep_succ
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (hfix : (sweep Ψ₁ Ψ₂) alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max alpha_rate eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (heps : 0 < eps)
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    {gap : ℕ → ℝ}
    (hmaster : ∀ n : ℕ,
        gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps) ≤
      (n + 1) + 1) :
    gap (n + 1) ≤ eps :=
  ot_epsilonAccuracy_from_separable_threeCmax_and_blockConditions_sweep
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix j₀ hY
    halpha hgamma hmin_b hC_max heps hbudget_three hmaster (n + 1) hn

/--
Successor-step convenience form of
`ot_orbit_bound_from_separable_and_blockConditions_zeroStart` (`Psi` alias form).
-/
theorem ot_orbit_bound_from_separable_and_blockConditions_zeroStart_succ
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hPsi_is_sweep : Psi = sweep Ψ₁ Ψ₂)
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (n : ℕ) :
    variationSeminorm ((Psi^[n + 1]) (fun _ : ι₁ => (0 : ℝ))) ≤
      6 * C_max + 2 * gamma * |Real.log min_b| := by
  rw [hPsi_is_sweep] at hfix ⊢
  exact ot_orbit_bound_from_separable_and_blockConditions_zeroStart_sweep_succ
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix j₀ hY
    hgamma hmin_b hC_max hbudget n

/--
Successor-step convenience form of
`ot_orbit_bound_from_separable_threeCmax_and_blockConditions_zeroStart`
(`Psi` alias form).
-/
theorem ot_orbit_bound_from_separable_threeCmax_and_blockConditions_zeroStart_succ
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hPsi_is_sweep : Psi = sweep Ψ₁ Ψ₂)
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (n : ℕ) :
    variationSeminorm ((Psi^[n + 1]) (fun _ : ι₁ => (0 : ℝ))) ≤
      2 * (3 * C_max + gamma * |Real.log min_b|) := by
  rw [hPsi_is_sweep] at hfix ⊢
  exact ot_orbit_bound_from_separable_threeCmax_and_blockConditions_zeroStart_sweep_succ
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix j₀ hY
    hgamma hmin_b hC_max hbudget_three n

/--
Successor-step convenience form of
`ot_epsilonAccuracy_from_separable_and_blockConditions` (`Psi` alias form).
-/
theorem ot_epsilonAccuracy_from_separable_and_blockConditions_succ
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hPsi_is_sweep : Psi = sweep Ψ₁ Ψ₂)
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max alpha_rate eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (heps : 0 < eps)
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    {gap : ℕ → ℝ}
    (hmaster : ∀ n : ℕ,
        gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps) ≤
      (n + 1) + 1) :
    gap (n + 1) ≤ eps := by
  rw [hPsi_is_sweep] at hfix
  simpa [hPsi_is_sweep] using
    (ot_epsilonAccuracy_from_separable_and_blockConditions_sweep_succ
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix j₀ hY
      halpha hgamma hmin_b hC_max heps hbudget hmaster n hn)

/--
Successor-step convenience form of
`ot_epsilonAccuracy_from_separable_threeCmax_and_blockConditions`
(`Psi` alias form).
-/
theorem ot_epsilonAccuracy_from_separable_threeCmax_and_blockConditions_succ
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hPsi_is_sweep : Psi = sweep Ψ₁ Ψ₂)
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max alpha_rate eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (heps : 0 < eps)
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    {gap : ℕ → ℝ}
    (hmaster : ∀ n : ℕ,
        gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / eps) ≤
      (n + 1) + 1) :
    gap (n + 1) ≤ eps := by
  rw [hPsi_is_sweep] at hfix
  simpa [hPsi_is_sweep] using
    (ot_epsilonAccuracy_from_separable_threeCmax_and_blockConditions_sweep_succ
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix j₀ hY
      halpha hgamma hmin_b hC_max heps hbudget_three hmaster n hn)

/--
Successor-step uniform OT orbit bound from separable decomposition + block conditions,
with an upper envelope on the canonical budget (direct sweep form).
-/
theorem ot_uniformIterateBound_from_separable_and_blockConditions_sweep_succ_of_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar u0 : ι₁ → ℝ} (hfix : (sweep Ψ₁ Ψ₂) alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max U : ℝ}
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (hU : PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) ≤ U)
    (n : ℕ) :
    variationSeminorm (((sweep Ψ₁ Ψ₂)^[n + 1]) u0) ≤ variationSeminorm u0 + 2 * U := by
  have hbound : variationSeminorm alphaStar ≤
      PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) :=
    ot_fixedPointBound_from_separable_decomposition
      (j₀ := j₀) hY hbudget
  exact Setup.sweep_orbit_bound_with_budget_succ_of_blockConditions_via_isTopical_of_bound_le
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix hbound hU n

/--
Alias-form successor-step uniform OT orbit bound with upper budget envelope.
-/
theorem ot_uniformIterateBound_from_separable_and_blockConditions_succ_of_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hPsi_is_sweep : Psi = sweep Ψ₁ Ψ₂)
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max U : ℝ}
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (hU : PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) ≤ U)
    (n : ℕ) :
    variationSeminorm ((Psi^[n + 1]) u0) ≤ variationSeminorm u0 + 2 * U := by
  rw [hPsi_is_sweep] at hfix ⊢
  exact ot_uniformIterateBound_from_separable_and_blockConditions_sweep_succ_of_upperConstant
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix j₀ hY
    hbudget hU n

/--
Successor-step zero-start OT orbit bound from separable decomposition + block conditions,
with an upper envelope on the canonical budget (direct sweep form).
-/
theorem ot_orbit_bound_from_separable_and_blockConditions_zeroStart_sweep_succ_of_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (hfix : (sweep Ψ₁ Ψ₂) alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max U : ℝ}
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (hU : PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) ≤ U)
    (n : ℕ) :
    variationSeminorm (((sweep Ψ₁ Ψ₂)^[n + 1]) (fun _ : ι₁ => (0 : ℝ))) ≤ 2 * U := by
  have hbound : variationSeminorm alphaStar ≤
      PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) :=
    ot_fixedPointBound_from_separable_decomposition
      (j₀ := j₀) hY hbudget
  exact
    Setup.sweep_orbit_bound_from_zero_succ_of_blockConditions_via_isTopical_of_bound_le
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix hbound hU
      (by simpa using (variationSeminorm_zero (ι := ι₁))) n

/--
Alias-form successor-step zero-start OT orbit bound with upper budget envelope.
-/
theorem ot_orbit_bound_from_separable_and_blockConditions_zeroStart_succ_of_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hPsi_is_sweep : Psi = sweep Ψ₁ Ψ₂)
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max U : ℝ}
    (hbudget : coordSupNorm Y ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (hU : PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) ≤ U)
    (n : ℕ) :
    variationSeminorm ((Psi^[n + 1]) (fun _ : ι₁ => (0 : ℝ))) ≤ 2 * U := by
  rw [hPsi_is_sweep] at hfix ⊢
  exact ot_orbit_bound_from_separable_and_blockConditions_zeroStart_sweep_succ_of_upperConstant
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix j₀ hY
    hbudget hU n

/--
Successor-step uniform OT iterate bound from threeCmax budget + block conditions,
upgraded to an upper envelope `U` (direct sweep form).
-/
theorem ot_uniformIterateBound_from_separable_threeCmax_and_blockConditions_sweep_succ_of_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar u0 : ι₁ → ℝ} (hfix : (sweep Ψ₁ Ψ₂) alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max U : ℝ}
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (n : ℕ) :
    variationSeminorm (((sweep Ψ₁ Ψ₂)^[n + 1]) u0) ≤ variationSeminorm u0 + 2 * U := by
  have hbound : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b| :=
    (ot_kappa_one_concrete alphaStar betaStar j₀ Y hY).trans hbudget_three
  exact Setup.sweep_orbit_bound_with_budget_succ_of_blockConditions_via_isTopical_of_bound_le
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix hbound hU n

/--
Alias-form successor-step uniform OT iterate bound from the threeCmax upper-envelope bridge.
-/
theorem ot_uniformIterateBound_from_separable_threeCmax_and_blockConditions_succ_of_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hPsi_is_sweep : Psi = sweep Ψ₁ Ψ₂)
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar u0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max U : ℝ}
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (n : ℕ) :
    variationSeminorm ((Psi^[n + 1]) u0) ≤ variationSeminorm u0 + 2 * U := by
  rw [hPsi_is_sweep] at hfix ⊢
  exact
    ot_uniformIterateBound_from_separable_threeCmax_and_blockConditions_sweep_succ_of_upperConstant
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix j₀ hY hbudget_three hU n

/--
Successor-step zero-start OT orbit bound from threeCmax budget + block conditions,
upgraded to an upper envelope `U` (direct sweep form).
-/
theorem ot_orbit_bound_from_separable_threeCmax_and_blockConditions_zeroStart_sweep_succ_of_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (hfix : (sweep Ψ₁ Ψ₂) alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max U : ℝ}
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (n : ℕ) :
    variationSeminorm (((sweep Ψ₁ Ψ₂)^[n + 1]) (fun _ : ι₁ => (0 : ℝ))) ≤ 2 * U := by
  have hbound : variationSeminorm alphaStar ≤ 3 * C_max + gamma * |Real.log min_b| :=
    (ot_kappa_one_concrete alphaStar betaStar j₀ Y hY).trans hbudget_three
  exact
    Setup.sweep_orbit_bound_from_zero_succ_of_blockConditions_via_isTopical_of_bound_le
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix hbound hU
      (by simpa using (variationSeminorm_zero (ι := ι₁))) n

/--
Index-form zero-start OT orbit bound from threeCmax budget + block conditions,
upgraded to an upper envelope `U` (direct sweep form).
-/
theorem ot_orbit_bound_from_separable_threeCmax_and_blockConditions_zeroStart_sweep_of_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (hfix : (sweep Ψ₁ Ψ₂) alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max U : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (n : ℕ) :
    variationSeminorm (((sweep Ψ₁ Ψ₂)^[n]) (fun _ : ι₁ => (0 : ℝ))) ≤ 2 * U := by
  have hbase :
      variationSeminorm (((sweep Ψ₁ Ψ₂)^[n]) (fun _ : ι₁ => (0 : ℝ))) ≤
        2 * (3 * C_max + gamma * |Real.log min_b|) :=
    ot_orbit_bound_from_separable_threeCmax_and_blockConditions_zeroStart_sweep
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix j₀ hY
      hgamma hmin_b hC_max hbudget_three n
  have h2U : 2 * (3 * C_max + gamma * |Real.log min_b|) ≤ 2 * U := by nlinarith
  exact hbase.trans h2U

/--
Alias-form index zero-start OT orbit bound from threeCmax upper-envelope bridge.
-/
theorem ot_orbit_bound_from_separable_threeCmax_and_blockConditions_zeroStart_of_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hPsi_is_sweep : Psi = sweep Ψ₁ Ψ₂)
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max U : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (n : ℕ) :
    variationSeminorm ((Psi^[n]) (fun _ : ι₁ => (0 : ℝ))) ≤ 2 * U := by
  rw [hPsi_is_sweep] at hfix ⊢
  exact ot_orbit_bound_from_separable_threeCmax_and_blockConditions_zeroStart_sweep_of_upperConstant
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix j₀ hY
    hgamma hmin_b hC_max hbudget_three hU n

/--
Alias-form successor-step zero-start OT orbit bound from threeCmax upper-envelope bridge.
-/
theorem ot_orbit_bound_from_separable_threeCmax_and_blockConditions_zeroStart_succ_of_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hPsi_is_sweep : Psi = sweep Ψ₁ Ψ₂)
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max U : ℝ}
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (n : ℕ) :
    variationSeminorm ((Psi^[n + 1]) (fun _ : ι₁ => (0 : ℝ))) ≤ 2 * U := by
  rw [hPsi_is_sweep] at hfix ⊢
  exact ot_orbit_bound_from_separable_threeCmax_and_blockConditions_zeroStart_sweep_succ_of_upperConstant
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix j₀ hY
    hbudget_three hU n

/--
Index-form `ε`-accuracy from a closed-form upper-envelope rate constant.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  have hmasterU : ∀ m : ℕ, gap m ≤ alpha_rate * (2 * U) / (m + 1 : ℝ) := by
    intro m
    have hm := hmaster m
    have hnum : alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) ≤
        alpha_rate * (2 * U) :=
      mul_le_mul_of_nonneg_left hupper halpha
    have hdiv : alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) /
        (m + 1 : ℝ) ≤ alpha_rate * (2 * U) / (m + 1 : ℝ) :=
      div_le_div_of_nonneg_right hnum (by positivity)
    exact hm.trans hdiv
  exact DualConvergence.dualRate_iterationThreshold_of_closedFormCeil
    (gap := gap) (C := alpha_rate * (2 * U)) (eps := eps) hmasterU heps n hn

/--
Successor-index `ε`-accuracy from a closed-form upper-envelope rate constant.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil_succ
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ (n + 1) + 1) :
    gap (n + 1) ≤ eps := by
  have hmasterU : ∀ m : ℕ, gap m ≤ alpha_rate * (2 * U) / (m + 1 : ℝ) := by
    intro m
    have hm := hmaster m
    have hnum : alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) ≤
        alpha_rate * (2 * U) :=
      mul_le_mul_of_nonneg_left hupper halpha
    have hdiv : alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) /
        (m + 1 : ℝ) ≤ alpha_rate * (2 * U) / (m + 1 : ℝ) :=
      div_le_div_of_nonneg_right hnum (by positivity)
    exact hm.trans hdiv
  exact DualConvergence.dualRate_iterationThreshold_of_closedFormCeil
    (gap := gap) (C := alpha_rate * (2 * U)) (eps := eps) hmasterU heps (n + 1) hn

/--
Index-form closed-form-ceil OT `ε`-accuracy from a natural-number index upper bound.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil_of_natBound
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ) (m : ℕ)
    (hnm : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ m)
    (hmn : m ≤ n + 1) :
    gap n ≤ eps := by
  exact ot_epsilonAccuracy_from_upperConstant_closedFormCeil
    halpha heps hupper hmaster n (le_trans hnm hmn)

/--
Successor-index closed-form-ceil OT `ε`-accuracy from a natural-number index upper bound.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil_succ_of_natBound
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ) (m : ℕ)
    (hnm : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ m)
    (hmn : m ≤ (n + 1) + 1) :
    gap (n + 1) ≤ eps := by
  exact ot_epsilonAccuracy_from_upperConstant_closedFormCeil_succ
    halpha heps hupper hmaster n (le_trans hnm hmn)

/--
Index-form `ε`-accuracy from a threeCmax upper envelope, routed through the
upper-constant closed-form-ceil wrapper.
-/
theorem ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper_three : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  have hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U := by nlinarith
  exact ot_epsilonAccuracy_from_upperConstant_closedFormCeil
    halpha heps hupper hmaster n hn

/--
Successor-index `ε`-accuracy from a threeCmax upper envelope, routed through the
upper-constant closed-form-ceil wrapper.
-/
theorem ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_succ
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper_three : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ (n + 1) + 1) :
    gap (n + 1) ≤ eps := by
  have hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U := by nlinarith
  exact ot_epsilonAccuracy_from_upperConstant_closedFormCeil_succ
    halpha heps hupper hmaster n hn

/--
Index-form `ε`-accuracy from a threeCmax upper envelope and a natural-number
bound on the closed-form ceiling threshold.
-/
theorem ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_of_natBound
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper_three : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ) (m : ℕ)
    (hnm : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ m)
    (hmn : m ≤ n + 1) :
    gap n ≤ eps := by
  exact ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil
    halpha heps hupper_three hmaster n (le_trans hnm hmn)

/--
Index-monotone convenience form of upper-constant closed-form-ceil `ε`-accuracy.

If the closed-form threshold is known at index `n`, then it also certifies all `m ≥ n`.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil_of_ge_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ n + 1)
    (m : ℕ)
    (hnm : n ≤ m) :
    gap m ≤ eps := by
  exact ot_epsilonAccuracy_from_upperConstant_closedFormCeil
    halpha heps hupper hmaster m (le_trans hn (Nat.succ_le_succ hnm))

/--
Index-monotone convenience form of threeCmax upper-envelope closed-form-ceil `ε`-accuracy.
-/
theorem ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_of_ge_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper_three : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ n + 1)
    (m : ℕ)
    (hnm : n ≤ m) :
    gap m ≤ eps := by
  exact ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil
    halpha heps hupper_three hmaster m (le_trans hn (Nat.succ_le_succ hnm))

/--
Successor-index `ε`-accuracy from a threeCmax upper envelope and a natural-number
bound on the closed-form ceiling threshold.
-/
theorem ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_succ_of_natBound
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper_three : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ) (m : ℕ)
    (hnm : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ m)
    (hmn : m ≤ (n + 1) + 1) :
    gap (n + 1) ≤ eps := by
  exact ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_succ
    halpha heps hupper_three hmaster n (le_trans hnm hmn)

/--
Closed-form-ceil OT `ε`-accuracy evaluated at its explicit ceiling index.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil_at_ceil_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ)) :
    gap (Nat.ceil (alpha_rate * (2 * U) / eps)) ≤ eps := by
  exact ot_epsilonAccuracy_from_upperConstant_closedFormCeil
    halpha heps hupper hmaster
    (Nat.ceil (alpha_rate * (2 * U) / eps))
    (Nat.le_add_right _ 1)

/--
Successor-index closed-form-ceil OT `ε`-accuracy evaluated at the explicit ceiling index.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil_succ_at_ceil_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ)) :
    gap (Nat.ceil (alpha_rate * (2 * U) / eps) + 1) ≤ eps := by
  let N : ℕ := Nat.ceil (alpha_rate * (2 * U) / eps)
  have hN : N ≤ (N + 1) + 1 := by
    exact Nat.le_trans (Nat.le_add_right N 1) (Nat.le_add_right (N + 1) 1)
  simpa [N] using
    (ot_epsilonAccuracy_from_upperConstant_closedFormCeil_succ
      halpha heps hupper hmaster N hN)

/--
ThreeCmax closed-form-ceil OT `ε`-accuracy evaluated at its explicit ceiling index.
-/
theorem ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_at_ceil_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper_three : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ)) :
    gap (Nat.ceil (alpha_rate * (2 * U) / eps)) ≤ eps := by
  exact ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil
    halpha heps hupper_three hmaster
    (Nat.ceil (alpha_rate * (2 * U) / eps))
    (Nat.le_add_right _ 1)

/--
Successor-index threeCmax closed-form-ceil OT `ε`-accuracy evaluated at the explicit ceiling index.
-/
theorem ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_succ_at_ceil_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper_three : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ)) :
    gap (Nat.ceil (alpha_rate * (2 * U) / eps) + 1) ≤ eps := by
  let N : ℕ := Nat.ceil (alpha_rate * (2 * U) / eps)
  have hN : N ≤ (N + 1) + 1 := by
    exact Nat.le_trans (Nat.le_add_right N 1) (Nat.le_add_right (N + 1) 1)
  simpa [N] using
    (ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_succ
      halpha heps hupper_three hmaster N hN)

/--
Successor-index index-monotone OT `ε`-accuracy from an upper-constant closed-form ceiling threshold.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil_succ_of_ge_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ n + 1)
    (m : ℕ)
    (hnm : n ≤ m) :
    gap (m + 1) ≤ eps := by
  have hnm' : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ m + 1 := by
    exact le_trans hn (Nat.succ_le_succ hnm)
  exact ot_epsilonAccuracy_from_upperConstant_closedFormCeil_succ
    halpha heps hupper hmaster m (le_trans hnm' (Nat.le_add_right (m + 1) 1))

/--
Successor-index index-monotone OT `ε`-accuracy from a threeCmax closed-form ceiling threshold.
-/
theorem ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_succ_of_ge_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper_three : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ n + 1)
    (m : ℕ)
    (hnm : n ≤ m) :
    gap (m + 1) ≤ eps := by
  have hnm' : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ m + 1 := by
    exact le_trans hn (Nat.succ_le_succ hnm)
  exact ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_succ
    halpha heps hupper_three hmaster m (le_trans hnm' (Nat.le_add_right (m + 1) 1))

/--
Nat-bound specialization of upper-constant closed-form-ceil OT `ε`-accuracy at the explicit ceiling index.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil_of_natBound_at_ceil_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  exact ot_epsilonAccuracy_from_upperConstant_closedFormCeil_of_natBound
    halpha heps hupper hmaster n
    (Nat.ceil (alpha_rate * (2 * U) / eps))
    (Nat.le_refl _)
    hn

/--
Nat-bound specialization of threeCmax closed-form-ceil OT `ε`-accuracy at the explicit ceiling index.
-/
theorem ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_of_natBound_at_ceil_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper_three : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  exact ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_of_natBound
    halpha heps hupper_three hmaster n
    (Nat.ceil (alpha_rate * (2 * U) / eps))
    (Nat.le_refl _)
    hn

/--
Threshold-transport form of upper-constant closed-form-ceil OT `ε`-accuracy.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil_of_threshold_le
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k n : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k)
    (hkn : k ≤ n + 1) :
    gap n ≤ eps := by
  exact ot_epsilonAccuracy_from_upperConstant_closedFormCeil_of_natBound
    halpha heps hupper hmaster n k hk hkn

/--
Threshold-transport form of threeCmax closed-form-ceil OT `ε`-accuracy.
-/
theorem ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_of_threshold_le
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper_three : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k n : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k)
    (hkn : k ≤ n + 1) :
    gap n ≤ eps := by
  exact ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_of_natBound
    halpha heps hupper_three hmaster n k hk hkn

/--
Threshold-transport form of successor-index upper-constant closed-form-ceil OT `ε`-accuracy.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil_succ_of_threshold_le
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k n : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k)
    (hkn : k ≤ (n + 1) + 1) :
    gap (n + 1) ≤ eps := by
  exact ot_epsilonAccuracy_from_upperConstant_closedFormCeil_succ_of_natBound
    halpha heps hupper hmaster n k hk hkn

/--
Threshold-transport form of successor-index threeCmax closed-form-ceil OT `ε`-accuracy.
-/
theorem ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_succ_of_threshold_le
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper_three : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k n : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k)
    (hkn : k ≤ (n + 1) + 1) :
    gap (n + 1) ≤ eps := by
  exact ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_succ_of_natBound
    halpha heps hupper_three hmaster n k hk hkn

/--
Monotone threshold-transport form of upper-constant closed-form-ceil OT `ε`-accuracy.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil_of_threshold_le_and_ge_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k n m : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k)
    (hkn : k ≤ n + 1)
    (hnm : n ≤ m) :
    gap m ≤ eps := by
  exact ot_epsilonAccuracy_from_upperConstant_closedFormCeil_of_ge_index
    halpha heps hupper hmaster n (le_trans hk hkn) m hnm

/--
Monotone threshold-transport form of threeCmax closed-form-ceil OT `ε`-accuracy.
-/
theorem ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_of_threshold_le_and_ge_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper_three : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k n m : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k)
    (hkn : k ≤ n + 1)
    (hnm : n ≤ m) :
    gap m ≤ eps := by
  exact ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_of_ge_index
    halpha heps hupper_three hmaster n (le_trans hk hkn) m hnm

/--
Monotone threshold-transport form of successor-index upper-constant closed-form-ceil OT `ε`-accuracy.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil_succ_of_threshold_le_and_ge_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k n m : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k)
    (hkn : k ≤ n + 1)
    (hnm : n ≤ m) :
    gap (m + 1) ≤ eps := by
  exact ot_epsilonAccuracy_from_upperConstant_closedFormCeil_succ_of_ge_index
    halpha heps hupper hmaster n (le_trans hk hkn) m hnm

/--
Monotone threshold-transport form of successor-index threeCmax closed-form-ceil OT `ε`-accuracy.
-/
theorem ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_succ_of_threshold_le_and_ge_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper_three : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k n m : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k)
    (hkn : k ≤ n + 1)
    (hnm : n ≤ m) :
    gap (m + 1) ≤ eps := by
  exact ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_succ_of_ge_index
    halpha heps hupper_three hmaster n (le_trans hk hkn) m hnm

/--
At-ceiling-index threshold-transport form of upper-constant closed-form-ceil OT `ε`-accuracy.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil_of_threshold_le_at_ceil_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k) :
    gap (k - 1) ≤ eps := by
  have hkn : k ≤ (k - 1) + 1 := by
    cases k with
    | zero => simp
    | succ k => simp
  exact ot_epsilonAccuracy_from_upperConstant_closedFormCeil_of_threshold_le
    halpha heps hupper hmaster k (k - 1) hk hkn

/--
At-ceiling-index threshold-transport form of threeCmax closed-form-ceil OT `ε`-accuracy.
-/
theorem ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_of_threshold_le_at_ceil_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper_three : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k) :
    gap (k - 1) ≤ eps := by
  have hkn : k ≤ (k - 1) + 1 := by
    cases k with
    | zero => simp
    | succ k => simp
  exact ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_of_threshold_le
    halpha heps hupper_three hmaster k (k - 1) hk hkn

/--
Successor at-ceiling-index threshold-transport form of upper-constant closed-form-ceil OT
`ε`-accuracy.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil_succ_of_threshold_le_at_ceil_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k) :
    gap ((k - 1) + 1) ≤ eps := by
  have hkn : k ≤ (k - 1) + 1 := by
    cases k with
    | zero => simp
    | succ k => simp
  have hkn_succ : k ≤ ((k - 1) + 1) + 1 := by
    exact le_trans hkn (Nat.le_add_right ((k - 1) + 1) 1)
  exact ot_epsilonAccuracy_from_upperConstant_closedFormCeil_succ_of_threshold_le
    halpha heps hupper hmaster k (k - 1) hk hkn_succ

/--
Successor at-ceiling-index threshold-transport form of threeCmax closed-form-ceil OT
`ε`-accuracy.
-/
theorem
    ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_succ_of_threshold_le_at_ceil_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper_three : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k) :
    gap ((k - 1) + 1) ≤ eps := by
  have hkn : k ≤ (k - 1) + 1 := by
    cases k with
    | zero => simp
    | succ k => simp
  have hkn_succ : k ≤ ((k - 1) + 1) + 1 := by
    exact le_trans hkn (Nat.le_add_right ((k - 1) + 1) 1)
  exact ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_succ_of_threshold_le
    halpha heps hupper_three hmaster k (k - 1) hk hkn_succ

/--
Monotone-index transport from an at-ceiling threshold certificate for upper-constant
closed-form-ceil OT `ε`-accuracy.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil_of_threshold_le_at_ceil_index_of_ge_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k m : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k)
    (hkm : k - 1 ≤ m) :
    gap m ≤ eps := by
  have hkn : k ≤ (k - 1) + 1 := by
    cases k with
    | zero => simp
    | succ k => simp
  exact ot_epsilonAccuracy_from_upperConstant_closedFormCeil_of_threshold_le_and_ge_index
    halpha heps hupper hmaster k (k - 1) m hk hkn hkm

/--
Monotone-index transport from an at-ceiling threshold certificate for threeCmax
closed-form-ceil OT `ε`-accuracy.
-/
theorem ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_of_threshold_le_at_ceil_index_of_ge_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper_three : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k m : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k)
    (hkm : k - 1 ≤ m) :
    gap m ≤ eps := by
  have hkn : k ≤ (k - 1) + 1 := by
    cases k with
    | zero => simp
    | succ k => simp
  exact ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_of_threshold_le_and_ge_index
    halpha heps hupper_three hmaster k (k - 1) m hk hkn hkm

/--
Monotone successor-index transport from an at-ceiling threshold certificate for upper-constant
closed-form-ceil OT `ε`-accuracy.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil_succ_of_threshold_le_at_ceil_index_of_ge_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k m : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k)
    (hkm : k - 1 ≤ m) :
    gap (m + 1) ≤ eps := by
  have hkn : k ≤ (k - 1) + 1 := by
    cases k with
    | zero => simp
    | succ k => simp
  exact ot_epsilonAccuracy_from_upperConstant_closedFormCeil_succ_of_threshold_le_and_ge_index
    halpha heps hupper hmaster k (k - 1) m hk hkn hkm

/--
Monotone successor-index transport from an at-ceiling threshold certificate for threeCmax
closed-form-ceil OT `ε`-accuracy.
-/
theorem
    ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_succ_of_threshold_le_at_ceil_index_of_ge_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper_three : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k m : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k)
    (hkm : k - 1 ≤ m) :
    gap (m + 1) ≤ eps := by
  have hkn : k ≤ (k - 1) + 1 := by
    cases k with
    | zero => simp
    | succ k => simp
  exact ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_succ_of_threshold_le_and_ge_index
    halpha heps hupper_three hmaster k (k - 1) m hk hkn hkm

/--
Natural-bound transport from an at-ceiling threshold certificate for upper-constant
closed-form-ceil OT `ε`-accuracy.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil_of_threshold_le_at_ceil_index_of_natBound
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k N : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k)
    (hNk : k - 1 ≤ N) :
    gap N ≤ eps := by
  exact ot_epsilonAccuracy_from_upperConstant_closedFormCeil_of_threshold_le_at_ceil_index_of_ge_index
    halpha heps hupper hmaster k N hk hNk

/--
Natural-bound transport from an at-ceiling threshold certificate for threeCmax
closed-form-ceil OT `ε`-accuracy.
-/
theorem ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_of_threshold_le_at_ceil_index_of_natBound
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper_three : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k N : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k)
    (hNk : k - 1 ≤ N) :
    gap N ≤ eps := by
  exact ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_of_threshold_le_at_ceil_index_of_ge_index
    halpha heps hupper_three hmaster k N hk hNk

/--
Natural-bound transport from an at-ceiling threshold certificate for successor-index
upper-constant closed-form-ceil OT `ε`-accuracy.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil_succ_of_threshold_le_at_ceil_index_of_natBound
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k N : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k)
    (hNk : k - 1 ≤ N) :
    gap (N + 1) ≤ eps := by
  exact ot_epsilonAccuracy_from_upperConstant_closedFormCeil_succ_of_threshold_le_at_ceil_index_of_ge_index
    halpha heps hupper hmaster k N hk hNk

/--
Natural-bound transport from an at-ceiling threshold certificate for successor-index
threeCmax closed-form-ceil OT `ε`-accuracy.
-/
theorem ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_succ_of_threshold_le_at_ceil_index_of_natBound
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper_three : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k N : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k)
    (hNk : k - 1 ≤ N) :
    gap (N + 1) ≤ eps := by
  exact ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_succ_of_threshold_le_at_ceil_index_of_ge_index
    halpha heps hupper_three hmaster k N hk hNk

/--
Natural-bound then monotone-index transport for at-ceiling threshold upper-constant
closed-form-ceil OT `ε`-accuracy.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil_of_threshold_le_at_ceil_index_of_natBound_of_ge_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k N m : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k)
    (hNk : k - 1 ≤ N)
    (hNm : N ≤ m) :
    gap m ≤ eps := by
  exact ot_epsilonAccuracy_from_upperConstant_closedFormCeil_of_threshold_le_at_ceil_index_of_ge_index
    halpha heps hupper hmaster k m hk (le_trans hNk hNm)

/--
Natural-bound then monotone-index transport for at-ceiling threshold threeCmax
closed-form-ceil OT `ε`-accuracy.
-/
theorem ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_of_threshold_le_at_ceil_index_of_natBound_of_ge_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper_three : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k N m : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k)
    (hNk : k - 1 ≤ N)
    (hNm : N ≤ m) :
    gap m ≤ eps := by
  exact ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_of_threshold_le_at_ceil_index_of_ge_index
    halpha heps hupper_three hmaster k m hk (le_trans hNk hNm)

/--
Natural-bound then monotone-index transport for at-ceiling threshold successor-index
upper-constant closed-form-ceil OT `ε`-accuracy.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil_succ_of_threshold_le_at_ceil_index_of_natBound_of_ge_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k N m : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k)
    (hNk : k - 1 ≤ N)
    (hNm : N ≤ m) :
    gap (m + 1) ≤ eps := by
  exact ot_epsilonAccuracy_from_upperConstant_closedFormCeil_succ_of_threshold_le_at_ceil_index_of_ge_index
    halpha heps hupper hmaster k m hk (le_trans hNk hNm)

/--
Natural-bound then monotone-index transport for at-ceiling threshold successor-index
threeCmax closed-form-ceil OT `ε`-accuracy.
-/
theorem
    ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_succ_of_threshold_le_at_ceil_index_of_natBound_of_ge_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper_three : 3 * C_max + gamma * |Real.log min_b| ≤ U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k N m : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k)
    (hNk : k - 1 ≤ N)
    (hNm : N ≤ m) :
    gap (m + 1) ≤ eps := by
  exact ot_epsilonAccuracy_from_threeCmax_upperConstant_closedFormCeil_succ_of_threshold_le_at_ceil_index_of_ge_index
    halpha heps hupper_three hmaster k m hk (le_trans hNk hNm)

/--
Threshold-index specialization of natural-bound upper-constant at-ceiling threshold transport.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil_of_threshold_le_at_ceil_index_of_natBound_at_threshold_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k) :
    gap (k - 1) ≤ eps := by
  exact ot_epsilonAccuracy_from_upperConstant_closedFormCeil_of_threshold_le_at_ceil_index_of_natBound
    halpha heps hupper hmaster k (k - 1) hk (Nat.le_refl _)

/--
Successor threshold-index specialization of natural-bound upper-constant at-ceiling threshold
transport.
-/
theorem ot_epsilonAccuracy_from_upperConstant_closedFormCeil_succ_of_threshold_le_at_ceil_index_of_natBound_at_threshold_index
    {gap : ℕ → ℝ}
    {alpha_rate C_max gamma min_b U eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (heps : 0 < eps)
    (hupper : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ 2 * U)
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha_rate * (6 * C_max + 2 * gamma * |Real.log min_b|) / (n + 1 : ℝ))
    (k : ℕ)
    (hk : Nat.ceil (alpha_rate * (2 * U) / eps) ≤ k) :
    gap ((k - 1) + 1) ≤ eps := by
  exact ot_epsilonAccuracy_from_upperConstant_closedFormCeil_succ_of_threshold_le_at_ceil_index_of_natBound
    halpha heps hupper hmaster k (k - 1) hk (Nat.le_refl _)

end OT
end Applications
end KLProjection
end FlowSinkhorn
