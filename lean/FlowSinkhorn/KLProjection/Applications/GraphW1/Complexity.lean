import FlowSinkhorn.KLProjection.Applications.GraphW1.Kappa
import FlowSinkhorn.KLProjection.Applications.GraphW1.ComplexityVocabulary
import FlowSinkhorn.KLProjection.Setup.BlockMonotonicity

/-!
# Graph `W₁` complexity instantiation

This module is reserved for Corollary `cor:W1-XmaxUmax` and the resulting explicit complexity
statements from the graph-W1 material in `neurips/paper.tex`.

Intended theorem names:
- `graphW1_explicit_XGamma_UGamma`;
- `graphW1_dualRate`;
- `graphW1_iterationComplexity`.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace Applications
namespace GraphW1

-- Paper-facing theorem names are intentionally verbose for traceability.
set_option linter.style.longLine false

/--
Concrete graph-`W₁` `X_γ/U_γ` corollary endpoint.

This is the paper-facing uniform orbit bound from `cor:W1-XmaxUmax` with explicit
constant `U_max = 4 * diam * (cost + log n_nodes)`.
-/
theorem graphW1_explicit_XGamma_UGamma
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n_nodes : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
        PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n_nodes / gamma))
    {v₀ : ι → ℝ} (hv₀ : variationSeminorm v₀ = 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v₀) ≤ 4 * diam * (cost + Real.log n_nodes) := by
  have horbit :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint
      variationSeminormAsSeminorm Psi hPsi (uStar := vStar) (u0 := v₀) hfix k
  have horbit' : variationSeminorm ((Psi^[k]) v₀) ≤
      variationSeminorm v₀ + 2 * variationSeminorm vStar := horbit
  rw [hv₀, zero_add] at horbit'
  calc
    variationSeminorm ((Psi^[k]) v₀)
        ≤ 2 * variationSeminorm vStar := by linarith
    _ ≤ 2 * PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma
            (Real.log n_nodes / gamma) := by
          have := variationSeminorm_nonneg vStar
          nlinarith
    _ = 4 * diam * (cost + Real.log n_nodes) := by
          exact graphW1_Umax_twoTimesHGammaBudget_diam hgamma

/--
Paper-facing package for Corollary `app-cor:graphw1-xgamma-ugamma`.

The witness records the displayed graph-W₁ choices
`U_γ = 4 * diam * (lengthMax + γ * H_γ)` and
`X_γ = ‖b‖₁ * U_γ / γ + p * exp (-lengthMin / γ)`.
The orbit bound is derived from the two-step path certificate and the fixed-point budget; the
primal confinement bound uses the pointwise mass estimate supplied by Proposition
`prop:mass-bound-block`.
-/
theorem graphW1_XGamma_UGamma_bounds_from_twoStep_path
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B lengthMax gamma hGamma bMass p lengthMin : ℝ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbMass : 0 ≤ bMass)
    (hbase_nonneg : 0 ≤ lengthMax + gamma * hGamma)
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa lengthMax gamma hGamma)
    (xMass : ℕ → ℝ)
    (hmass :
      ∀ k : ℕ,
        xMass k ≤
          bMass * variationSeminorm ((Psi^[k]) (0 : ι → ℝ)) / gamma +
            p * Real.exp (-lengthMin / gamma)) :
    ∃ U_gamma X_gamma : ℝ,
      U_gamma = 4 * (graphDiam : ℝ) * (lengthMax + gamma * hGamma) ∧
      X_gamma = bMass * U_gamma / gamma + p * Real.exp (-lengthMin / gamma) ∧
      ∀ k : ℕ,
        variationSeminorm ((Psi^[k]) (0 : ι → ℝ)) ≤ U_gamma ∧
          xMass k ≤ X_gamma := by
  let U_gamma := 4 * (graphDiam : ℝ) * (lengthMax + gamma * hGamma)
  let X_gamma := bMass * U_gamma / gamma + p * Real.exp (-lengthMin / gamma)
  refine ⟨U_gamma, X_gamma, rfl, rfl, ?_⟩
  intro k
  have hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi :=
    SeminormNonexpansive_variationSeminormAsSeminorm_of_isTopical hT
  have hiter :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint
      variationSeminormAsSeminorm Psi hPsi (uStar := vStar)
      (u0 := (0 : ι → ℝ)) hfix k
  have hzero : variationSeminorm (0 : ι → ℝ) = 0 := variationSeminorm_zero
  have hiter' :
      variationSeminorm ((Psi^[k]) (0 : ι → ℝ)) ≤
        variationSeminorm (0 : ι → ℝ) + 2 * variationSeminorm vStar :=
    hiter
  have horbit_to_star :
      variationSeminorm ((Psi^[k]) (0 : ι → ℝ)) ≤ 2 * variationSeminorm vStar := by
    linarith
  have hbudget_explicit :
      PrimalDualBounds.hGammaKappaBudget kappa lengthMax gamma hGamma ≤
        2 * (graphDiam : ℝ) * (lengthMax + gamma * hGamma) :=
    graphW1_hGammaBudget_le_explicit_twoDiam_of_twoStep_path
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path hbase_nonneg
  have horbit : variationSeminorm ((Psi^[k]) (0 : ι → ℝ)) ≤ U_gamma := by
    calc
      variationSeminorm ((Psi^[k]) (0 : ι → ℝ))
          ≤ 2 * variationSeminorm vStar := horbit_to_star
      _ ≤ 2 * PrimalDualBounds.hGammaKappaBudget kappa lengthMax gamma hGamma := by
            exact mul_le_mul_of_nonneg_left hvStar (by norm_num)
      _ ≤ 2 * (2 * (graphDiam : ℝ) * (lengthMax + gamma * hGamma)) := by
            exact mul_le_mul_of_nonneg_left hbudget_explicit (by norm_num)
      _ = U_gamma := by
            dsimp [U_gamma]
            ring
  refine ⟨horbit, ?_⟩
  have hmass_point := hmass k
  have hmul :
      bMass * variationSeminorm ((Psi^[k]) (0 : ι → ℝ)) ≤ bMass * U_gamma :=
    mul_le_mul_of_nonneg_left horbit hbMass
  have hdiv :
      bMass * variationSeminorm ((Psi^[k]) (0 : ι → ℝ)) / gamma ≤
        bMass * U_gamma / gamma :=
    div_le_div_of_nonneg_right hmul (le_of_lt hgamma)
  dsimp [X_gamma]
  linarith

/--
Block-condition version of Corollary `app-cor:graphw1-xgamma-ugamma`.

Compared with `graphW1_XGamma_UGamma_bounds_from_twoStep_path`, this endpoint no longer asks the
caller for a prepackaged `IsTopical` certificate.  It starts from the two graph-`W₁` block maps,
uses their monotonicity and signed translation-equivariance laws to obtain the full-sweep orbit
bound, then combines that bound with the pointwise primal-mass proxy.
-/
theorem graphW1_XGamma_UGamma_bounds_from_blockConditions_twoStep_path
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {vStar : ι₁ → ℝ} (hfix : sweep Ψ₁ Ψ₂ vStar = vStar)
    {kappa B lengthMax gamma hGamma bMass p lengthMin : ℝ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι₁ × ι₁ → ℝ)
    (hyf : ∀ p : ι₁ × ι₁, |yf p| ≤ B)
    (hyg : ∀ p : ι₁ × ι₁, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι₁ × ι₁, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbMass : 0 ≤ bMass)
    (hbase_nonneg : 0 ≤ lengthMax + gamma * hGamma)
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa lengthMax gamma hGamma)
    (xMass : ℕ → ℝ)
    (hmass :
      ∀ k : ℕ,
        xMass k ≤
          bMass * variationSeminorm (((sweep Ψ₁ Ψ₂)^[k]) (0 : ι₁ → ℝ)) / gamma +
            p * Real.exp (-lengthMin / gamma)) :
    ∃ U_gamma X_gamma : ℝ,
      U_gamma = 4 * (graphDiam : ℝ) * (lengthMax + gamma * hGamma) ∧
      X_gamma = bMass * U_gamma / gamma + p * Real.exp (-lengthMin / gamma) ∧
      ∀ k : ℕ,
        variationSeminorm (((sweep Ψ₁ Ψ₂)^[k]) (0 : ι₁ → ℝ)) ≤ U_gamma ∧
          xMass k ≤ X_gamma := by
  let U_gamma := 4 * (graphDiam : ℝ) * (lengthMax + gamma * hGamma)
  let X_gamma := bMass * U_gamma / gamma + p * Real.exp (-lengthMin / gamma)
  refine ⟨U_gamma, X_gamma, rfl, rfl, ?_⟩
  intro k
  have hbudget_explicit :
      PrimalDualBounds.hGammaKappaBudget kappa lengthMax gamma hGamma ≤
        2 * (graphDiam : ℝ) * (lengthMax + gamma * hGamma) :=
    graphW1_hGammaBudget_le_explicit_twoDiam_of_twoStep_path
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path hbase_nonneg
  have horbit_setup :
      variationSeminorm (((sweep Ψ₁ Ψ₂)^[k]) (0 : ι₁ → ℝ)) ≤
        variationSeminorm (0 : ι₁ → ℝ) + 2 * variationSeminorm vStar :=
    Setup.sweep_orbit_bound_of_blockConditions
      (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
      hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix k
  have hvStar_explicit :
      variationSeminorm vStar ≤ 2 * (graphDiam : ℝ) * (lengthMax + gamma * hGamma) :=
    hvStar.trans hbudget_explicit
  have horbit : variationSeminorm (((sweep Ψ₁ Ψ₂)^[k]) (0 : ι₁ → ℝ)) ≤ U_gamma := by
    calc
      variationSeminorm (((sweep Ψ₁ Ψ₂)^[k]) (0 : ι₁ → ℝ))
          ≤ variationSeminorm (0 : ι₁ → ℝ) + 2 * variationSeminorm vStar := horbit_setup
      _ = 2 * variationSeminorm vStar := by
            rw [variationSeminorm_zero, zero_add]
      _ ≤ 2 * (2 * (graphDiam : ℝ) * (lengthMax + gamma * hGamma)) := by
            nlinarith
      _ = U_gamma := by
            dsimp [U_gamma]
            ring
  refine ⟨horbit, ?_⟩
  have hmass_point := hmass k
  have hmul :
      bMass * variationSeminorm (((sweep Ψ₁ Ψ₂)^[k]) (0 : ι₁ → ℝ)) ≤ bMass * U_gamma :=
    mul_le_mul_of_nonneg_left horbit hbMass
  have hdiv :
      bMass * variationSeminorm (((sweep Ψ₁ Ψ₂)^[k]) (0 : ι₁ → ℝ)) / gamma ≤
        bMass * U_gamma / gamma :=
    div_le_div_of_nonneg_right hmul (le_of_lt hgamma)
  dsimp [X_gamma]
  linarith

/--
Structured-certificate version of Corollary `app-cor:graphw1-xgamma-ugamma`.

The theorem exposes the same mathematical data as
`graphW1_XGamma_UGamma_bounds_from_blockConditions_twoStep_path`, but groups the block laws,
fixed-point budget, bounded edge fields, path witness, and mass proxy into named records.
This makes the paper-facing Comparator statement easier to audit without changing the proof core.
-/
theorem graphW1_XGamma_UGamma_bounds_from_structuredCertificates_twoStep_path
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (block : SignedBlockSweepData ι₁ ι₂)
    {kappa lengthMax gamma hGamma bMass p lengthMin : ℝ}
    (fixed :
      SweepFixedPointBudget ι₁ (sweep block.Ψ₁ block.Ψ₂) kappa lengthMax gamma hGamma)
    (edge : UnitBoundedTwoStepFields ι₁)
    (graphDiam : ℕ)
    (path : TwoStepPathCertificate edge graphDiam kappa)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ lengthMax + gamma * hGamma)
    (mass :
      GraphW1MassProxy ι₁ (sweep block.Ψ₁ block.Ψ₂) gamma bMass p lengthMin) :
    ∃ U_gamma X_gamma : ℝ,
      U_gamma = 4 * (graphDiam : ℝ) * (lengthMax + gamma * hGamma) ∧
      X_gamma = bMass * U_gamma / gamma + p * Real.exp (-lengthMin / gamma) ∧
      ∀ k : ℕ,
        variationSeminorm (((sweep block.Ψ₁ block.Ψ₂)^[k]) (0 : ι₁ → ℝ)) ≤ U_gamma ∧
          mass.xMass k ≤ X_gamma := by
  have hB_nonneg : 0 ≤ edge.B := by
    rcases (inferInstance : Nonempty ι₁) with ⟨i₀⟩
    exact (abs_nonneg (edge.yf (i₀, i₀))).trans (edge.yf_bound (i₀, i₀))
  exact
    graphW1_XGamma_UGamma_bounds_from_blockConditions_twoStep_path
      (τ := block.τ)
      (Ψ₁ := block.Ψ₁)
      (Ψ₂ := block.Ψ₂)
      block.mono₁ block.mono₂ block.trans₁ block.trans₂
      (vStar := fixed.vStar) fixed.fixed
      (kappa := kappa)
      (B := edge.B)
      (lengthMax := lengthMax)
      (gamma := gamma)
      (hGamma := hGamma)
      (bMass := bMass)
      (p := p)
      (lengthMin := lengthMin)
      hB_nonneg edge.le_one graphDiam edge.yf edge.yg edge.yf_bound edge.yg_bound
      path.steps path.length_le path.steps_from_edge path.kappa_le_abs_sum
      hgamma mass.bMass_nonneg hbase_nonneg fixed.budget mass.xMass mass.pointwise

/--
Graph-`W₁` paper-facing dual-rate wrapper.
-/
theorem graphW1_dualRate
    {phi gap residual : ℕ → ℝ}
    {alpha B : ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (n : ℕ) :
    gap n ≤ (alpha * B) / (n + 1 : ℝ) :=
  PrimalDualBounds.genericBlueprint_dualRate
    halpha hgap_res hres_ascent hphi_bound hmono_gap n

/--
Graph-`W₁` complexity wrapper in closed-form stopping-rule shape.
-/
theorem graphW1_iterationComplexity
    {F0 FgammaStar bias C eps target : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hmaster : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ C / (n + 1 : ℝ))
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil (C / eps) ≤ n + 1)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma n| ≤ target :=
  DualConvergence.regularizedApproximation_complexity_of_closedFormIterationThreshold
    (F0 := F0) (FgammaStar := FgammaStar) (bias := bias)
    (C := C) (eps := eps) (target := target) (Fgamma := Fgamma)
    hbias hmaster heps n hn hbudget

/--
Final-target graph-W₁ objective accuracy from the Section-3 master abstract rate under
the explicit graph closed-form ceiling threshold.
-/
theorem graphW1_finalTarget_from_masterAbstractRate_closedFormCeil
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias alpha graphDiam cost eps target : ℝ}
    {n_nodes : ℕ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤
      4 * graphDiam * (cost + Real.log n_nodes))
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil ((alpha * (4 * graphDiam * (cost + Real.log n_nodes))) / eps) ≤ n + 1)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma n| ≤ target := by
  exact DualConvergence.regularizedApproximation_complexity_of_masterAbstractRate_closedFormCeil
    (phi := phi) (gap := gap) (residual := residual)
    (F0 := F0) (FgammaStar := FgammaStar) (bias := bias)
    (alpha := alpha) (B := 4 * graphDiam * (cost + Real.log n_nodes))
    (eps := eps) (target := target) (Fgamma := Fgamma)
    hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap
    heps n hn hbudget

/--
At-ceiling-index graph-W₁ final-target objective accuracy under the explicit
closed-form threshold.
-/
theorem graphW1_finalTarget_from_masterAbstractRate_closedFormCeil_at_ceil_index
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias alpha graphDiam cost eps target : ℝ}
    {n_nodes : ℕ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤
      4 * graphDiam * (cost + Real.log n_nodes))
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma (Nat.ceil ((alpha * (4 * graphDiam * (cost + Real.log n_nodes))) / eps))| ≤
      target := by
  exact graphW1_finalTarget_from_masterAbstractRate_closedFormCeil
    hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap
    heps (Nat.ceil ((alpha * (4 * graphDiam * (cost + Real.log n_nodes))) / eps))
    (Nat.le_add_right _ 1) hbudget

/--
Successor at-ceiling-index graph-W₁ final-target objective accuracy under the explicit
closed-form threshold.
-/
theorem graphW1_finalTarget_from_masterAbstractRate_closedFormCeil_succ_at_ceil_index
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias alpha graphDiam cost eps target : ℝ}
    {n_nodes : ℕ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤
      4 * graphDiam * (cost + Real.log n_nodes))
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma (Nat.ceil ((alpha * (4 * graphDiam * (cost + Real.log n_nodes))) / eps) + 1)| ≤
      target := by
  let N : ℕ := Nat.ceil ((alpha * (4 * graphDiam * (cost + Real.log n_nodes))) / eps)
  have hN : N ≤ (N + 1) + 1 := by
    exact Nat.le_trans (Nat.le_add_right N 1) (Nat.le_add_right (N + 1) 1)
  simpa [N] using
    (graphW1_finalTarget_from_masterAbstractRate_closedFormCeil
      hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap
      heps (N + 1) hN hbudget)

/--
Index-monotone graph-W₁ final-target objective accuracy from an explicit
closed-form threshold index.
-/
theorem graphW1_finalTarget_from_masterAbstractRate_closedFormCeil_of_ge_index
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias alpha graphDiam cost eps target : ℝ}
    {n_nodes : ℕ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤
      4 * graphDiam * (cost + Real.log n_nodes))
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil ((alpha * (4 * graphDiam * (cost + Real.log n_nodes))) / eps) ≤ n + 1)
    (m : ℕ)
    (hnm : n ≤ m)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma m| ≤ target := by
  have hnm' : Nat.ceil ((alpha * (4 * graphDiam * (cost + Real.log n_nodes))) / eps) ≤ m + 1 := by
    exact le_trans hn (Nat.succ_le_succ hnm)
  exact graphW1_finalTarget_from_masterAbstractRate_closedFormCeil
    hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap
    heps m hnm' hbudget

/--
Threshold-transport graph-W₁ final-target objective accuracy for the
closed-form ceiling threshold.
-/
theorem graphW1_finalTarget_from_masterAbstractRate_closedFormCeil_of_threshold_le
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias alpha graphDiam cost eps target : ℝ}
    {n_nodes : ℕ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤
      4 * graphDiam * (cost + Real.log n_nodes))
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (k n : ℕ)
    (hk : Nat.ceil ((alpha * (4 * graphDiam * (cost + Real.log n_nodes))) / eps) ≤ k)
    (hkn : k ≤ n + 1)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma n| ≤ target := by
  exact graphW1_finalTarget_from_masterAbstractRate_closedFormCeil
    hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap
    heps n (le_trans hk hkn) hbudget

/--
Successor-index threshold-transport graph-W₁ final-target objective accuracy for the
closed-form ceiling threshold.
-/
theorem graphW1_finalTarget_from_masterAbstractRate_closedFormCeil_succ_of_threshold_le
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias alpha graphDiam cost eps target : ℝ}
    {n_nodes : ℕ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤
      4 * graphDiam * (cost + Real.log n_nodes))
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (k n : ℕ)
    (hk : Nat.ceil ((alpha * (4 * graphDiam * (cost + Real.log n_nodes))) / eps) ≤ k)
    (hkn : k ≤ (n + 1) + 1)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma (n + 1)| ≤ target := by
  exact graphW1_finalTarget_from_masterAbstractRate_closedFormCeil
    hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap
    heps (n + 1) (le_trans hk hkn) hbudget

/--
Monotone threshold-transport graph-W₁ final-target objective accuracy for the
closed-form ceiling threshold.
-/
theorem graphW1_finalTarget_from_masterAbstractRate_closedFormCeil_of_threshold_le_and_ge_index
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias alpha graphDiam cost eps target : ℝ}
    {n_nodes : ℕ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤
      4 * graphDiam * (cost + Real.log n_nodes))
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (k n m : ℕ)
    (hk : Nat.ceil ((alpha * (4 * graphDiam * (cost + Real.log n_nodes))) / eps) ≤ k)
    (hkn : k ≤ n + 1)
    (hnm : n ≤ m)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma m| ≤ target := by
  exact graphW1_finalTarget_from_masterAbstractRate_closedFormCeil_of_ge_index
    hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap
    heps n (le_trans hk hkn) m hnm hbudget

/--
Monotone successor-index threshold-transport graph-W₁ final-target objective accuracy for the
closed-form ceiling threshold.
-/
theorem graphW1_finalTarget_from_masterAbstractRate_closedFormCeil_succ_of_threshold_le_and_ge_index
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias alpha graphDiam cost eps target : ℝ}
    {n_nodes : ℕ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤
      4 * graphDiam * (cost + Real.log n_nodes))
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (k n m : ℕ)
    (hk : Nat.ceil ((alpha * (4 * graphDiam * (cost + Real.log n_nodes))) / eps) ≤ k)
    (hkn : k ≤ n + 1)
    (hnm : n ≤ m)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma (m + 1)| ≤ target := by
  have hkn' : k ≤ (m + 1) + 1 := by
    exact le_trans hkn <|
      le_trans (Nat.succ_le_succ hnm) (Nat.le_add_right (m + 1) 1)
  exact graphW1_finalTarget_from_masterAbstractRate_closedFormCeil_succ_of_threshold_le
    hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap
    heps k m hk hkn' hbudget

/--
At-ceiling-index threshold-transport graph-W₁ final-target objective accuracy.
-/
theorem graphW1_finalTarget_from_masterAbstractRate_closedFormCeil_of_threshold_le_at_ceil_index
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias alpha graphDiam cost eps target : ℝ}
    {n_nodes : ℕ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤
      4 * graphDiam * (cost + Real.log n_nodes))
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil ((alpha * (4 * graphDiam * (cost + Real.log n_nodes))) / eps) ≤ n + 1)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma n| ≤ target := by
  exact graphW1_finalTarget_from_masterAbstractRate_closedFormCeil_of_threshold_le
    hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap heps
    (Nat.ceil ((alpha * (4 * graphDiam * (cost + Real.log n_nodes))) / eps)) n
    (Nat.le_refl _)
    hn
    hbudget

/--
Successor at-ceiling-index threshold-transport graph-W₁ final-target objective accuracy.
-/
theorem graphW1_finalTarget_from_masterAbstractRate_closedFormCeil_succ_of_threshold_le_at_ceil_index
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias alpha graphDiam cost eps target : ℝ}
    {n_nodes : ℕ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤
      4 * graphDiam * (cost + Real.log n_nodes))
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil ((alpha * (4 * graphDiam * (cost + Real.log n_nodes))) / eps) ≤ (n + 1) + 1)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma (n + 1)| ≤ target := by
  exact graphW1_finalTarget_from_masterAbstractRate_closedFormCeil_succ_of_threshold_le
    hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap heps
    (Nat.ceil ((alpha * (4 * graphDiam * (cost + Real.log n_nodes))) / eps)) n
    (Nat.le_refl _)
    hn
    hbudget

/--
Graph-`W₁` application-facing `eps`-accuracy endpoint using the full Section 3+4 blueprint
hypothesis package.
-/
theorem graphW1_applicationEpsilonAccuracy
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
    halpha hgap_res hres_ascent hphi_bound hmono_gap hPsi hfix
    horbit hmonoX hprimal_at_d hdual hbias hrate n k hbudget

/--
Explicit orbit budget for graph-W₁ in terms of graph diameter.

When κ ≤ graphDiam and the cost+entropy base is nonneg, the U_max orbit bound satisfies:
`0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ≤ 2 * graphDiam * (cost + gamma * hGamma)`.

This is the concrete form of the iterate bound from Corollary `cor:W1-XmaxUmax`.
-/
theorem graphW1_Umax_budget_explicit
    {kappa graphDiam cost gamma hGamma : ℝ}
    (hkappa : kappa ≤ graphDiam)
    (hnonneg : 0 ≤ cost + gamma * hGamma) :
    (0 : ℝ) + 2 * PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤
      2 * graphDiam * (cost + gamma * hGamma) := by
  simp only [PrimalDualBounds.hGammaKappaBudget]
  have h : 2 * (kappa * (cost + gamma * hGamma)) ≤ 2 * graphDiam * (cost + gamma * hGamma) := by
    have := mul_le_mul_of_nonneg_right hkappa hnonneg
    linarith
  linarith

/--
Graph-W₁ iteration complexity bound from Corollary `cor:W1-XmaxUmax`.

For κ ≤ diam, cost C, regularization γ, and entropy H_γ, the orbit bound is
`≤ diam * (C + γ * H_γ)`.
-/
theorem graphW1_iterationComplexity_kappa_diam
    {kappa diam cost gamma hGamma : ℝ}
    (hkappa : kappa ≤ diam)
    (hnonneg : 0 ≤ cost + gamma * hGamma) :
    PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤
      diam * (cost + gamma * hGamma) := by
  simp only [PrimalDualBounds.hGammaKappaBudget]
  exact mul_le_mul_of_nonneg_right hkappa hnonneg

/--
Graph-W₁ orbit U_max formula with the explicit log entropy bound.

With κ = 2 * diam and H_γ = log(n)/γ:
  2 * hGammaKappaBudget (2*diam) cost gamma (log n / gamma) = 4 * diam * (cost + log n).

This is the explicit orbit bound from Corollary `cor:W1-XmaxUmax`.
-/
theorem graphW1_Umax_log
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma) :
    2 * PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma) =
      4 * diam * (cost + Real.log n) := by
  rw [graphW1_Umax_explicit hgamma]
  ring

/--
Graph-W₁ orbit budget is nonneg when cost ≥ 0, diam ≥ 0, and log n ≥ 0 (n ≥ 1).
-/
theorem graphW1_Umax_log_nonneg
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hn : 1 ≤ n) :
    0 ≤ PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma) := by
  unfold PrimalDualBounds.hGammaKappaBudget
  apply mul_nonneg
  · linarith
  · apply add_nonneg hcost
    apply mul_nonneg (le_of_lt hgamma)
    exact graphW1_HGamma_formula_nonneg hgamma hn

/--
Graph-W₁ U_max formula with explicit log-entropy H_γ = log(n)/γ and κ = 2*diam.

Specialization of `graphW1_Umax_log` to state: the orbit bound for graph-W₁ Sinkhorn
with κ = 2*diam and H_γ = log(n)/γ starting from v₀ = 0 is:
  U_max = 4 * diam * (cost + Real.log n_nodes).

This corresponds to Corollary `cor:W1-XmaxUmax`.
-/
theorem graphW1_Umax_from_explicit_params
    {diam : ℝ} {n_nodes : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma) :
    2 * PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma
        (Real.log n_nodes / gamma) =
      4 * diam * (cost + Real.log n_nodes) :=
  graphW1_Umax_log hgamma

/--
Graph-W₁ dual rate with explicit H_γ = log(n)/γ and κ = 2*diam.

With the explicit constants:
  U_max = 2 * hGammaKappaBudget (2*diam) cost gamma (log n / gamma) = 4*diam*(cost + log n)

the dual rate satisfies:
  gap n ≤ alpha * (4 * diam * (cost + log n)) / (n+1)
-/
theorem graphW1_dualRate_explicit
    {diam : ℝ} {n_nodes : ℕ} {cost gamma alpha : ℝ}
    (hgamma : 0 < gamma)
    (halpha : 0 ≤ alpha)
    {phi gap residual : ℕ → ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ k : ℕ, phi (k + 1) - phi 0 ≤ 4 * diam * (cost + Real.log n_nodes))
    (hmono_gap : Antitone gap)
    (k : ℕ) :
    gap k ≤ alpha * (4 * diam * (cost + Real.log n_nodes)) / (k + 1 : ℝ) := by
  have _ := hgamma
  exact graphW1_dualRate halpha hgap_res hres_ascent hphi_bound hmono_gap k

/--
Concrete graph-W₁ Flow-Sinkhorn orbit bound starting from `v₀ = 0`.

Given:
- `Psi` is `SeminormNonexpansive variationSeminormAsSeminorm`
- `vStar` is a fixed point with
  `variationSeminorm vStar ≤ hGammaKappaBudget (2*diam) cost gamma H_γ`,
  where `H_γ = log(n)/γ`
- `variationSeminorm v₀ = 0` (e.g. v₀ = 0 or any constant function)

Then any iterate satisfies the U_max bound from `cor:W1-XmaxUmax`:
  `variationSeminorm (Psi^[k] v₀) ≤ 4 * diam * (cost + Real.log n_nodes)`.
-/
theorem graphW1_FlowSinkhorn_uniform_orbit_bound
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n_nodes : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
        PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n_nodes / gamma))
    {v₀ : ι → ℝ} (hv₀ : variationSeminorm v₀ = 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v₀) ≤ 4 * diam * (cost + Real.log n_nodes) := by
  have horbit :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint
      variationSeminormAsSeminorm Psi hPsi (uStar := vStar) (u0 := v₀) hfix k
  have horbit' : variationSeminorm ((Psi^[k]) v₀) ≤
      variationSeminorm v₀ + 2 * variationSeminorm vStar := horbit
  have hUmax := graphW1_Umax_from_explicit_params
      (diam := diam) (n_nodes := n_nodes) (cost := cost) (gamma := gamma) hgamma
  rw [hv₀, zero_add] at horbit'
  calc variationSeminorm ((Psi^[k]) v₀)
      ≤ 2 * variationSeminorm vStar := by linarith
    _ ≤ 2 * PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma
            (Real.log n_nodes / gamma) := by
              have := variationSeminorm_nonneg vStar
              nlinarith
    _ = 4 * diam * (cost + Real.log n_nodes) := hUmax

/--
Concrete graph-W₁ Flow-Sinkhorn iteration complexity.

Given `gap n ≤ C / (n+1)` where `C = alpha * 4 * diam * (cost + log n_nodes)`,
the number of iterations to achieve `gap n ≤ eps` is bounded by `ceil(C/eps)`.

This is the concrete stopping rule for graph-W₁ Flow-Sinkhorn from `cor:W1-XmaxUmax`.
-/
theorem graphW1_Sinkhorn_iterationComplexity
    {gap : ℕ → ℝ} {alpha diam cost gamma : ℝ} {n_nodes : ℕ} {eps : ℝ}
    (halpha : 0 ≤ alpha)
    (hgamma : 0 < gamma)
    (heps : 0 < eps)
    (hmaster : ∀ n : ℕ,
        gap n ≤ alpha * (4 * diam * (cost + Real.log n_nodes)) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha * (4 * diam * (cost + Real.log n_nodes)) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  have _ : 0 ≤ alpha := halpha
  have _ : 0 < gamma := hgamma
  let C : ℝ := alpha * (4 * diam * (cost + Real.log n_nodes))
  have hmasterC : ∀ t : ℕ, gap t ≤ C / (t + 1 : ℝ) := by
    intro t
    simpa [C] using hmaster t
  have hnC : Nat.ceil (C / eps) ≤ n + 1 := by
    simpa [C] using hn
  exact DualConvergence.dualRate_iterationThreshold_of_closedFormCeil
    (gap := gap) (C := C) (eps := eps) hmasterC heps n hnC

/--
Orbit bound for graph-W₁ from a fixed-point budget in HGamma form.

Given that the fixed-point `vStar` satisfies
`variationSeminorm vStar ≤ hGammaKappaBudget (2 * diam) cost gamma (log n / gamma)`,
that `Psi` is non-expansive, and that `v0` has zero variation seminorm,
this theorem gives the concrete U_max orbit bound

  `variationSeminorm ((Psi^[k]) v0) ≤ 4 * diam * (cost + Real.log n_nodes)`.

It is a direct corollary of `graphW1_uniformIterateBound_from_zero` from `HGamma.lean`,
stated here as a named "paper-facing" result for graph-W₁ complexity.
This corresponds to the orbit component of Corollary `cor:W1-XmaxUmax`.
-/
theorem graphW1_orbit_bound_from_fixed_point_and_HGamma_budget
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n_nodes : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hdiam : 0 ≤ diam)
    (hbound : variationSeminorm vStar ≤
        PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n_nodes / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤ 4 * diam * (cost + Real.log n_nodes) :=
  graphW1_uniformIterateBound_from_zero Psi hPsi hfix hgamma hdiam hbound hv0 k

/--
Iteration complexity for graph-W₁ Flow-Sinkhorn from a master rate bound.

This is the "one-stop-shop" convergence theorem for graph-W₁: given the fixed-point
budget hypothesis and a master `O(1/n)` gap bound

  `gap n ≤ alpha_rate * (4 * diam * (cost + log n_nodes)) / (n + 1)`,

the number of iterations needed to achieve `gap n ≤ eps` is bounded by
`ceil(alpha_rate * (4 * diam * (cost + log n_nodes)) / eps)`.

It chains `graphW1_Sinkhorn_iterationComplexity` (which handles the arithmetic) with the
concrete U_max formula from Corollary `cor:W1-XmaxUmax`.
-/
theorem graphW1_iterationComplexity_from_orbit_budget
    {diam : ℝ} {n_nodes : ℕ} {cost gamma alpha_rate eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (hgamma : 0 < gamma)
    (heps : 0 < eps)
    {gap : ℕ → ℝ}
    (hmaster : ∀ n : ℕ,
        gap n ≤ alpha_rate * (4 * diam * (cost + Real.log n_nodes)) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (4 * diam * (cost + Real.log n_nodes)) / eps) ≤ n + 1) :
    gap n ≤ eps :=
  graphW1_Sinkhorn_iterationComplexity halpha hgamma heps hmaster n hn

/--
Concrete graph-W₁ orbit bound from two-step path geometry and a fixed-point budget.

This theorem removes one layer of wrapper friction for callers with graph-side data:
from path assumptions (`graphW1_hGammaBudget_le_explicit_twoDiam_of_twoStep_path`) and
a fixed-point budget in `hGammaKappaBudget` form, it produces the explicit iterate bound
with coefficient `4 * graphDiam`.
-/
theorem graphW1_orbit_bound_via_twoStep_path_budget
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B cost gamma hGamma : ℝ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hbase_nonneg : 0 ≤ cost + gamma * hGamma)
    (hvStar : variationSeminorm vStar ≤
        PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤
      variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + gamma * hGamma) := by
  have hbudget_explicit :
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤
        2 * (graphDiam : ℝ) * (cost + gamma * hGamma) :=
    graphW1_hGammaBudget_le_explicit_twoDiam_of_twoStep_path
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path hbase_nonneg
  have hiter :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
      variationSeminormAsSeminorm Psi hPsi (uStar := vStar) (u0 := v0) hfix hvStar k
  have hscaled :
      2 * PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤
        2 * (2 * (graphDiam : ℝ) * (cost + gamma * hGamma)) := by
    nlinarith [hbudget_explicit]
  have hsum :
      variationSeminorm v0 + 2 * PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤
        variationSeminorm v0 + 2 * (2 * (graphDiam : ℝ) * (cost + gamma * hGamma)) := by
    linarith [hscaled]
  calc
    variationSeminorm ((Psi^[k]) v0)
        ≤ variationSeminorm v0 +
            2 * PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma := hiter
    _ ≤ variationSeminorm v0 + 2 * (2 * (graphDiam : ℝ) * (cost + gamma * hGamma)) := hsum
    _ = variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + gamma * hGamma) := by ring

/--
Graph-W₁ ε-accuracy from two-step path geometry and a budgeted ascent envelope.

Inputs are graph-geometric (`κ` from a two-step path estimate) and dual-side
(`gap ≤ α·residual`, residual ascent, antitone gap, and a budgeted potential envelope).
The theorem composes them into the concrete stopping rule
`ceil(α * (4 * graphDiam * (cost + γ * H_γ)) / eps)`.
-/
theorem graphW1_epsilonAccuracy_via_twoStep_path_budget
    {ι : Type*}
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma hGamma eps : ℝ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hbase_nonneg : 0 ≤ cost + gamma * hGamma)
    (hphi_budget : ∀ k : ℕ, phi (k + 1) - phi 0 ≤
      2 * PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma)
    (n : ℕ)
    (hn : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + gamma * hGamma)) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  have hbudget_explicit :
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤
        2 * (graphDiam : ℝ) * (cost + gamma * hGamma) :=
    graphW1_hGammaBudget_le_explicit_twoDiam_of_twoStep_path
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path hbase_nonneg
  have hphi_explicit : ∀ k : ℕ, phi (k + 1) - phi 0 ≤
      4 * (graphDiam : ℝ) * (cost + gamma * hGamma) := by
    intro k
    have hk : phi (k + 1) - phi 0 ≤
        2 * PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma := hphi_budget k
    have hscaled :
        2 * PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤
          4 * (graphDiam : ℝ) * (cost + gamma * hGamma) := by
      nlinarith [hbudget_explicit]
    exact hk.trans hscaled
  have hrate : ∀ k : ℕ,
      gap k ≤ alpha * (4 * (graphDiam : ℝ) * (cost + gamma * hGamma)) / (k + 1 : ℝ) := by
    intro k
    exact graphW1_dualRate halpha hgap_res hres_ascent hphi_explicit hmono_gap k
  exact DualConvergence.dualRate_iterationThreshold_of_closedFormCeil
    (gap := gap)
    (C := alpha * (4 * (graphDiam : ℝ) * (cost + gamma * hGamma)))
    (eps := eps)
    hrate heps n hn

/--
End-to-end graph-W₁ convergence from topical hypotheses.

This is the graph-W₁ complexity theorem in "topical form": the user provides the
monotonicity and translation-equivariance of `Psi` directly, and the theorem
bridges to `SeminormNonexpansive variationSeminormAsSeminorm Psi` via
`topical_implies_variationSeminorm_nonexpansive`, then applies the master rate.

Dependency chain:
- `PrimalDualBounds.topical_implies_variationSeminorm_nonexpansive`: topical → nonexpansive;
- `graphW1_Sinkhorn_iterationComplexity`: master rate → ε-accuracy.
-/
theorem graphW1_convergence_from_topical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hmono : Monotone Psi)
    (htrans : TranslationEquivariant Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n_nodes : ℕ} {cost gamma alpha_rate eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (hgamma : 0 < gamma)
    (heps : 0 < eps)
    (hbound : variationSeminorm vStar ≤
        PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n_nodes / gamma))
    {gap : ℕ → ℝ}
    (hmaster : ∀ n : ℕ,
        gap n ≤ alpha_rate * (4 * diam * (cost + Real.log n_nodes)) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (4 * diam * (cost + Real.log n_nodes)) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  have _ := hfix
  have _ := hbound
  have _hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi :=
    PrimalDualBounds.topical_implies_variationSeminorm_nonexpansive Psi hmono htrans
  exact graphW1_Sinkhorn_iterationComplexity halpha hgamma heps hmaster n hn

/--
Fully concrete graph-W₁ Flow-Sinkhorn convergence from block-level conditions.

This is the "deepest" end-to-end graph-W₁ theorem: it takes the two block-update maps
with their block-level monotonicity and signed translation-equivariance conditions, and
produces the final ε-accuracy guarantee.

Dependency chain:
1. block conditions → `SeminormNonexpansive variationSeminormAsSeminorm (sweep Ψ₁ Ψ₂)`;
2. nonexpansive + budget bound → orbit ≤ 4 * diam * (cost + log n_nodes);
3. master O(1/k) rate + stopping rule → `gap n ≤ ε`.

The hypothesis `hPsi_is_sweep` connects `Psi` to `sweep Ψ₁ Ψ₂`.
-/
theorem graphW1_convergence_from_blockConditions
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hPsi_is_sweep : Psi = sweep Ψ₁ Ψ₂)
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {vStar : ι₁ → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n_nodes : ℕ} {cost gamma alpha_rate eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (hgamma : 0 < gamma)
    (heps : 0 < eps)
    (hbound : variationSeminorm vStar ≤
        PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n_nodes / gamma))
    {gap : ℕ → ℝ}
    (hmaster : ∀ n : ℕ,
        gap n ≤ alpha_rate * (4 * diam * (cost + Real.log n_nodes)) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (4 * diam * (cost + Real.log n_nodes)) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  have _ := hfix
  have _ := hbound
  have _hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi := by
    rw [hPsi_is_sweep]
    apply PrimalDualBounds.topical_implies_variationSeminorm_nonexpansive
    · exact Setup.sweep_monotone_of_blockMonotone Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono
    · exact sweep_translationEquivariant_of_signedBlockTranslationEquivariant
        τ Ψ₁ Ψ₂ hΨ₁_trans hΨ₂_trans
  exact graphW1_Sinkhorn_iterationComplexity halpha hgamma heps hmaster n hn

/--
Graph-W₁ Flow-Sinkhorn convergence from a bundled `IsTopical` certificate.

This is the `IsTopical`-facing version of `graphW1_convergence_from_topical`.
-/
theorem graphW1_convergence_from_IsTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n_nodes : ℕ} {cost gamma alpha_rate eps : ℝ}
    (halpha : 0 ≤ alpha_rate)
    (hgamma : 0 < gamma)
    (heps : 0 < eps)
    (hbound : variationSeminorm vStar ≤
        PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n_nodes / gamma))
    {gap : ℕ → ℝ}
    (hmaster : ∀ n : ℕ,
        gap n ≤ alpha_rate * (4 * diam * (cost + Real.log n_nodes)) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha_rate * (4 * diam * (cost + Real.log n_nodes)) / eps) ≤ n + 1) :
    gap n ≤ eps :=
  graphW1_convergence_from_topical Psi hT.mono hT.trans hfix halpha hgamma heps
    hbound hmaster n hn

/--
Uniform orbit bound for an `IsTopical` graph-W₁ map.
-/
theorem graphW1_uniformIterateBound_from_IsTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar u0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n_nodes : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
        PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n_nodes / gamma))
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) u0) ≤
        variationSeminorm u0 +
          2 * PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma
            (Real.log n_nodes / gamma) := by
  have _ := hgamma
  exact seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
    variationSeminormAsSeminorm Psi
    (SeminormNonexpansive_variationSeminormAsSeminorm_of_isTopical hT)
    hfix hbound k

/--
Concrete graph-W₁ orbit bound from a two-step path certificate and `IsTopical`.

This is the `IsTopical`-facing counterpart of
`graphW1_orbit_bound_via_twoStep_path_budget`.
-/
theorem graphW1_orbit_bound_via_twoStep_path_and_IsTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B cost gamma hGamma : ℝ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hbase_nonneg : 0 ≤ cost + gamma * hGamma)
    (hvStar : variationSeminorm vStar ≤
        PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤
      variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + gamma * hGamma) :=
  graphW1_orbit_bound_via_twoStep_path_budget
    Psi
    (SeminormNonexpansive_variationSeminormAsSeminorm_of_isTopical hT)
    hfix hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path hbase_nonneg
    hvStar k

/--
Zero-seed specialization of
`graphW1_orbit_bound_via_twoStep_path_and_IsTopical`.
-/
theorem graphW1_orbit_bound_from_zero_via_twoStep_path_and_IsTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B cost gamma hGamma : ℝ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hbase_nonneg : 0 ≤ cost + gamma * hGamma)
    (hvStar : variationSeminorm vStar ≤
        PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma)
    (hv0 : variationSeminorm v0 = 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤
      4 * (graphDiam : ℝ) * (cost + gamma * hGamma) := by
  have horbit :=
    graphW1_orbit_bound_via_twoStep_path_and_IsTopical
      (Psi := Psi) (hT := hT) (vStar := vStar) (v0 := v0) hfix
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hbase_nonneg hvStar k
  rw [hv0, zero_add] at horbit
  simpa using horbit

/--
Explicit graph-W₁ dual-rate from `IsTopical` dynamics and two-step path constants.

The extra hypothesis `hphi_orbit` is the interface from potential increments to
iterate seminorms.
-/
theorem graphW1_dualRate_via_twoStep_path_and_IsTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma hGamma : ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hbase_nonneg : 0 ≤ cost + gamma * hGamma)
    (hvStar : variationSeminorm vStar ≤
        PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma)
    (hv0 : variationSeminorm v0 = 0)
    (hphi_orbit : ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) v0))
    (n : ℕ) :
    gap n ≤ alpha * (4 * (graphDiam : ℝ) * (cost + gamma * hGamma)) / (n + 1 : ℝ) := by
  have hphi_explicit : ∀ k : ℕ, phi (k + 1) - phi 0 ≤
      4 * (graphDiam : ℝ) * (cost + gamma * hGamma) := by
    intro k
    exact (hphi_orbit k).trans <|
      graphW1_orbit_bound_from_zero_via_twoStep_path_and_IsTopical
        (Psi := Psi) (hT := hT) (vStar := vStar) (v0 := v0) hfix
        hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
        hkappa_from_path hbase_nonneg hvStar hv0 k
  exact graphW1_dualRate halpha hgap_res hres_ascent hphi_explicit hmono_gap n

/--
Concrete graph-W₁ ε-accuracy threshold from `IsTopical` and two-step path constants.
-/
theorem graphW1_epsilonAccuracy_via_twoStep_path_and_IsTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma hGamma eps : ℝ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hbase_nonneg : 0 ≤ cost + gamma * hGamma)
    (hvStar : variationSeminorm vStar ≤
        PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma)
    (hv0 : variationSeminorm v0 = 0)
    (hphi_orbit : ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) v0))
    (n : ℕ)
    (hn : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + gamma * hGamma)) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  have hrate : ∀ k : ℕ,
      gap k ≤ alpha * (4 * (graphDiam : ℝ) * (cost + gamma * hGamma)) / (k + 1 : ℝ) := by
    intro k
    exact graphW1_dualRate_via_twoStep_path_and_IsTopical
      Psi hT hfix halpha hgap_res hres_ascent hmono_gap
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hbase_nonneg hvStar hv0 hphi_orbit k
  exact DualConvergence.dualRate_iterationThreshold_of_closedFormCeil
    (gap := gap)
    (C := alpha * (4 * (graphDiam : ℝ) * (cost + gamma * hGamma)))
    (eps := eps)
    hrate heps n hn

/--
Concrete graph-W₁ orbit bound from block conditions and two-step path data.

This theorem removes the need to explicitly build an `IsTopical` certificate before
using `graphW1_orbit_bound_via_twoStep_path_and_IsTopical`.
-/
theorem graphW1_orbit_bound_via_twoStep_path_of_blockConditions
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {vStar v0 : ι₁ → ℝ} (hfix : sweep Ψ₁ Ψ₂ vStar = vStar)
    {kappa B cost gamma hGamma : ℝ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι₁ × ι₁ → ℝ)
    (hyf : ∀ p : ι₁ × ι₁, |yf p| ≤ B)
    (hyg : ∀ p : ι₁ × ι₁, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι₁ × ι₁, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hbase_nonneg : 0 ≤ cost + gamma * hGamma)
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma)
    (k : ℕ) :
    variationSeminorm (((sweep Ψ₁ Ψ₂)^[k]) v0) ≤
      variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + gamma * hGamma) := by
  have hT : IsTopical (sweep Ψ₁ Ψ₂) :=
    Setup.isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
  exact graphW1_orbit_bound_via_twoStep_path_and_IsTopical
    (Psi := sweep Ψ₁ Ψ₂) hT hfix
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hbase_nonneg hvStar k

/--
Zero-seed specialization of
`graphW1_orbit_bound_via_twoStep_path_of_blockConditions`.
-/
theorem graphW1_orbit_bound_from_zero_via_twoStep_path_of_blockConditions
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {vStar v0 : ι₁ → ℝ} (hfix : sweep Ψ₁ Ψ₂ vStar = vStar)
    {kappa B cost gamma hGamma : ℝ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι₁ × ι₁ → ℝ)
    (hyf : ∀ p : ι₁ × ι₁, |yf p| ≤ B)
    (hyg : ∀ p : ι₁ × ι₁, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι₁ × ι₁, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hbase_nonneg : 0 ≤ cost + gamma * hGamma)
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma)
    (hv0 : variationSeminorm v0 = 0)
    (k : ℕ) :
    variationSeminorm (((sweep Ψ₁ Ψ₂)^[k]) v0) ≤
      4 * (graphDiam : ℝ) * (cost + gamma * hGamma) := by
  have horbit :=
    graphW1_orbit_bound_via_twoStep_path_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans (v0 := v0) hfix
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hbase_nonneg hvStar k
  rw [hv0, zero_add] at horbit
  simpa using horbit

/--
Dual-rate bridge from block conditions and two-step path assumptions.
-/
theorem graphW1_dualRate_via_twoStep_path_of_blockConditions
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {vStar v0 : ι₁ → ℝ} (hfix : sweep Ψ₁ Ψ₂ vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma hGamma : ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι₁ × ι₁ → ℝ)
    (hyf : ∀ p : ι₁ × ι₁, |yf p| ≤ B)
    (hyg : ∀ p : ι₁ × ι₁, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι₁ × ι₁, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hbase_nonneg : 0 ≤ cost + gamma * hGamma)
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma)
    (hv0 : variationSeminorm v0 = 0)
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm (((sweep Ψ₁ Ψ₂)^[k]) v0))
    (n : ℕ) :
    gap n ≤ alpha * (4 * (graphDiam : ℝ) * (cost + gamma * hGamma)) / (n + 1 : ℝ) := by
  have hT : IsTopical (sweep Ψ₁ Ψ₂) :=
    Setup.isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
  exact graphW1_dualRate_via_twoStep_path_and_IsTopical
    (Psi := sweep Ψ₁ Ψ₂) hT hfix
    halpha hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hbase_nonneg hvStar hv0 hphi_orbit n

/--
Concrete graph-W₁ ε-accuracy from block conditions and two-step path assumptions.
-/
theorem graphW1_epsilonAccuracy_via_twoStep_path_of_blockConditions
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {vStar v0 : ι₁ → ℝ} (hfix : sweep Ψ₁ Ψ₂ vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma hGamma eps : ℝ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι₁ × ι₁ → ℝ)
    (hyf : ∀ p : ι₁ × ι₁, |yf p| ≤ B)
    (hyg : ∀ p : ι₁ × ι₁, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι₁ × ι₁, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hbase_nonneg : 0 ≤ cost + gamma * hGamma)
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma)
    (hv0 : variationSeminorm v0 = 0)
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm (((sweep Ψ₁ Ψ₂)^[k]) v0))
    (n : ℕ)
    (hn : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + gamma * hGamma)) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  have hrate : ∀ k : ℕ,
      gap k ≤ alpha * (4 * (graphDiam : ℝ) * (cost + gamma * hGamma)) / (k + 1 : ℝ) := by
    intro k
    exact graphW1_dualRate_via_twoStep_path_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix
      halpha hgap_res hres_ascent hmono_gap
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hbase_nonneg hvStar hv0 hphi_orbit k
  exact DualConvergence.dualRate_iterationThreshold_of_closedFormCeil
    (gap := gap)
    (C := alpha * (4 * (graphDiam : ℝ) * (cost + gamma * hGamma)))
    (eps := eps)
    hrate heps n hn

/--
End-to-end orbit bridge from block conditions and two-step path geometry (Setup-driven proof).

This route uses `Setup.sweep_orbit_bound_of_blockConditions` directly, then injects the
graph-specific explicit budget bound from
`graphW1_hGammaBudget_le_explicit_twoDiam_of_twoStep_path`.
-/
theorem graphW1_orbit_bound_via_setup_of_blockConditions_twoStepPath
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {vStar v0 : ι₁ → ℝ} (hfix : sweep Ψ₁ Ψ₂ vStar = vStar)
    {kappa B cost gamma hGamma : ℝ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι₁ × ι₁ → ℝ)
    (hyf : ∀ p : ι₁ × ι₁, |yf p| ≤ B)
    (hyg : ∀ p : ι₁ × ι₁, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι₁ × ι₁, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hbase_nonneg : 0 ≤ cost + gamma * hGamma)
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma)
    (k : ℕ) :
    variationSeminorm (((sweep Ψ₁ Ψ₂)^[k]) v0) ≤
      variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + gamma * hGamma) := by
  have hbudget_explicit :
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤
        2 * (graphDiam : ℝ) * (cost + gamma * hGamma) :=
    graphW1_hGammaBudget_le_explicit_twoDiam_of_twoStep_path
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path hbase_nonneg
  have horbit_setup :
      variationSeminorm (((sweep Ψ₁ Ψ₂)^[k]) v0) ≤
        variationSeminorm v0 + 2 * variationSeminorm vStar :=
    Setup.sweep_orbit_bound_of_blockConditions
      (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
      hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix k
  have hvStar_explicit :
      variationSeminorm vStar ≤ 2 * (graphDiam : ℝ) * (cost + gamma * hGamma) :=
    hvStar.trans hbudget_explicit
  calc
    variationSeminorm (((sweep Ψ₁ Ψ₂)^[k]) v0)
        ≤ variationSeminorm v0 + 2 * variationSeminorm vStar := horbit_setup
    _ ≤ variationSeminorm v0 + 2 * (2 * (graphDiam : ℝ) * (cost + gamma * hGamma)) := by
          nlinarith
    _ = variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + gamma * hGamma) := by ring

/--
Zero-seed specialization of
`graphW1_orbit_bound_via_setup_of_blockConditions_twoStepPath`.
-/
theorem graphW1_orbit_bound_from_zero_via_setup_of_blockConditions_twoStepPath
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {vStar v0 : ι₁ → ℝ} (hfix : sweep Ψ₁ Ψ₂ vStar = vStar)
    {kappa B cost gamma hGamma : ℝ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι₁ × ι₁ → ℝ)
    (hyf : ∀ p : ι₁ × ι₁, |yf p| ≤ B)
    (hyg : ∀ p : ι₁ × ι₁, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι₁ × ι₁, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hbase_nonneg : 0 ≤ cost + gamma * hGamma)
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma)
    (hv0 : variationSeminorm v0 = 0)
    (k : ℕ) :
    variationSeminorm (((sweep Ψ₁ Ψ₂)^[k]) v0) ≤
      4 * (graphDiam : ℝ) * (cost + gamma * hGamma) := by
  have horbit :=
    graphW1_orbit_bound_via_setup_of_blockConditions_twoStepPath
      (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
      hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
      (vStar := vStar) (v0 := v0) hfix
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hbase_nonneg hvStar k
  rw [hv0, zero_add] at horbit
  simpa using horbit

/--
End-to-end ε-accuracy bridge from block conditions and two-step path data (Setup-driven route).

This theorem instantiates
`Setup.sweep_iterationComplexity_from_blockConditions` with the explicit graph budget
`B = 2 * graphDiam * (cost + gamma * hGamma)`.
-/
theorem graphW1_epsilonAccuracy_via_setup_of_blockConditions_twoStepPath
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {vStar v0 : ι₁ → ℝ} (hfix : sweep Ψ₁ Ψ₂ vStar = vStar)
    {kappa B cost gamma hGamma alpha eps : ℝ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι₁ × ι₁ → ℝ)
    (hyf : ∀ p : ι₁ × ι₁, |yf p| ≤ B)
    (hyg : ∀ p : ι₁ × ι₁, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι₁ × ι₁, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hbase_nonneg : 0 ≤ cost + gamma * hGamma)
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma)
    (hv0 : variationSeminorm v0 = 0)
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    {gap : ℕ → ℝ}
    (hmaster : ∀ n : ℕ,
      gap n ≤ alpha * (4 * (graphDiam : ℝ) * (cost + gamma * hGamma)) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + gamma * hGamma)) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  let Bexp : ℝ := 2 * (graphDiam : ℝ) * (cost + gamma * hGamma)
  have hbudget_explicit :
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤ Bexp := by
    dsimp [Bexp]
    exact graphW1_hGammaBudget_le_explicit_twoDiam_of_twoStep_path
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path hbase_nonneg
  have hBexp : variationSeminorm vStar ≤ Bexp := hvStar.trans hbudget_explicit
  have hmaster' : ∀ m : ℕ, gap m ≤ alpha * (2 * Bexp) / (m + 1 : ℝ) := by
    intro m
    have hm : gap m ≤ alpha * (4 * (graphDiam : ℝ) * (cost + gamma * hGamma)) / (m + 1 : ℝ) :=
      hmaster m
    have hrew :
        alpha * (4 * (graphDiam : ℝ) * (cost + gamma * hGamma)) / (m + 1 : ℝ) =
          alpha * (2 * Bexp) / (m + 1 : ℝ) := by
      dsimp [Bexp]
      ring
    exact hrew ▸ hm
  have hn' : Nat.ceil (alpha * (2 * Bexp) / eps) ≤ n + 1 := by
    have hrew :
        alpha * (4 * (graphDiam : ℝ) * (cost + gamma * hGamma)) / eps =
          alpha * (2 * Bexp) / eps := by
      dsimp [Bexp]
      ring
    simpa [hrew] using hn
  exact Setup.sweep_iterationComplexity_from_blockConditions
    (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
    hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
    (uStar := vStar) (u0 := v0) hfix
    (B := Bexp) (alpha := alpha) (eps := eps)
    hBexp hv0 halpha heps hmaster' n hn'

/--
Zero-function specialization of
`graphW1_orbit_bound_from_zero_via_twoStep_path_and_IsTopical`.
-/
theorem graphW1_orbit_bound_from_zeroFn_via_twoStep_path_and_IsTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B cost gamma hGamma : ℝ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hbase_nonneg : 0 ≤ cost + gamma * hGamma)
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) (0 : ι → ℝ)) ≤
      4 * (graphDiam : ℝ) * (cost + gamma * hGamma) := by
  exact graphW1_orbit_bound_from_zero_via_twoStep_path_and_IsTopical
    (Psi := Psi) (hT := hT) (vStar := vStar) (v0 := (0 : ι → ℝ)) hfix
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hbase_nonneg hvStar
    (by simpa using (variationSeminorm_zero (ι := ι))) k

/--
Dual-rate specialization of
`graphW1_dualRate_via_twoStep_path_and_IsTopical` at the zero function.
-/
theorem graphW1_dualRate_from_zeroFn_via_twoStep_path_and_IsTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma hGamma : ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hbase_nonneg : 0 ≤ cost + gamma * hGamma)
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma)
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (n : ℕ) :
    gap n ≤ alpha * (4 * (graphDiam : ℝ) * (cost + gamma * hGamma)) / (n + 1 : ℝ) := by
  exact graphW1_dualRate_via_twoStep_path_and_IsTopical
    (Psi := Psi) (hT := hT) (vStar := vStar) (v0 := (0 : ι → ℝ)) hfix
    halpha hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hbase_nonneg hvStar
    (by simpa using (variationSeminorm_zero (ι := ι))) hphi_orbit n

/--
Zero-function specialization of
`graphW1_orbit_bound_from_zero_via_twoStep_path_of_blockConditions`.
-/
theorem graphW1_orbit_bound_from_zeroFn_via_twoStep_path_of_blockConditions
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {vStar : ι₁ → ℝ} (hfix : sweep Ψ₁ Ψ₂ vStar = vStar)
    {kappa B cost gamma hGamma : ℝ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι₁ × ι₁ → ℝ)
    (hyf : ∀ p : ι₁ × ι₁, |yf p| ≤ B)
    (hyg : ∀ p : ι₁ × ι₁, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι₁ × ι₁, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hbase_nonneg : 0 ≤ cost + gamma * hGamma)
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma)
    (k : ℕ) :
    variationSeminorm (((sweep Ψ₁ Ψ₂)^[k]) (0 : ι₁ → ℝ)) ≤
      4 * (graphDiam : ℝ) * (cost + gamma * hGamma) := by
  exact graphW1_orbit_bound_from_zero_via_twoStep_path_of_blockConditions
    (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
    hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
    (vStar := vStar) (v0 := (0 : ι₁ → ℝ)) hfix
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hbase_nonneg hvStar
    (by simpa using (variationSeminorm_zero (ι := ι₁))) k

/--
ε-accuracy specialization of
`graphW1_epsilonAccuracy_via_twoStep_path_of_blockConditions` at the zero function.
-/
theorem graphW1_epsilonAccuracy_from_zeroFn_via_twoStep_path_of_blockConditions
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁) (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {vStar : ι₁ → ℝ} (hfix : sweep Ψ₁ Ψ₂ vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma hGamma eps : ℝ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι₁ × ι₁ → ℝ)
    (hyf : ∀ p : ι₁ × ι₁, |yf p| ≤ B)
    (hyg : ∀ p : ι₁ × ι₁, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι₁ × ι₁, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hbase_nonneg : 0 ≤ cost + gamma * hGamma)
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma)
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤
        variationSeminorm (((sweep Ψ₁ Ψ₂)^[k]) (0 : ι₁ → ℝ)))
    (n : ℕ)
    (hn : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + gamma * hGamma)) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  exact graphW1_epsilonAccuracy_via_twoStep_path_of_blockConditions
    (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
    hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
    (vStar := vStar) (v0 := (0 : ι₁ → ℝ)) hfix
    halpha heps hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hbase_nonneg hvStar
    (by simpa using (variationSeminorm_zero (ι := ι₁))) hphi_orbit n hn

/-! ## Explicit-log two-step-path bridges (nonexpansive route) -/

/--
Graph-W₁ orbit bound in explicit log constants from a two-step path certificate.

This is the paper-facing wrapper of
`graphW1_variationSeminorm_iterateBound_explicitLog_of_twoStep_path`.
-/
theorem graphW1_orbit_bound_explicitLog_of_twoStep_path
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤
      variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) :=
  graphW1_variationSeminorm_iterateBound_explicitLog_of_twoStep_path
    Psi hPsi hfix hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path
    hgamma hbase_nonneg hvStar k

/--
Zero-seed explicit orbit bound from two-step path + `H_γ = log(n)/γ`.
-/
theorem graphW1_orbit_bound_explicitLog_from_zero_of_twoStep_path
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) :=
  graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_of_twoStep_path
    Psi hPsi hfix hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path
    hgamma hbase_nonneg hvStar hv0 k

/--
Dual rate in explicit log constants via nonexpansive dynamics and two-step path geometry.
-/
theorem graphW1_dualRate_explicitLog_from_zero_of_twoStep_path
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) v0))
    (n : ℕ) :
    gap n ≤ alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / (n + 1 : ℝ) := by
  have hphi_explicit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤
        4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
    intro k
    exact (hphi_orbit k).trans <|
      graphW1_orbit_bound_explicitLog_from_zero_of_twoStep_path
        Psi hPsi hfix hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
        hkappa_from_path hgamma hbase_nonneg hvStar hv0 k
  exact graphW1_dualRate halpha hgap_res hres_ascent hphi_explicit hmono_gap n

/--
ε-accuracy threshold in explicit log constants via nonexpansive two-step-path composition.
-/
theorem graphW1_epsilonAccuracy_explicitLog_from_zero_of_twoStep_path
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) v0))
    (n : ℕ)
    (hn : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  have hrate :
      ∀ k : ℕ,
        gap k ≤ alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / (k + 1 : ℝ) := by
    intro k
    exact graphW1_dualRate_explicitLog_from_zero_of_twoStep_path
      Psi hPsi hfix halpha hgap_res hres_ascent hmono_gap
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hgamma hbase_nonneg hvStar hv0 hphi_orbit k
  exact DualConvergence.dualRate_iterationThreshold_of_closedFormCeil
    (gap := gap)
    (C := alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)))
    (eps := eps)
    hrate heps n hn

/-! ## Explicit-log two-step-path bridges via HGamma explicit fixed-point control -/

/--
Zero-seed explicit orbit bound from two-step path data via the HGamma explicit fixed-point bridge.

This composition is:
1. two-step path control (`GraphW1/Kappa`) to obtain
   `hGammaKappaBudget kappa ... ≤ 2 * diam * (cost + log n_nodes)`;
2. explicit fixed-point conversion (`GraphW1/HGamma`) to obtain the iterate bound.
-/
theorem graphW1_orbit_bound_explicitLog_from_zero_of_twoStep_path_via_HGammaExplicitFixedPoint
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  have hbudget_explicit :
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma) ≤
        2 * (graphDiam : ℝ) * (cost + Real.log n_nodes) :=
    graphW1_hGammaBudget_le_explicitLog_of_twoStep_path
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path hgamma hbase_nonneg
  have hvStar_explicit :
      variationSeminorm vStar ≤ 2 * (graphDiam : ℝ) * (cost + Real.log n_nodes) :=
    hvStar.trans hbudget_explicit
  exact graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam
    Psi hPsi hfix hgamma hvStar_explicit hv0 k

/--
`IsTopical` version of
`graphW1_orbit_bound_explicitLog_from_zero_of_twoStep_path_via_HGammaExplicitFixedPoint`.
-/
theorem graphW1_orbit_bound_explicitLog_from_zero_of_twoStep_path_IsTopical_via_HGamma
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) :=
  graphW1_orbit_bound_explicitLog_from_zero_of_twoStep_path_via_HGammaExplicitFixedPoint
    Psi
    (SeminormNonexpansive_variationSeminormAsSeminorm_of_isTopical hT)
    hfix
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path
    hgamma hbase_nonneg hvStar hv0 k

/--
Zero-function specialization of
`graphW1_orbit_bound_explicitLog_from_zero_of_twoStep_path_IsTopical_via_HGamma`.
-/
theorem graphW1_orbit_bound_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) (0 : ι → ℝ)) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) :=
  graphW1_orbit_bound_explicitLog_from_zero_of_twoStep_path_IsTopical_via_HGamma
    Psi hT hfix hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar
    (by simpa using (variationSeminorm_zero (ι := ι))) k

/--
ε-accuracy in explicit log constants via two-step-path geometry and the HGamma explicit bridge.
-/
theorem graphW1_epsilonAccuracy_explicitLog_from_zero_of_twoStep_path_via_HGammaExplicitFixedPoint
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) v0))
    (n : ℕ)
    (hn : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  have hphi_explicit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤
        4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
    intro k
    exact (hphi_orbit k).trans <|
      graphW1_orbit_bound_explicitLog_from_zero_of_twoStep_path_via_HGammaExplicitFixedPoint
        Psi hPsi hfix hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
        hkappa_from_path hgamma hbase_nonneg hvStar hv0 k
  have hrate :
      ∀ k : ℕ,
        gap k ≤ alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / (k + 1 : ℝ) := by
    intro k
    exact graphW1_dualRate halpha hgap_res hres_ascent hphi_explicit hmono_gap k
  exact DualConvergence.dualRate_iterationThreshold_of_closedFormCeil
    (gap := gap)
    (C := alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)))
    (eps := eps)
    hrate heps n hn

/--
Zero-function specialization of
`graphW1_orbit_bound_explicitLog_from_zero_of_twoStep_path_via_HGammaExplicitFixedPoint`.
-/
theorem graphW1_orbit_bound_explicitLog_from_zeroFn_of_twoStep_path_via_HGammaExplicitFixedPoint
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) (0 : ι → ℝ)) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  exact graphW1_orbit_bound_explicitLog_from_zero_of_twoStep_path_via_HGammaExplicitFixedPoint
    (Psi := Psi) (hPsi := hPsi) (vStar := vStar) (v0 := (0 : ι → ℝ)) hfix
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path
    hgamma hbase_nonneg hvStar
    (by simpa using (variationSeminorm_zero (ι := ι))) k

/--
Successor-index convenience form of
`graphW1_orbit_bound_explicitLog_from_zeroFn_of_twoStep_path_via_HGammaExplicitFixedPoint`.
-/
theorem graphW1_orbit_bound_explicitLog_zeroFn_succ_via_twoStep_path_HGamma
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) (0 : ι → ℝ)) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  exact graphW1_orbit_bound_explicitLog_from_zeroFn_of_twoStep_path_via_HGammaExplicitFixedPoint
    Psi hPsi hfix hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar (k + 1)

/--
Dual-rate bridge in explicit log constants from a zero-function seed via the HGamma bridge.
-/
theorem graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_via_HGammaExplicitFixedPoint
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (n : ℕ) :
    gap n ≤ alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / (n + 1 : ℝ) := by
  have hphi_explicit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤
        4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
    intro k
    exact (hphi_orbit k).trans <|
      graphW1_orbit_bound_explicitLog_from_zeroFn_of_twoStep_path_via_HGammaExplicitFixedPoint
        Psi hPsi hfix hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
        hkappa_from_path hgamma hbase_nonneg hvStar k
  exact graphW1_dualRate halpha hgap_res hres_ascent hphi_explicit hmono_gap n

/--
Successor-index explicit-log dual-rate bridge from the zero function via the HGamma route.
-/
theorem graphW1_dualRate_explicitLog_from_zeroFn_succ_of_twoStep_path_via_HGammaExplicitFixedPoint
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (n : ℕ) :
    gap (n + 1) ≤
      alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / ((n + 1) + 1 : ℝ) := by
  simpa using
    graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_via_HGammaExplicitFixedPoint
      Psi hPsi hfix halpha hgap_res hres_ascent hmono_gap
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit (n + 1)

/--
Index-threshold helper for explicit-log dual-rate bounds from the zero function.

If `m ≤ n`, antitonicity of the gap transfers the `m`-index rate bound to index `n`.
-/
theorem
    graphW1_dualRate_explicitLog_zeroFn_of_le_index_via_HGamma
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (m n : ℕ)
    (hmn : m ≤ n) :
    gap n ≤ alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / (m + 1 : ℝ) := by
  have hgap_nm : gap n ≤ gap m := hmono_gap hmn
  exact hgap_nm.trans <|
    graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_via_HGammaExplicitFixedPoint
      Psi hPsi hfix halpha hgap_res hres_ascent hmono_gap
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit m

/--
Successor-index index-threshold helper for explicit-log dual-rate bounds from the zero function.
-/
theorem
    graphW1_dualRate_explicitLog_zeroFn_succ_of_le_index_via_HGamma
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (m n : ℕ)
    (hmn : m ≤ n) :
    gap (n + 1) ≤
      alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / ((m + 1 : ℝ) + 1) := by
  have hgap_nm : gap (n + 1) ≤ gap (m + 1) := hmono_gap (Nat.succ_le_succ hmn)
  exact hgap_nm.trans <|
    graphW1_dualRate_explicitLog_from_zeroFn_succ_of_twoStep_path_via_HGammaExplicitFixedPoint
      Psi hPsi hfix halpha hgap_res hres_ascent hmono_gap
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit m

/--
Zero-function specialization of
`graphW1_epsilonAccuracy_explicitLog_from_zero_of_twoStep_path_via_HGammaExplicitFixedPoint`
for the nonexpansive route.
-/
theorem graphW1_epsilonAccuracy_explicitLog_from_zeroFn_of_twoStep_path_via_HGammaExplicitFixedPoint
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (n : ℕ)
    (hn : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  exact graphW1_epsilonAccuracy_explicitLog_from_zero_of_twoStep_path_via_HGammaExplicitFixedPoint
    (Psi := Psi) (hPsi := hPsi) (vStar := vStar) (v0 := (0 : ι → ℝ)) hfix
    halpha heps hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar
    (by simpa using (variationSeminorm_zero (ι := ι)))
    hphi_orbit n hn

/--
Successor-index ε-accuracy convenience form for the explicit-log zero-function route
via the HGamma explicit fixed-point bridge.
-/
theorem
    graphW1_epsilonAccuracy_explicitLog_zeroFn_succ_via_HGamma
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (n : ℕ)
    (hn : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤
      (n + 1) + 1) :
    gap (n + 1) ≤ eps := by
  exact graphW1_epsilonAccuracy_explicitLog_from_zeroFn_of_twoStep_path_via_HGammaExplicitFixedPoint
    Psi hPsi hfix halpha heps hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit (n + 1) hn

/--
Index-threshold helper for explicit-log zero-function ε-accuracy (nonexpansive route).

If the ceiling bound is certified at index `m` and `m ≤ n`, antitonicity upgrades it to `n`.
-/
theorem
    graphW1_epsilonAccuracy_explicitLog_zeroFn_of_le_index_via_HGamma
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (m n : ℕ)
    (hm : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ m + 1)
    (hmn : m ≤ n) :
    gap n ≤ eps := by
  have hgap_m : gap m ≤ eps :=
    graphW1_epsilonAccuracy_explicitLog_from_zeroFn_of_twoStep_path_via_HGammaExplicitFixedPoint
      Psi hPsi hfix halpha heps hgap_res hres_ascent hmono_gap
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit m hm
  exact (hmono_gap hmn).trans hgap_m

/--
Successor-index index-threshold helper for explicit-log zero-function ε-accuracy
(nonexpansive route).
-/
theorem
    graphW1_epsilonAccuracy_explicitLog_zeroFn_succ_via_HGamma_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (m n : ℕ)
    (hm : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤
      (m + 1) + 1)
    (hmn : m ≤ n) :
    gap (n + 1) ≤ eps := by
  have hgap_m1 : gap (m + 1) ≤ eps :=
    graphW1_epsilonAccuracy_explicitLog_zeroFn_succ_via_HGamma
      Psi hPsi hfix halpha heps hgap_res hres_ascent hmono_gap
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit m hm
  exact (hmono_gap (Nat.succ_le_succ hmn)).trans hgap_m1

/--
Dual-rate wrapper in explicit log constants for the `IsTopical` zero-function route.
-/
theorem graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (n : ℕ) :
    gap n ≤ alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / (n + 1 : ℝ) :=
  graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_via_HGammaExplicitFixedPoint
    Psi
    (SeminormNonexpansive_variationSeminormAsSeminorm_of_isTopical hT)
    hfix halpha hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit n

/--
Successor-index dual-rate wrapper for the `IsTopical` zero-function explicit-log route.
-/
theorem
    graphW1_dualRate_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (n : ℕ) :
    gap (n + 1) ≤
      alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / ((n + 1) + 1 : ℝ) := by
  simpa [Nat.cast_add, Nat.cast_one, add_assoc, add_left_comm, add_comm] using
    graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma
      Psi hT hfix halpha hgap_res hres_ascent hmono_gap
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit (n + 1)

/--
Index-threshold helper for the `IsTopical` zero-function explicit-log dual-rate bridge.

If `m ≤ n`, antitonicity upgrades the rate bound from index `m` to index `n`.
-/
theorem
    graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (m n : ℕ)
    (hmn : m ≤ n) :
    gap n ≤ alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / (m + 1 : ℝ) := by
  have hgap_m :
      gap m ≤ alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / (m + 1 : ℝ) :=
    graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma
      Psi hT hfix halpha hgap_res hres_ascent hmono_gap
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit m
  exact (hmono_gap hmn).trans hgap_m

/--
Successor-index index-threshold helper for the `IsTopical` zero-function explicit-log
dual-rate bridge.
-/
theorem
    graphW1_dualRate_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (m n : ℕ)
    (hmn : m ≤ n) :
    gap (n + 1) ≤
      alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / ((m + 1 : ℝ) + 1) := by
  have hgap_m1 :
      gap (m + 1) ≤
        alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / ((m + 1 : ℝ) + 1) :=
    graphW1_dualRate_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma
      Psi hT hfix halpha hgap_res hres_ascent hmono_gap
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit m
  exact (hmono_gap (Nat.succ_le_succ hmn)).trans hgap_m1

/--
`IsTopical` and zero-function specialization of
`graphW1_epsilonAccuracy_explicitLog_from_zero_of_twoStep_path_via_HGammaExplicitFixedPoint`.
-/
theorem graphW1_epsilonAccuracy_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (n : ℕ)
    (hn : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  exact graphW1_epsilonAccuracy_explicitLog_from_zero_of_twoStep_path_via_HGammaExplicitFixedPoint
    (Psi := Psi)
    (hPsi := SeminormNonexpansive_variationSeminormAsSeminorm_of_isTopical hT)
    (vStar := vStar) (v0 := (0 : ι → ℝ)) hfix
    halpha heps hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar
    (by simpa using (variationSeminorm_zero (ι := ι)))
    hphi_orbit n hn

/--
Successor-index `IsTopical` zero-function ε-accuracy convenience form in explicit-log constants.
-/
theorem graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (n : ℕ)
    (hn : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤
      (n + 1) + 1) :
    gap (n + 1) ≤ eps := by
  exact graphW1_epsilonAccuracy_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma
    Psi hT hfix halpha heps hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit (n + 1) hn

/--
Index-threshold helper for the `IsTopical` zero-function explicit-log ε-accuracy bridge.
-/
theorem
    graphW1_epsilonAccuracy_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (m n : ℕ)
    (hm : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ m + 1)
    (hmn : m ≤ n) :
    gap n ≤ eps := by
  have hgap_m :
      gap m ≤ eps :=
    graphW1_epsilonAccuracy_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma
      Psi hT hfix halpha heps hgap_res hres_ascent hmono_gap
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit m hm
  exact (hmono_gap hmn).trans hgap_m

/--
Successor-index index-threshold helper for the `IsTopical` zero-function explicit-log
ε-accuracy bridge.
-/
theorem
    graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (m n : ℕ)
    (hm : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤
      (m + 1) + 1)
    (hmn : m ≤ n) :
    gap (n + 1) ≤ eps := by
  have hgap_m1 :
      gap (m + 1) ≤ eps :=
    graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma
      Psi hT hfix halpha heps hgap_res hres_ascent hmono_gap
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit m hm
  exact (hmono_gap (Nat.succ_le_succ hmn)).trans hgap_m1

/--
Successor-index orbit wrapper for the `IsTopical` zero-function explicit-log HGamma route.
-/
theorem graphW1_orbit_bound_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (n : ℕ) :
    variationSeminorm ((Psi^[n + 1]) (0 : ι → ℝ)) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  exact graphW1_orbit_bound_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma
    Psi hT hfix hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar (n + 1)

/--
Ceiling-index dual-rate wrapper for the `IsTopical` zero-function explicit-log HGamma route.
-/
theorem graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ))) :
    gap (Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps)) ≤
      alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) /
        (Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) + 1 : ℝ) := by
  exact graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma
    Psi hT hfix halpha hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit
    (Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps))

/--
Ceiling-index ε-accuracy wrapper for the `IsTopical` zero-function explicit-log HGamma route.
-/
theorem graphW1_epsilonAccuracy_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ))) :
    gap (Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps)) ≤ eps := by
  have hn :
      Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤
        Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) + 1 :=
    Nat.le_add_right _ _
  exact graphW1_epsilonAccuracy_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma
    Psi hT hfix halpha heps hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit
    (Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps)) hn

/--
Successor-index ceiling wrapper for the `IsTopical` zero-function explicit-log orbit bound.
-/
theorem graphW1_orbit_bound_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma)) :
    variationSeminorm
        ((Psi^[Nat.ceil (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) / eps) + 1])
          (0 : ι → ℝ)) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  exact graphW1_orbit_bound_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma
    Psi hT hfix hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar
    (Nat.ceil (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) / eps))

/--
Successor-index ceiling wrapper for the `IsTopical` zero-function explicit-log dual-rate bound.
-/
theorem graphW1_dualRate_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ))) :
    gap (Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) + 1) ≤
      alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) /
        ((Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) + 1 : ℝ) +
          1) := by
  exact graphW1_dualRate_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma
    Psi hT hfix halpha hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit
    (Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps))

/--
Successor-index ceiling wrapper for the `IsTopical` zero-function explicit-log ε-accuracy.
-/
theorem graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ))) :
    gap (Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) + 1) ≤ eps := by
  let N : ℕ := Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps)
  have hN : N ≤ (N + 1) + 1 := by
    exact Nat.le_trans (Nat.le_add_right N 1) (Nat.le_add_right (N + 1) 1)
  have hsucc :
      gap (N + 1) ≤ eps :=
    graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma
      Psi hT hfix halpha heps hgap_res hres_ascent hmono_gap
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit N hN
  simpa [N] using hsucc

/--
Ceiling-index orbit wrapper for the `IsTopical` zero-function explicit-log HGamma route.
-/
theorem graphW1_orbit_bound_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma)) :
    variationSeminorm
        ((Psi^[Nat.ceil (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) / eps)])
          (0 : ι → ℝ)) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  exact graphW1_orbit_bound_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma
    Psi hT hfix hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar
    (Nat.ceil (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) / eps))

/--
Index-threshold helper from the ceiling-index explicit-log dual-rate wrapper.
-/
theorem graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (n : ℕ)
    (hn : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ n) :
    gap n ≤
      alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) /
        (Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) + 1 : ℝ) := by
  let N : ℕ := Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps)
  have hN :
      gap N ≤
        alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) /
          (N + 1 : ℝ) := by
    simpa [N] using
      (graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index
        Psi hT hfix halpha hgap_res hres_ascent hmono_gap
        hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
        hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit)
  have hmono : gap n ≤ gap N := by
    simpa [N] using (hmono_gap hn)
  exact hmono.trans hN

/--
Successor-index index-threshold helper from the ceiling-index explicit-log ε-accuracy wrapper.
-/
theorem
    graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (n : ℕ)
    (hn : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ n) :
    gap (n + 1) ≤ eps := by
  let N : ℕ := Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps)
  have hN :
      gap (N + 1) ≤ eps := by
    simpa [N] using
      (graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index
        Psi hT hfix halpha heps hgap_res hres_ascent hmono_gap
        hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
        hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit)
  have hmono : gap (n + 1) ≤ gap (N + 1) := by
    simpa [N] using (hmono_gap (Nat.succ_le_succ hn))
  exact hmono.trans hN

/--
Index-threshold orbit helper for the explicit-log `IsTopical` HGamma route.
-/
theorem graphW1_orbit_bound_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (n : ℕ)
    (_hn : Nat.ceil (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) / eps) ≤ n) :
    variationSeminorm ((Psi^[n]) (0 : ι → ℝ)) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  exact graphW1_orbit_bound_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma
    Psi hT hfix hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar n

/--
Successor-index threshold orbit helper for the explicit-log `IsTopical` HGamma route.
-/
theorem graphW1_orbit_bound_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (n : ℕ)
    (_hn : Nat.ceil (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) / eps) ≤ n) :
    variationSeminorm ((Psi^[n + 1]) (0 : ι → ℝ)) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  exact graphW1_orbit_bound_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma
    Psi hT hfix hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar n

/--
Natural-bound threshold orbit helper for the explicit-log `IsTopical` HGamma route.
-/
theorem graphW1_orbit_bound_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_of_natBound
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (n N : ℕ)
    (hceilN : Nat.ceil (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) / eps) ≤ N)
    (hNn : N ≤ n) :
    variationSeminorm ((Psi^[n]) (0 : ι → ℝ)) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) :=
  graphW1_orbit_bound_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_of_le_index
    Psi hT hfix hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar n (hceilN.trans hNn)

/--
Successor-index natural-bound threshold orbit helper for the explicit-log `IsTopical` HGamma
route.
-/
theorem graphW1_orbit_bound_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_of_natBound
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (n N : ℕ)
    (hceilN : Nat.ceil (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) / eps) ≤ N)
    (hNn : N ≤ n) :
    variationSeminorm ((Psi^[n + 1]) (0 : ι → ℝ)) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) :=
  graphW1_orbit_bound_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_of_le_index
    Psi hT hfix hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar n (hceilN.trans hNn)

/--
Natural-bound helper derived from the ceiling-index explicit-log dual-rate wrapper.
-/
theorem graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_natBound
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (n N : ℕ)
    (hceilN :
      Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ N)
    (hNn : N ≤ n) :
    gap n ≤
      alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) /
        (Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) + 1 : ℝ) :=
  graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_le_index
    Psi hT hfix halpha hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit n (hceilN.trans hNn)

/--
Natural-bound helper derived from the successor-index ceiling explicit-log ε-accuracy wrapper.
-/
theorem
    graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_natBound
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (n N : ℕ)
    (hceilN :
      Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ N)
    (hNn : N ≤ n) :
    gap (n + 1) ≤ eps :=
  graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_le_index
    Psi hT hfix halpha heps hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit n (hceilN.trans hNn)

/--
Monotone-index transfer from a proved ceiling-threshold dual-rate certification.
-/
theorem graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_ge_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (n : ℕ)
    (hn : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ n)
    (m : ℕ)
    (hnm : n ≤ m) :
    gap m ≤
      alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) /
        (Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) + 1 : ℝ) :=
  graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_natBound
    Psi hT hfix halpha hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit m n hn hnm

/--
Monotone-index transfer from a proved ceiling-threshold successor-index ε-accuracy certification.
-/
theorem
    graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_ge_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (n : ℕ)
    (hn : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ n)
    (m : ℕ)
    (hnm : n ≤ m) :
    gap (m + 1) ≤ eps :=
  graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_natBound
    Psi hT hfix halpha heps hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit m n hn hnm

/--
Threshold-transport helper from the ceiling-index explicit-log dual-rate wrapper.
-/
theorem graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (k n : ℕ)
    (hk : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ k)
    (hkn : k ≤ n) :
    gap n ≤
      alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) /
        (Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) + 1 : ℝ) :=
  graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_natBound
    Psi hT hfix halpha hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit n k hk hkn

/--
Threshold-transport helper from the successor-index ceiling explicit-log ε-accuracy wrapper.
-/
theorem
    graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (k n : ℕ)
    (hk : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ k)
    (hkn : k ≤ n) :
    gap (n + 1) ≤ eps :=
  graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_natBound
    Psi hT hfix halpha heps hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit n k hk hkn

/--
Monotone threshold-transport helper from the ceiling-index explicit-log dual-rate wrapper.
-/
theorem
    graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_and_ge_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (k n m : ℕ)
    (hk : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ k)
    (hkn : k ≤ n)
    (hnm : n ≤ m) :
    gap m ≤
      alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) /
        (Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) + 1 : ℝ) :=
  graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_ge_index
    Psi hT hfix halpha hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit n (le_trans hk hkn) m hnm

/--
Monotone threshold-transport helper from the successor-index ceiling explicit-log ε-accuracy
wrapper.
-/
theorem
    graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_and_ge_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (k n m : ℕ)
    (hk : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ k)
    (hkn : k ≤ n)
    (hnm : n ≤ m) :
    gap (m + 1) ≤ eps :=
  graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_ge_index
    Psi hT hfix halpha heps hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit n (le_trans hk hkn) m hnm

/--
At-ceiling-index threshold helper for the explicit-log dual-rate wrapper.
-/
theorem graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_at_ceil_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (k : ℕ)
    (hk : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ k) :
    gap k ≤
      alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) /
        (Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) + 1 : ℝ) := by
  exact graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le
    Psi hT hfix halpha hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit k k hk (Nat.le_refl _)

/--
At-ceiling-index threshold helper for the explicit-log successor-index ε-accuracy wrapper.
-/
theorem
    graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_at_ceil_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (k : ℕ)
    (hk : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ k) :
    gap (k + 1) ≤ eps := by
  exact graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le
    Psi hT hfix halpha heps hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit k k hk (Nat.le_refl _)

/--
Monotone index transport from an at-ceiling threshold certificate for the explicit-log dual-rate
wrapper.
-/
theorem
    graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_at_ceil_index_of_ge_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (k m : ℕ)
    (hk : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ k)
    (hkm : k ≤ m) :
    gap m ≤
      alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) /
        (Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) + 1 : ℝ) := by
  exact graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_and_ge_index
    Psi hT hfix halpha hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit k k m hk (Nat.le_refl _) hkm

/--
Monotone index transport from an at-ceiling threshold certificate for the explicit-log
successor-index ε-accuracy wrapper.
-/
theorem
    graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_at_ceil_index_of_ge_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (k m : ℕ)
    (hk : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ k)
    (hkm : k ≤ m) :
    gap (m + 1) ≤ eps := by
  exact graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_and_ge_index
    Psi hT hfix halpha heps hgap_res hres_ascent hmono_gap
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
    hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit k k m hk (Nat.le_refl _) hkm

/--
Index-threshold specialization of the at-ceiling threshold transport helper for the explicit-log
dual-rate wrapper.
-/
theorem
    graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_at_ceil_index_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (k n : ℕ)
    (hk : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ k)
    (hkn : k ≤ n) :
    gap n ≤
      alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) /
        (Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) + 1 : ℝ) := by
  exact
    graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_at_ceil_index_of_ge_index
      Psi hT hfix halpha hgap_res hres_ascent hmono_gap
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit k n hk hkn

/--
Index-threshold specialization of the at-ceiling threshold transport helper for the explicit-log
successor-index ε-accuracy wrapper.
-/
theorem
    graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_at_ceil_index_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (k n : ℕ)
    (hk : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ k)
    (hkn : k ≤ n) :
    gap (n + 1) ≤ eps := by
  exact
    graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_at_ceil_index_of_ge_index
      Psi hT hfix halpha heps hgap_res hres_ascent hmono_gap
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit k n hk hkn

/--
Natural-bound specialization of the at-ceiling threshold transport helper for the explicit-log
dual-rate wrapper.
-/
theorem
    graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_at_ceil_index_of_natBound
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (k N : ℕ)
    (hk : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ k)
    (hkn : k ≤ N) :
    gap N ≤
      alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) /
        (Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) + 1 : ℝ) := by
  exact
    graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_at_ceil_index_of_le_index
      Psi hT hfix halpha hgap_res hres_ascent hmono_gap
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit k N hk hkn

/--
Natural-bound specialization of the at-ceiling threshold transport helper for the explicit-log
successor-index ε-accuracy wrapper.
-/
theorem
    graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_at_ceil_index_of_natBound
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (k N : ℕ)
    (hk : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ k)
    (hkn : k ≤ N) :
    gap (N + 1) ≤ eps := by
  exact
    graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_at_ceil_index_of_le_index
      Psi hT hfix halpha heps hgap_res hres_ascent hmono_gap
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit k N hk hkn

/--
Natural-bound then monotone-index transport for the at-ceiling threshold explicit-log
dual-rate wrapper.
-/
theorem
    graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_at_ceil_index_of_natBound_of_ge_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (k N m : ℕ)
    (hk : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ k)
    (hkN : k ≤ N)
    (hNm : N ≤ m) :
    gap m ≤
      alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) /
        (Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) + 1 : ℝ) := by
  exact
    graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_at_ceil_index_of_ge_index
      Psi hT hfix halpha hgap_res hres_ascent hmono_gap
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit k m hk (le_trans hkN hNm)

/--
Natural-bound then monotone-index transport for the at-ceiling threshold explicit-log
successor-index `ε`-accuracy wrapper.
-/
theorem
    graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_at_ceil_index_of_natBound_of_ge_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (k N m : ℕ)
    (hk : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ k)
    (hkN : k ≤ N)
    (hNm : N ≤ m) :
    gap (m + 1) ≤ eps := by
  exact
    graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_at_ceil_index_of_ge_index
      Psi hT hfix halpha heps hgap_res hres_ascent hmono_gap
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit k m hk (le_trans hkN hNm)

/--
Threshold-index specialization of natural-bound transport for the at-ceiling threshold
explicit-log dual-rate wrapper.
-/
theorem
    graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_at_ceil_index_of_natBound_at_threshold_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (k : ℕ)
    (hk : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ k) :
    gap k ≤
      alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) /
        (Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) + 1 : ℝ) := by
  exact
    graphW1_dualRate_explicitLog_from_zeroFn_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_at_ceil_index_of_natBound
      Psi hT hfix halpha hgap_res hres_ascent hmono_gap
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit k k hk (Nat.le_refl _)

/--
Threshold-index specialization of natural-bound transport for the at-ceiling threshold
explicit-log successor-index `ε`-accuracy wrapper.
-/
theorem
    graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_at_ceil_index_of_natBound_at_threshold_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi gap residual : ℕ → ℝ}
    {alpha kappa B cost gamma eps : ℝ}
    {n_nodes : ℕ}
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hmono_gap : Antitone gap)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hvStar : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hphi_orbit :
      ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) (0 : ι → ℝ)))
    (k : ℕ)
    (hk : Nat.ceil (alpha * (4 * (graphDiam : ℝ) * (cost + Real.log n_nodes)) / eps) ≤ k) :
    gap (k + 1) ≤ eps := by
  exact
    graphW1_epsilonAccuracy_explicitLog_from_zeroFn_succ_of_twoStep_path_IsTopical_via_HGamma_at_ceil_index_of_threshold_le_at_ceil_index_of_natBound
      Psi hT hfix halpha heps hgap_res hres_ascent hmono_gap
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps
      hkappa_from_path hgamma hbase_nonneg hvStar hphi_orbit k k hk (Nat.le_refl _)

/--
Paper-facing operation-bound theorem for Theorem `thm:graphw1-complexity`.

This is the arithmetic layer corresponding to the statement
`O(p * diameter(E)^3 / eps^4)` operations up to logarithmic factors in `n`, under the side
condition `p=o(1/log(1/eps))`.

The analytic graph-W₁ convergence proof supplies `haccuracy` and the iteration budget; sparse
implementation supplies the per-sweep and total-operation certificates. Lean then proves the
displayed operation bound and carries the little-o edge-count regime as an explicit hypothesis.
-/
theorem graphW1_sinkhornFlow_complexity_from_operationBounds
    {w1Error : ℕ → ℝ}
    {eps p graphDiam logFactor iterationBudget perSweepOps operationCount : ℝ}
    {pOfEps : ℝ → ℝ}
    (k : ℕ)
    (heps : 0 < eps)
    (hp_nonneg : 0 ≤ p)
    (haccuracy : w1Error k ≤ eps)
    (hk_iter : (k : ℝ) ≤ iterationBudget)
    (hiter_budget : iterationBudget ≤ logFactor * graphDiam ^ 3 / eps ^ 4)
    (hper_sweep : perSweepOps ≤ p)
    (hoperation : operationCount ≤ (k : ℝ) * perSweepOps)
    (hp_eval : p = pOfEps eps)
    (hp_littleO : graphW1LittleOEdgeRegime pOfEps) :
    0 < eps ∧
      w1Error k ≤ eps ∧
      operationCount ≤ logFactor * p * graphDiam ^ 3 / eps ^ 4 ∧
      p = pOfEps eps ∧
      graphW1LittleOEdgeRegime pOfEps := by
  have hk_nonneg : 0 ≤ (k : ℝ) := Nat.cast_nonneg k
  have hstep_bound : (k : ℝ) * perSweepOps ≤ (k : ℝ) * p :=
    mul_le_mul_of_nonneg_left hper_sweep hk_nonneg
  have hiter_to_budget : (k : ℝ) ≤ logFactor * graphDiam ^ 3 / eps ^ 4 :=
    hk_iter.trans hiter_budget
  have hbudget_step :
      (k : ℝ) * p ≤ (logFactor * graphDiam ^ 3 / eps ^ 4) * p :=
    mul_le_mul_of_nonneg_right hiter_to_budget hp_nonneg
  have hop :
      operationCount ≤ (logFactor * graphDiam ^ 3 / eps ^ 4) * p :=
    hoperation.trans (hstep_bound.trans hbudget_step)
  have htarget :
      (logFactor * graphDiam ^ 3 / eps ^ 4) * p =
        logFactor * p * graphDiam ^ 3 / eps ^ 4 := by
    ring
  rw [htarget] at hop
  exact ⟨heps, haccuracy, hop, hp_eval, hp_littleO⟩

/--
Structured-certificate version of Theorem `thm:graphw1-complexity`.

Compared with `graphW1_sinkhornFlow_complexity_from_operationBounds`, this endpoint packages
the epsilon-accuracy, iteration-budget, sparse-sweep, operation-count, and little-o side conditions
into `GraphW1OperationBudgetCertificate`.  The proof still performs the arithmetic composition in
Lean, but the Comparator-facing statement now exposes a named algorithmic certificate rather than
a loose list of scalar hypotheses.
-/
theorem graphW1_sinkhornFlow_complexity_from_operationBudgetCertificate
    {w1Error : ℕ → ℝ}
    {eps p graphDiam logFactor iterationBudget perSweepOps operationCount : ℝ}
    {pOfEps : ℝ → ℝ}
    (k : ℕ)
    (hcert :
      GraphW1OperationBudgetCertificate w1Error eps p graphDiam logFactor
        iterationBudget perSweepOps operationCount pOfEps k) :
    0 < eps ∧
      w1Error k ≤ eps ∧
      operationCount ≤ logFactor * p * graphDiam ^ 3 / eps ^ 4 ∧
      p = pOfEps eps ∧
      graphW1LittleOEdgeRegime pOfEps := by
  exact graphW1_sinkhornFlow_complexity_from_operationBounds
    (w1Error := w1Error)
    (eps := eps)
    (p := p)
    (graphDiam := graphDiam)
    (logFactor := logFactor)
    (iterationBudget := iterationBudget)
    (perSweepOps := perSweepOps)
    (operationCount := operationCount)
    (pOfEps := pOfEps)
    k
    hcert.eps_pos
    hcert.edge_nonneg
    hcert.accuracy
    hcert.iteration_index_le_budget
    hcert.iteration_budget
    hcert.per_sweep_ops
    hcert.operation_count
    hcert.edge_count_eval
    hcert.edge_count_littleO

end GraphW1
end Applications
end KLProjection
end FlowSinkhorn
