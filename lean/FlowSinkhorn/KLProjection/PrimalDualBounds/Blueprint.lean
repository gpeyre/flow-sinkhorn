import FlowSinkhorn.KLProjection.PrimalDualBounds.FixedPointControl
import FlowSinkhorn.KLProjection.DualConvergence.Rate
import FlowSinkhorn.KLProjection.Topical

/-!
# Generic blueprint packaging

This module is meant to be the Lean entry point for the full generic convergence blueprint of the
paper.

Paper role:
it packages Sections 3 and 4 together into a single theorem family that application modules can
instantiate by proving explicit bounds on `X_γ`, `H_γ`, `κ`, and the block quotient norm.

Intended theorem names:
- `genericBlueprint_dualRate`;
- `genericBlueprint_uniformOrbitBound`;
- `genericBlueprint_primalMassBound`;
- `genericBlueprint_section3And4Bounds`;
- `genericBlueprint_applicationBlueprint`;
- `genericBlueprint_applicationEpsilonAccuracy_fromBlueprint`;
- `genericBlueprint_applicationEpsilonAccuracy`;
- `genericBlueprint_singleApplicationEntrypoint`;
- `genericBlueprint_applicationRecipe`;
- `genericBlueprint_complexity`.

Recommended application entrypoints:
- `genericBlueprint_applicationEpsilonAccuracy` when you want the final `eps`-accuracy endpoint
  from the full explicit hypothesis package;
- `genericBlueprint_singleApplicationEntrypoint` when you already have the Section 3+4 conclusion
  and only need the final `eps`-accuracy endpoint;
- `genericBlueprint_applicationBlueprint` for the full Section 3+4 conclusion from explicit
  hypotheses;
- `genericBlueprint_applicationRecipe` and
  `genericBlueprint_applicationEpsilonAccuracy_of_section3And4Bounds` as compatibility wrappers
  for older call sites.

Design note:
keeping this packaging in its own file makes the application modules read like the paper: each
application imports the generic blueprint and then discharges the hypotheses with concrete bounds.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace PrimalDualBounds

open Function

variable {𝕜 E : Type*}
variable [NormedField 𝕜] [AddCommGroup E] [Module 𝕜 E]

/-! ## Section 3: O(1/k) dual rate -/

/--
Section 3 master rate theorem, paper-facing form.

This is the `O(1/k)` input that applications discharge with concrete constants.
-/
theorem genericBlueprint_dualRate
    {phi gap residual : ℕ → ℝ}
    {alpha B : ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (n : ℕ) :
    gap n ≤ (alpha * B) / (n + 1 : ℝ) :=
  DualConvergence.dualRate_O_one_over_k_of_ascent_gap_control
    halpha hgap_res hres_ascent hphi_bound hmono_gap n

/-! ## Section 4: uniform orbit bound -/

/--
Section 4 orbit budget theorem in canonical `H_γ/κ` form.

Applications usually call this after proving a fixed-point estimate for the optimizer.
-/
theorem genericBlueprint_uniformOrbitBound
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ p u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma :=
  by
    simpa [hGammaKappaBudget] using
      uniformIterateBound_of_nonexpansive_of_budget
        p Psi hPsi hfix hbound k

/--
Primal confinement transfer from a dual budget.

This is the paper's monotone `X_γ`-from-dual-radius step.
-/
theorem genericBlueprint_primalMassBound
    {d U X : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ U) :
    X ≤ Xbound U :=
  primalBound_fromDualBudget hmono hprimal_at_d hdual

/--
Abstract Section 3+4 blueprint package.

This keeps the orbit budget as an abstract constant, which is convenient for intermediate proofs
before the application-specific `H_γ/κ` estimate is substituted.
-/
theorem genericBlueprint_section3And4Bounds
    {phi gap residual : ℕ → ℝ}
    {alpha Brate Borbit d budget X : ℝ}
    {p : Seminorm 𝕜 E} (Psi : E → E)
    {Xbound : ℝ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    (horbit : p uStar ≤ Borbit)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (n k : ℕ) :
    gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
    p ((Psi^[k]) u0) ≤ p u0 + 2 * Borbit ∧
    X ≤ Xbound budget := by
  constructor
  · exact genericBlueprint_dualRate halpha hgap_res hres_ascent hphi_bound hmono_gap n
  · constructor
    · exact uniformIterateBound_of_nonexpansive_of_budget p Psi hPsi hfix horbit k
    · exact genericBlueprint_primalMassBound hmonoX hprimal_at_d hdual

/--
Application-facing Section 3+4 blueprint package.

Compared to `genericBlueprint_section3And4Bounds`, this version uses the canonical `H_γ/κ`
orbit budget directly, which is the form most applications will want to import.
-/
theorem genericBlueprint_applicationBlueprint
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma : ℝ}
    {p : Seminorm 𝕜 E} (Psi : E → E)
    {Xbound : ℝ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    (horbit : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (n k : ℕ) :
    gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
    p ((Psi^[k]) u0) ≤ p u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
    X ≤ Xbound budget := by
  constructor
  · exact genericBlueprint_dualRate halpha hgap_res hres_ascent hphi_bound hmono_gap n
  · constructor
    · exact uniformIterateBound_of_nonexpansive_of_HGamma_kappa p Psi hPsi hfix horbit k
    · exact genericBlueprint_primalMassBound hmonoX hprimal_at_d hdual

/-! ## Combined blueprint: bias + rate → ε accuracy -/

/--
Final abstract packaging of Theorem `thm:approx-linprog`:
bias + dual-rate budget imply an `eps`-accurate unregularized objective bound.
-/
theorem genericBlueprint_complexity
    {F0 FgammaStar bias C eps : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ C / (n + 1 : ℝ))
    (n : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps :=
  DualConvergence.regularizedApproximation_complexity hbias hrate n hbudget

/--
Tiny lemma: lift the Section 3+4 blueprint conclusion to the final `eps`-accuracy statement.

This is the small boilerplate reducer for call sites that already proved the bundled blueprint
hypothesis.
-/
theorem genericBlueprint_applicationEpsilonAccuracy_fromBlueprint
    {gap : ℕ → ℝ}
    {alpha Brate budget X kappa cost gamma hGamma bias C eps : ℝ}
    {p : Seminorm 𝕜 E} (Psi : E → E) {u0 : E}
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    {n k : ℕ}
    (happ :
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      p ((Psi^[k]) u0) ≤ p u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ C / (n + 1 : ℝ))
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps :=
  by
    have _ := happ
    exact genericBlueprint_complexity hbias hrate n hbudget

/--
Canonical application entrypoint.

This is the theorem applications should use first: it takes the full explicit hypothesis package
and returns the final `eps`-accuracy guarantee directly.
-/
theorem genericBlueprint_applicationEpsilonAccuracy
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
    (horbit : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ C / (n + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps := by
  exact genericBlueprint_applicationEpsilonAccuracy_fromBlueprint
    (Psi := Psi)
    (u0 := u0)
    (happ := genericBlueprint_applicationBlueprint
      (Psi := Psi) halpha hgap_res hres_ascent hphi_bound hmono_gap hPsi
      (uStar := uStar) (u0 := u0) hfix horbit hmonoX hprimal_at_d hdual n k)
    hbias hrate hbudget

/--
Compatibility wrapper: full Section 3+4 conclusion plus epsilon accuracy.

This remains available for callers that already have the bundled Section 3+4 conclusion.
-/
theorem genericBlueprint_applicationEpsilonAccuracy_of_section3And4Bounds
    {gap : ℕ → ℝ}
    {alpha Brate budget X kappa cost gamma hGamma bias C eps : ℝ}
    {p : Seminorm 𝕜 E} (Psi : E → E) {u0 : E}
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    {n k : ℕ}
    (hblueprint :
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      p ((Psi^[k]) u0) ≤ p u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ C / (n + 1 : ℝ))
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps :=
  genericBlueprint_applicationEpsilonAccuracy_fromBlueprint
    (Psi := Psi) (u0 := u0) (happ := hblueprint) hbias hrate hbudget

/--
Compatibility alias keeping the historical `applicationRecipe` name.
-/
theorem genericBlueprint_applicationRecipe
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
    (horbit : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (n k : ℕ)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ C / (n + 1 : ℝ))
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps :=
  genericBlueprint_applicationEpsilonAccuracy
    (Psi := Psi) (u0 := u0)
    halpha hgap_res hres_ascent hphi_bound hmono_gap hPsi hfix horbit hmonoX
    hprimal_at_d hdual hbias hrate n k hbudget

/--
Compatibility alias keeping the historical onboarding name.
-/
theorem genericBlueprint_singleApplicationEntrypoint
    {gap : ℕ → ℝ}
    {alpha Brate budget X kappa cost gamma hGamma bias C eps : ℝ}
    {p : Seminorm 𝕜 E} (Psi : E → E) {u0 : E}
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    {n k : ℕ}
    (hblueprint :
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      p ((Psi^[k]) u0) ≤ p u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ C / (n + 1 : ℝ))
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps :=
  genericBlueprint_applicationEpsilonAccuracy_of_section3And4Bounds
    (Psi := Psi) (u0 := u0) (hblueprint := hblueprint) hbias hrate hbudget

/--
Full generic assumptions imply both final epsilon-accuracy and the base orbit radius.

This convenience wrapper targets call sites that want the final objective guarantee and
the iterate bound in one theorem, without switching to `IsTopical`-specific APIs.
-/
theorem genericBlueprint_applicationEpsilonAccuracy_and_orbit_with_base
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    {p : Seminorm 𝕜 E} (Psi : E → E)
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    (horbit : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      p ((Psi^[k]) u0) ≤ p u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma := by
  have hblueprint :
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
        p ((Psi^[k]) u0) ≤ p u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
        X ≤ Xbound budget :=
    genericBlueprint_applicationBlueprint
      (Psi := Psi) (uStar := uStar) (u0 := u0)
      halpha hgap_res hres_ascent hphi_bound hmono_gap hPsi
      hfix horbit hmonoX hprimal_at_d hdual n k
  refine ⟨?_, hblueprint.2.1⟩
  exact genericBlueprint_applicationEpsilonAccuracy_fromBlueprint
    (Psi := Psi) (u0 := u0) (happ := hblueprint) hbias hrate hbudget

/--
Full generic assumptions imply epsilon-accuracy and a zero-seed orbit radius.

This is the `p u0 = 0` specialization of
`genericBlueprint_applicationEpsilonAccuracy_and_orbit_with_base`.
-/
theorem genericBlueprint_applicationEpsilonAccuracy_and_orbit_from_zero
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    {p : Seminorm 𝕜 E} (Psi : E → E)
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    (horbit : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hzero : p u0 = 0)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      p ((Psi^[k]) u0) ≤ 2 * hGammaKappaBudget kappa cost gamma hGamma := by
  have hpair := genericBlueprint_applicationEpsilonAccuracy_and_orbit_with_base
    (Psi := Psi) (uStar := uStar) (u0 := u0)
    halpha hgap_res hres_ascent hphi_bound hmono_gap hPsi
    hfix horbit hmonoX hprimal_at_d hdual hbias hrate n k hbudget
  refine ⟨hpair.1, ?_⟩
  rw [hzero, zero_add] at hpair
  exact hpair.2

/--
Full generic assumptions imply epsilon-accuracy, dual-rate certification, and base orbit control.

This bundles the three downstream ingredients most often consumed together by application modules.
-/
theorem genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_orbit_with_base
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    {p : Seminorm 𝕜 E} (Psi : E → E)
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    (horbit : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      p ((Psi^[k]) u0) ≤ p u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma := by
  have hpair := genericBlueprint_applicationEpsilonAccuracy_and_orbit_with_base
    (Psi := Psi) (uStar := uStar) (u0 := u0)
    halpha hgap_res hres_ascent hphi_bound hmono_gap hPsi
    hfix horbit hmonoX hprimal_at_d hdual hbias hrate n k hbudget
  refine ⟨hpair.1, ?_, hpair.2⟩
  exact genericBlueprint_dualRate halpha hgap_res hres_ascent hphi_bound hmono_gap n

/--
Full generic assumptions imply epsilon-accuracy, dual-rate certification, and zero-seed orbit.

This is the `p u0 = 0` specialization of
`genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_orbit_with_base`.
-/
theorem genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_orbit_from_zero
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    {p : Seminorm 𝕜 E} (Psi : E → E)
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    (horbit : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hzero : p u0 = 0)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      p ((Psi^[k]) u0) ≤ 2 * hGammaKappaBudget kappa cost gamma hGamma := by
  have hpack := genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_orbit_with_base
    (Psi := Psi) (uStar := uStar) (u0 := u0)
    halpha hgap_res hres_ascent hphi_bound hmono_gap hPsi
    hfix horbit hmonoX hprimal_at_d hdual hbias hrate n k hbudget
  refine ⟨hpack.1, hpack.2.1, ?_⟩
  have horbitBase : p ((Psi^[k]) u0) ≤ p u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma :=
    hpack.2.2
  rw [hzero, zero_add] at horbitBase
  exact horbitBase

/--
Full generic assumptions imply epsilon, dual-rate, iterate nonexpansiveness, and base orbit.

This extends
`genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_orbit_with_base` by packaging
the nonexpansive control of the iterate map `Psi^[k]`.
-/
theorem genericBlueprint_applicationEpsilonDualRate_and_iterate_nonexpansive_and_orbit_with_base
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    {p : Seminorm 𝕜 E} (Psi : E → E)
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    (horbit : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive p (Psi^[k]) ∧
      p ((Psi^[k]) u0) ≤ p u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma := by
  have hpack := genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_orbit_with_base
    (Psi := Psi) (uStar := uStar) (u0 := u0)
    halpha hgap_res hres_ascent hphi_bound hmono_gap hPsi
    hfix horbit hmonoX hprimal_at_d hdual hbias hrate n k hbudget
  refine ⟨hpack.1, hpack.2.1, ?_, hpack.2.2⟩
  exact SeminormNonexpansive_iterate p Psi hPsi k

/--
Full generic assumptions imply epsilon, dual-rate, iterate nonexpansiveness, and zero-base orbit.

This is the `p u0 = 0` specialization of
`genericBlueprint_applicationEpsilonDualRate_and_iterate_nonexpansive_and_orbit_with_base`.
-/
theorem genericBlueprint_applicationEpsilonDualRate_and_iterate_nonexpansive_and_orbit_from_zero
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    {p : Seminorm 𝕜 E} (Psi : E → E)
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    (horbit : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hzero : p u0 = 0)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive p (Psi^[k]) ∧
      p ((Psi^[k]) u0) ≤ 2 * hGammaKappaBudget kappa cost gamma hGamma := by
  have hpack :=
    genericBlueprint_applicationEpsilonDualRate_and_iterate_nonexpansive_and_orbit_with_base
      (Psi := Psi) (uStar := uStar) (u0 := u0)
      halpha hgap_res hres_ascent hphi_bound hmono_gap hPsi
      hfix horbit hmonoX hprimal_at_d hdual hbias hrate n k hbudget
  refine ⟨hpack.1, hpack.2.1, hpack.2.2.1, ?_⟩
  have horbitBase :
      p ((Psi^[k]) u0) ≤ p u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma :=
    hpack.2.2.2
  rw [hzero, zero_add] at horbitBase
  exact horbitBase

/--
Full generic assumptions imply epsilon, dual-rate, primal-mass control, and base orbit.

This adds the Section-4 primal confinement output `X ≤ Xbound budget` to the reusable
bundle used by application modules.
-/
theorem genericBlueprint_applicationEpsilonDualRate_and_primalMass_and_orbit_with_base
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    {p : Seminorm 𝕜 E} (Psi : E → E)
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    (horbit : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      p ((Psi^[k]) u0) ≤ p u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget := by
  have hblueprint :
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
        p ((Psi^[k]) u0) ≤ p u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
        X ≤ Xbound budget :=
    genericBlueprint_applicationBlueprint
      (Psi := Psi) (uStar := uStar) (u0 := u0)
      halpha hgap_res hres_ascent hphi_bound hmono_gap hPsi
      hfix horbit hmonoX hprimal_at_d hdual n k
  refine ⟨?_, hblueprint.1, hblueprint.2.1, hblueprint.2.2⟩
  exact genericBlueprint_applicationEpsilonAccuracy_fromBlueprint
    (Psi := Psi) (u0 := u0) (happ := hblueprint) hbias hrate hbudget

/--
Full generic assumptions imply epsilon, dual-rate, primal-mass control, and zero-base orbit.

This is the `p u0 = 0` specialization of
`genericBlueprint_applicationEpsilonDualRate_and_primalMass_and_orbit_with_base`.
-/
theorem genericBlueprint_applicationEpsilonDualRate_and_primalMass_and_orbit_from_zero
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    {p : Seminorm 𝕜 E} (Psi : E → E)
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    (horbit : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hzero : p u0 = 0)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      p ((Psi^[k]) u0) ≤ 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget := by
  have hpack :=
    genericBlueprint_applicationEpsilonDualRate_and_primalMass_and_orbit_with_base
      (Psi := Psi) (uStar := uStar) (u0 := u0)
      halpha hgap_res hres_ascent hphi_bound hmono_gap hPsi
      hfix horbit hmonoX hprimal_at_d hdual hbias hrate n k hbudget
  refine ⟨hpack.1, hpack.2.1, ?_, hpack.2.2.2⟩
  have horbitBase :
      p ((Psi^[k]) u0) ≤ p u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma :=
    hpack.2.2.1
  rw [hzero, zero_add] at horbitBase
  exact horbitBase

/--
Full generic assumptions imply epsilon, dual-rate, primal-mass, iterate nonexpansive, and orbit.

This is a one-call downstream bundle carrying both Section-4 controls plus iterate
nonexpansiveness for later perturbation/quotient arguments.
-/
theorem genericBlueprint_applicationEpsilonDualRate_and_primalMass_and_iterate_and_orbit_with_base
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    {p : Seminorm 𝕜 E} (Psi : E → E)
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    (horbit : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive p (Psi^[k]) ∧
      p ((Psi^[k]) u0) ≤ p u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget := by
  have hpack :=
    genericBlueprint_applicationEpsilonDualRate_and_primalMass_and_orbit_with_base
      (Psi := Psi) (uStar := uStar) (u0 := u0)
      halpha hgap_res hres_ascent hphi_bound hmono_gap hPsi
      hfix horbit hmonoX hprimal_at_d hdual hbias hrate n k hbudget
  refine ⟨hpack.1, hpack.2.1, ?_, hpack.2.2.1, hpack.2.2.2⟩
  exact SeminormNonexpansive_iterate p Psi hPsi k

/--
Zero-seed specialization of the full generic bundle
`epsilon + dual-rate + primal-mass + iterate-nonexpansive + orbit`.
-/
theorem genericBlueprint_applicationEpsilonDualRate_and_primalMass_and_iterate_and_orbit_from_zero
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    {p : Seminorm 𝕜 E} (Psi : E → E)
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    (horbit : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hzero : p u0 = 0)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive p (Psi^[k]) ∧
      p ((Psi^[k]) u0) ≤ 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget := by
  have hpack :=
    genericBlueprint_applicationEpsilonDualRate_and_primalMass_and_iterate_and_orbit_with_base
      (p := p) (Psi := Psi) (uStar := uStar) (u0 := u0)
      halpha hgap_res hres_ascent hphi_bound hmono_gap hPsi
      hfix horbit hmonoX hprimal_at_d hdual hbias hrate n k hbudget
  refine ⟨hpack.1, hpack.2.1, hpack.2.2.1, ?_, hpack.2.2.2.2⟩
  have horbitBase :
      p ((Psi^[k]) u0) ≤ p u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma :=
    hpack.2.2.2.1
  rw [hzero, zero_add] at horbitBase
  exact horbitBase

/--
Successor-index form of the full generic bundle
`epsilon + dual-rate + primal-mass + iterate-nonexpansive + orbit`.
-/
theorem genericBlueprint_applicationEpsilonDualRate_and_primalMass_and_iterate_and_orbit_with_base_succ
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    {p : Seminorm 𝕜 E} (Psi : E → E)
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    (horbit : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive p (Psi^[k + 1]) ∧
      p ((Psi^[k + 1]) u0) ≤ p u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget := by
  simpa [Nat.succ_eq_add_one] using
    genericBlueprint_applicationEpsilonDualRate_and_primalMass_and_iterate_and_orbit_with_base
      (p := p) (Psi := Psi) (uStar := uStar) (u0 := u0)
      halpha hgap_res hres_ascent hphi_bound hmono_gap hPsi
      hfix horbit hmonoX hprimal_at_d hdual hbias hrate n (k + 1) hbudget

/--
Successor-index zero-seed form of the full generic bundle
`epsilon + dual-rate + primal-mass + iterate-nonexpansive + orbit`.
-/
theorem genericBlueprint_applicationEpsilonDualRate_and_primalMass_and_iterate_and_orbit_from_zero_succ
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    {p : Seminorm 𝕜 E} (Psi : E → E)
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    (horbit : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hzero : p u0 = 0)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive p (Psi^[k + 1]) ∧
      p ((Psi^[k + 1]) u0) ≤ 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget := by
  simpa [Nat.succ_eq_add_one] using
    genericBlueprint_applicationEpsilonDualRate_and_primalMass_and_iterate_and_orbit_from_zero
      (p := p) (Psi := Psi) (uStar := uStar) (u0 := u0)
      halpha hgap_res hres_ascent hphi_bound hmono_gap hPsi
      hfix horbit hzero hmonoX hprimal_at_d hdual hbias hrate n (k + 1) hbudget


/-! ## Bridge: topical maps → variation nonexpansiveness -/

/--
Topical maps are `SeminormNonexpansive` for the variation seminorm.

This bridges the setup layer (Proposition `prop:topical-nonexpansive`) and the generic
blueprint layer: any monotone, translation-equivariant map `T` satisfies the
`SeminormNonexpansive` predicate for `variationSeminormAsSeminorm`.
-/
theorem topical_implies_variationSeminorm_nonexpansive
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (T : (ι → ℝ) → (ι → ℝ))
    (hmono : Monotone T)
    (htrans : TranslationEquivariant T) :
    SeminormNonexpansive variationSeminormAsSeminorm T := by
  intro x y
  -- `variationSeminormAsSeminorm` is `Seminorm.of variationSeminorm ...`, so its
  -- underlying function is definitionally `variationSeminorm`.
  change variationSeminorm (T x - T y) ≤ variationSeminorm (x - y)
  exact variationSeminorm_nonexpansive_of_topical T hmono htrans x y

/--
Iterate of a topical map is `SeminormNonexpansive` for the variation seminorm.

Combines `topical_implies_variationSeminorm_nonexpansive` with
`SeminormNonexpansive_iterate`: if `T` is monotone and translation-equivariant,
then every iterate `T^[k]` is also `SeminormNonexpansive variationSeminormAsSeminorm`.
-/
theorem topical_implies_SeminormNonexpansive_iterate
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (T : (ι → ℝ) → (ι → ℝ))
    (hmono : Monotone T)
    (htrans : TranslationEquivariant T)
    (k : ℕ) :
    SeminormNonexpansive variationSeminormAsSeminorm (T^[k]) :=
  SeminormNonexpansive_iterate variationSeminormAsSeminorm T
    (topical_implies_variationSeminorm_nonexpansive T hmono htrans) k

/--
Uniform orbit bound for a topical map.

If `T` is monotone and translation-equivariant (`T` is "topical"), `uStar` is a fixed
point of `T` with `variationSeminorm uStar ≤ B`, and `u0` satisfies
`variationSeminorm u0 = 0`, then every iterate satisfies
  `variationSeminorm (T^[k] u0) ≤ 2 * B`.

This is the orbit-bound consequence of Proposition `prop:topical-nonexpansive` that
applications can use directly once they have the topical hypotheses.
-/
theorem topical_implies_orbit_bound
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (T : (ι → ℝ) → (ι → ℝ))
    (hmono : Monotone T)
    (htrans : TranslationEquivariant T)
    {uStar u0 : ι → ℝ}
    (hfix : T uStar = uStar)
    {B : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((T^[k]) u0) ≤ 2 * B := by
  have hne : SeminormNonexpansive variationSeminormAsSeminorm T :=
    topical_implies_variationSeminorm_nonexpansive T hmono htrans
  have horbit :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
      variationSeminormAsSeminorm T hne (uStar := uStar) (u0 := u0) hfix hbound k
  -- `variationSeminormAsSeminorm x` is definitionally `variationSeminorm x`
  have horbit' : variationSeminorm ((T^[k]) u0) ≤ variationSeminorm u0 + 2 * B := horbit
  rw [hzero, zero_add] at horbit'
  linarith

/--
Uniform orbit bound for a topical map from an arbitrary starting point.

This is the base estimate before specializing to zero-variation seeds:
if `T` is topical and `uStar` is a fixed point with `variationSeminorm uStar ≤ B`,
then every iterate is bounded by `variationSeminorm u0 + 2 * B`.
-/
theorem topical_implies_orbit_bound_with_base
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (T : (ι → ℝ) → (ι → ℝ))
    (hmono : Monotone T)
    (htrans : TranslationEquivariant T)
    {uStar u0 : ι → ℝ}
    (hfix : T uStar = uStar)
    {B : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (k : ℕ) :
    variationSeminorm ((T^[k]) u0) ≤ variationSeminorm u0 + 2 * B := by
  have hne : SeminormNonexpansive variationSeminormAsSeminorm T :=
    topical_implies_variationSeminorm_nonexpansive T hmono htrans
  simpa using
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
      variationSeminormAsSeminorm T hne (uStar := uStar) (u0 := u0) hfix hbound k

/--
`IsTopical` bridge to `SeminormNonexpansive` for `variationSeminormAsSeminorm`.

Applications often expose topicality as a bundled `IsTopical` witness; this theorem
provides the direct nonexpansive packaging needed by generic orbit lemmas.
-/
theorem isTopical_implies_variationSeminorm_nonexpansive
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (T : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical T) :
    SeminormNonexpansive variationSeminormAsSeminorm T :=
  topical_implies_variationSeminorm_nonexpansive T hT.mono hT.trans

/--
Uniform orbit bound for an `IsTopical` map from an arbitrary starting point.

This is the bundled-`IsTopical` counterpart of
`topical_implies_orbit_bound_with_base`.
-/
theorem isTopical_orbit_bound_with_base
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (T : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical T)
    {uStar u0 : ι → ℝ}
    (hfix : T uStar = uStar)
    {B : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (k : ℕ) :
    variationSeminorm ((T^[k]) u0) ≤ variationSeminorm u0 + 2 * B :=
  topical_implies_orbit_bound_with_base T hT.mono hT.trans hfix hbound k

/--
Orbit bound for an `IsTopical` map from a zero-variation starting point.

This chains `IsTopical` → `SeminormNonexpansive` → uniform orbit bound in a single
convenient theorem for applications that have the `IsTopical` predicate directly.
-/
theorem isTopical_orbit_bound_from_zero
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (T : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical T)
    {uStar u0 : ι → ℝ}
    (hfix : T uStar = uStar)
    {B : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((T^[k]) u0) ≤ 2 * B := by
  have hbase : variationSeminorm ((T^[k]) u0) ≤ variationSeminorm u0 + 2 * B :=
    isTopical_orbit_bound_with_base T hT hfix hbound k
  rw [hzero, zero_add] at hbase
  exact hbase

/--
`IsTopical` gives both nonexpansiveness and the base orbit estimate in one package.
-/
theorem isTopical_nonexpansive_and_orbit_with_base
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (T : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical T)
    {uStar u0 : ι → ℝ}
    (hfix : T uStar = uStar)
    {B : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (k : ℕ) :
    SeminormNonexpansive variationSeminormAsSeminorm T ∧
      variationSeminorm ((T^[k]) u0) ≤ variationSeminorm u0 + 2 * B := by
  refine ⟨isTopical_implies_variationSeminorm_nonexpansive T hT, ?_⟩
  exact isTopical_orbit_bound_with_base T hT hfix hbound k

/--
`IsTopical` gives both nonexpansiveness and zero-seed orbit control in one package.
-/
theorem isTopical_nonexpansive_and_orbit_from_zero
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (T : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical T)
    {uStar u0 : ι → ℝ}
    (hfix : T uStar = uStar)
    {B : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    SeminormNonexpansive variationSeminormAsSeminorm T ∧
      variationSeminorm ((T^[k]) u0) ≤ 2 * B := by
  refine ⟨isTopical_implies_variationSeminorm_nonexpansive T hT, ?_⟩
  exact isTopical_orbit_bound_from_zero T hT hfix hbound hzero k

/-! ## App-facing topical blueprint packaging -/

/--
Section-3 dual rate + Section-4 topical orbit bound (base-form).

This is the minimal application-facing pair: combine the abstract `O(1/k)` dual rate
with an `IsTopical` iterate bound in one theorem.
-/
theorem genericBlueprint_dualRate_and_isTopical_orbit_with_base
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate kappa cost gamma hGamma : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (n k : ℕ) :
    gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      variationSeminorm ((Psi^[k]) u0) ≤
        variationSeminorm u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma := by
  constructor
  · exact genericBlueprint_dualRate halpha hgap_res hres_ascent hphi_bound hmono_gap n
  · exact isTopical_orbit_bound_with_base Psi hT hfix hbound k

/--
Section-3 dual rate + Section-4 topical orbit bound from a zero-variation seed.

This is the zero-base convenience form of
`genericBlueprint_dualRate_and_isTopical_orbit_with_base`.
-/
theorem genericBlueprint_dualRate_and_isTopical_orbit_from_zero
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate kappa cost gamma hGamma : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hzero : variationSeminorm u0 = 0)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (n k : ℕ) :
    gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      variationSeminorm ((Psi^[k]) u0) ≤ 2 * hGammaKappaBudget kappa cost gamma hGamma := by
  constructor
  · exact genericBlueprint_dualRate halpha hgap_res hres_ascent hphi_bound hmono_gap n
  · exact isTopical_orbit_bound_from_zero Psi hT hfix hbound hzero k

/--
Section-3 dual rate paired with iterate nonexpansiveness from `IsTopical`.

This theorem is useful when an application has a separate fixed-point argument and
only needs the dual rate together with nonexpansiveness of `Psi^[k]`.
-/
theorem genericBlueprint_dualRate_and_isTopical_iterate_nonexpansive
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (n k : ℕ) :
    gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k]) := by
  refine ⟨?_, ?_⟩
  · exact genericBlueprint_dualRate halpha hgap_res hres_ascent hphi_bound hmono_gap n
  · exact SeminormNonexpansive_iterate variationSeminormAsSeminorm Psi
      (isTopical_implies_variationSeminorm_nonexpansive Psi hT) k

/--
Section-3 dual rate bundled with both iterate nonexpansiveness and base-orbit control.

This is an application-facing "single-call" bridge when callers need all three facts:
`O(1/k)` dual rate, certified nonexpansiveness of `Psi^[k]`, and a base-seed iterate radius.
-/
theorem genericBlueprint_dualRate_and_isTopical_iterate_nonexpansive_and_orbit_with_base
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate kappa cost gamma hGamma : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (n k : ℕ) :
    gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k]) ∧
      variationSeminorm ((Psi^[k]) u0) ≤
        variationSeminorm u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma := by
  refine ⟨?_, ?_, ?_⟩
  · exact genericBlueprint_dualRate halpha hgap_res hres_ascent hphi_bound hmono_gap n
  · exact SeminormNonexpansive_iterate variationSeminormAsSeminorm Psi
      (isTopical_implies_variationSeminorm_nonexpansive Psi hT) k
  · exact isTopical_orbit_bound_with_base Psi hT hfix hbound k

/--
Section-3 dual rate bundled with iterate nonexpansiveness and zero-seed orbit control.

This is the zero-variation-seed specialization of
`genericBlueprint_dualRate_and_isTopical_iterate_nonexpansive_and_orbit_with_base`.
-/
theorem genericBlueprint_dualRate_and_isTopical_iterate_nonexpansive_and_orbit_from_zero
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate kappa cost gamma hGamma : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hzero : variationSeminorm u0 = 0)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (n k : ℕ) :
    gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k]) ∧
      variationSeminorm ((Psi^[k]) u0) ≤
        2 * hGammaKappaBudget kappa cost gamma hGamma := by
  refine ⟨?_, ?_, ?_⟩
  · exact genericBlueprint_dualRate halpha hgap_res hres_ascent hphi_bound hmono_gap n
  · exact SeminormNonexpansive_iterate variationSeminormAsSeminorm Psi
      (isTopical_implies_variationSeminorm_nonexpansive Psi hT) k
  · exact isTopical_orbit_bound_from_zero Psi hT hfix hbound hzero k

/--
Application-facing Section 3+4 blueprint package with topical dynamics.

Compared to `genericBlueprint_applicationBlueprint`, this endpoint accepts `IsTopical Psi`
directly and internally handles the nonexpansive bridge.
-/
theorem genericBlueprint_applicationBlueprint_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (n k : ℕ) :
    gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      variationSeminorm ((Psi^[k]) u0) ≤
        variationSeminorm u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget := by
  refine ⟨?_, ?_, ?_⟩
  · exact genericBlueprint_dualRate halpha hgap_res hres_ascent hphi_bound hmono_gap n
  · exact isTopical_orbit_bound_with_base Psi hT hfix hbound k
  · exact genericBlueprint_primalMassBound hmonoX hprimal_at_d hdual

/--
Application-facing Section 3+4 blueprint package from a zero-variation seed.

This is the zero-seed specialization of
`genericBlueprint_applicationBlueprint_of_isTopical`.
-/
theorem genericBlueprint_applicationBlueprint_of_isTopical_from_zero
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hzero : variationSeminorm u0 = 0)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (n k : ℕ) :
    gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      variationSeminorm ((Psi^[k]) u0) ≤ 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget := by
  refine ⟨?_, ?_, ?_⟩
  · exact genericBlueprint_dualRate halpha hgap_res hres_ascent hphi_bound hmono_gap n
  · exact isTopical_orbit_bound_from_zero Psi hT hfix hbound hzero k
  · exact genericBlueprint_primalMassBound hmonoX hprimal_at_d hdual

/--
Final epsilon-accuracy endpoint from topical dynamics and explicit blueprint assumptions.

This is the `IsTopical` analogue of `genericBlueprint_applicationEpsilonAccuracy`.
-/
theorem genericBlueprint_applicationEpsilonAccuracy_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps := by
  exact genericBlueprint_applicationEpsilonAccuracy
    (p := variationSeminormAsSeminorm)
    (Psi := Psi)
    (uStar := uStar) (u0 := u0)
    halpha hgap_res hres_ascent hphi_bound hmono_gap
    (isTopical_implies_variationSeminorm_nonexpansive Psi hT)
    hfix hbound hmonoX hprimal_at_d hdual hbias hrate n k hbudget

/--
Final epsilon-accuracy endpoint bundled with the topical base-orbit guarantee.

This is an application-style endpoint when callers want both:
1. the final `eps` objective guarantee, and
2. the certified iterate radius from the same topical/fixed-point assumptions.
-/
theorem genericBlueprint_applicationEpsilonAccuracy_and_orbit_with_base_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      variationSeminorm ((Psi^[k]) u0) ≤
        variationSeminorm u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma := by
  refine ⟨?_, ?_⟩
  · exact genericBlueprint_applicationEpsilonAccuracy_of_isTopical
      (Psi := Psi) (hT := hT) (uStar := uStar) (u0 := u0)
      hfix hbound halpha hgap_res hres_ascent hphi_bound hmono_gap
      hmonoX hprimal_at_d hdual hbias hrate n k hbudget
  · exact isTopical_orbit_bound_with_base Psi hT hfix hbound k

/--
Final epsilon-accuracy endpoint bundled with the topical zero-seed orbit guarantee.

This is the zero-seed companion of
`genericBlueprint_applicationEpsilonAccuracy_and_orbit_with_base_of_isTopical`.
-/
theorem genericBlueprint_applicationEpsilonAccuracy_and_orbit_from_zero_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hzero : variationSeminorm u0 = 0)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      variationSeminorm ((Psi^[k]) u0) ≤
        2 * hGammaKappaBudget kappa cost gamma hGamma := by
  refine ⟨?_, ?_⟩
  · exact genericBlueprint_applicationEpsilonAccuracy_of_isTopical
      (Psi := Psi) (hT := hT) (uStar := uStar) (u0 := u0)
      hfix hbound halpha hgap_res hres_ascent hphi_bound hmono_gap
      hmonoX hprimal_at_d hdual hbias hrate n k hbudget
  · exact isTopical_orbit_bound_from_zero Psi hT hfix hbound hzero k

/--
Final epsilon-accuracy endpoint bundled with iterate nonexpansiveness from topicality.

This endpoint is useful when applications need the final optimization guarantee together
with certified contraction control for the iterate map `Psi^[k]`.
-/
theorem genericBlueprint_applicationEpsilonAccuracy_and_iterate_nonexpansive_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k]) := by
  refine ⟨?_, ?_⟩
  · exact genericBlueprint_applicationEpsilonAccuracy_of_isTopical
      (Psi := Psi) (hT := hT) (uStar := uStar) (u0 := u0)
      hfix hbound halpha hgap_res hres_ascent hphi_bound hmono_gap
      hmonoX hprimal_at_d hdual hbias hrate n k hbudget
  · exact topical_implies_SeminormNonexpansive_iterate Psi hT.mono hT.trans k

/--
Final endpoint bundling epsilon-accuracy, dual-rate certification, and base-orbit control.

This bridge is useful for application reports that need a single theorem returning:
objective accuracy, explicit `O(1/k)` dual-rate value at index `n`, and iterate radius at `k`.
-/
theorem genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_orbit_with_base_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      variationSeminorm ((Psi^[k]) u0) ≤
        variationSeminorm u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma := by
  refine ⟨?_, ?_, ?_⟩
  · exact genericBlueprint_applicationEpsilonAccuracy_of_isTopical
      (Psi := Psi) (hT := hT) (uStar := uStar) (u0 := u0)
      hfix hbound halpha hgap_res hres_ascent hphi_bound hmono_gap
      hmonoX hprimal_at_d hdual hbias hrate n k hbudget
  · exact genericBlueprint_dualRate halpha hgap_res hres_ascent hphi_bound hmono_gap n
  · exact isTopical_orbit_bound_with_base Psi hT hfix hbound k

/--
Final endpoint bundling epsilon-accuracy, dual-rate certification, and zero-seed orbit control.

This is the zero-seed companion of
`genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_orbit_with_base_of_isTopical`.
-/
theorem genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_orbit_from_zero_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hzero : variationSeminorm u0 = 0)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      variationSeminorm ((Psi^[k]) u0) ≤
        2 * hGammaKappaBudget kappa cost gamma hGamma := by
  refine ⟨?_, ?_, ?_⟩
  · exact genericBlueprint_applicationEpsilonAccuracy_of_isTopical
      (Psi := Psi) (hT := hT) (uStar := uStar) (u0 := u0)
      hfix hbound halpha hgap_res hres_ascent hphi_bound hmono_gap
      hmonoX hprimal_at_d hdual hbias hrate n k hbudget
  · exact genericBlueprint_dualRate halpha hgap_res hres_ascent hphi_bound hmono_gap n
  · exact isTopical_orbit_bound_from_zero Psi hT hfix hbound hzero k

/--
Topical bridge bundling dual-rate, primal-mass, iterate nonexpansiveness, and base-orbit.

This is the pre-epsilon endpoint: it packages all Section 3+4 consequences plus iterate
nonexpansiveness for downstream uses that postpone the final bias/rate-to-epsilon step.
-/
theorem genericBlueprint_applicationDualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (n k : ℕ) :
    gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k]) ∧
      variationSeminorm ((Psi^[k]) u0) ≤
        variationSeminorm u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact genericBlueprint_dualRate halpha hgap_res hres_ascent hphi_bound hmono_gap n
  · exact topical_implies_SeminormNonexpansive_iterate Psi hT.mono hT.trans k
  · exact isTopical_orbit_bound_with_base Psi hT hfix hbound k
  · exact genericBlueprint_primalMassBound hmonoX hprimal_at_d hdual

/--
Topical bridge bundling epsilon-accuracy with dual-rate, primal-mass, iterate
nonexpansiveness, and base-orbit control.

This is the all-in-one application endpoint under topical assumptions.
-/
theorem genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k]) ∧
      variationSeminorm ((Psi^[k]) u0) ≤
        variationSeminorm u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget := by
  have hpack :
      |F0 - Fgamma n| ≤ eps ∧
        gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
        SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k]) ∧
        variationSeminorm ((Psi^[k]) u0) ≤
          variationSeminorm u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma := by
    exact genericBlueprint_applicationEpsilonDualRate_and_iterate_nonexpansive_and_orbit_with_base
      (p := variationSeminormAsSeminorm)
      (Psi := Psi) (uStar := uStar) (u0 := u0)
      halpha hgap_res hres_ascent hphi_bound hmono_gap
      (isTopical_implies_variationSeminorm_nonexpansive Psi hT)
      hfix hbound hmonoX hprimal_at_d hdual hbias hrate n k hbudget
  refine ⟨hpack.1, hpack.2.1, hpack.2.2.1, hpack.2.2.2, ?_⟩
  exact genericBlueprint_primalMassBound hmonoX hprimal_at_d hdual

/--
Zero-seed version of the full topical all-in-one application endpoint.

This specializes
`genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical`
under `variationSeminorm u0 = 0`.
-/
theorem genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_from_zero_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hzero : variationSeminorm u0 = 0)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k]) ∧
      variationSeminorm ((Psi^[k]) u0) ≤ 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget := by
  have hpack :=
    genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical
      (Psi := Psi) (hT := hT) (uStar := uStar) (u0 := u0)
      hfix hbound halpha hgap_res hres_ascent hphi_bound hmono_gap
      hmonoX hprimal_at_d hdual hbias hrate n k hbudget
  refine ⟨hpack.1, hpack.2.1, hpack.2.2.1, ?_, hpack.2.2.2.2⟩
  have horbitBase :
      variationSeminorm ((Psi^[k]) u0) ≤
        variationSeminorm u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma :=
    hpack.2.2.2.1
  rw [hzero, zero_add] at horbitBase
  exact horbitBase

/--
Successor-index topical version of the full all-in-one application endpoint.

This is the `k + 1` iterate wrapper around
`genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical`.
-/
theorem genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k + 1]) ∧
      variationSeminorm ((Psi^[k + 1]) u0) ≤
        variationSeminorm u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget := by
  simpa [Nat.succ_eq_add_one] using
    genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical
      (Psi := Psi) (hT := hT) (uStar := uStar) (u0 := u0)
      hfix hbound halpha hgap_res hres_ascent hphi_bound hmono_gap
      hmonoX hprimal_at_d hdual hbias hrate n (k + 1) hbudget

/--
Successor-index zero-seed topical version of the full all-in-one endpoint.

This is the `k + 1` iterate wrapper around
`genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_from_zero_of_isTopical`.
-/
theorem genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_from_zero_of_isTopical_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hzero : variationSeminorm u0 = 0)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k + 1]) ∧
      variationSeminorm ((Psi^[k + 1]) u0) ≤ 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget := by
  simpa [Nat.succ_eq_add_one] using
    genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_from_zero_of_isTopical
      (Psi := Psi) (hT := hT) (uStar := uStar) (u0 := u0)
      hfix hbound hzero halpha hgap_res hres_ascent hphi_bound hmono_gap
      hmonoX hprimal_at_d hdual hbias hrate n (k + 1) hbudget

/--
Successor-index topical pre-epsilon endpoint bundling dual-rate, primal-mass,
iterate nonexpansiveness, and base-orbit.
-/
theorem genericBlueprint_applicationDualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (n k : ℕ) :
    gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k + 1]) ∧
      variationSeminorm ((Psi^[k + 1]) u0) ≤
        variationSeminorm u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget := by
  simpa [Nat.succ_eq_add_one] using
    genericBlueprint_applicationDualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical
      (Psi := Psi) (hT := hT) (uStar := uStar) (u0 := u0)
      hfix hbound halpha hgap_res hres_ascent hphi_bound hmono_gap
      hmonoX hprimal_at_d hdual n (k + 1)

/--
Index-relaxed form of the topical all-in-one endpoint with base orbit control.

This wrapper is convenient when a workflow tracks a larger iterate bound `kMax`
but only needs the certified bundle at some `k ≤ kMax`.
-/
theorem genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n kMax k : ℕ)
    (_hk : k ≤ kMax)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k]) ∧
      variationSeminorm ((Psi^[k]) u0) ≤
        variationSeminorm u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget := by
  exact
    genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical
      (Psi := Psi) (hT := hT) (uStar := uStar) (u0 := u0)
      hfix hbound halpha hgap_res hres_ascent hphi_bound hmono_gap
      hmonoX hprimal_at_d hdual hbias hrate n k hbudget

/--
Index-relaxed zero-seed form of the topical all-in-one endpoint.

As above, this keeps an ambient bound `kMax` available while extracting the
certification at any `k ≤ kMax`.
-/
theorem genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_from_zero_of_isTopical_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hzero : variationSeminorm u0 = 0)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n kMax k : ℕ)
    (_hk : k ≤ kMax)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k]) ∧
      variationSeminorm ((Psi^[k]) u0) ≤ 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget := by
  exact
    genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_from_zero_of_isTopical
      (Psi := Psi) (hT := hT) (uStar := uStar) (u0 := u0)
      hfix hbound hzero halpha hgap_res hres_ascent hphi_bound hmono_gap
      hmonoX hprimal_at_d hdual hbias hrate n k hbudget

/--
Index-relaxed topical pre-epsilon bundle wrapper.

This is the `dualRate + primalMass + iterate + orbit` counterpart of the two
all-in-one `of_le_index` wrappers above.
-/
theorem genericBlueprint_applicationDualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (n kMax k : ℕ)
    (_hk : k ≤ kMax) :
    gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k]) ∧
      variationSeminorm ((Psi^[k]) u0) ≤
        variationSeminorm u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget := by
  exact
    genericBlueprint_applicationDualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical
      (Psi := Psi) (hT := hT) (uStar := uStar) (u0 := u0)
      hfix hbound halpha hgap_res hres_ascent hphi_bound hmono_gap
      hmonoX hprimal_at_d hdual n k

/--
Natural-bound companion of the topical all-in-one endpoint with base orbit control.
-/
theorem genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical_of_natBound
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k N : ℕ)
    (_hNk : N ≤ k)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k]) ∧
      variationSeminorm ((Psi^[k]) u0) ≤
        variationSeminorm u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget :=
  genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical_of_le_index
    (Psi := Psi) (hT := hT) (uStar := uStar) (u0 := u0)
    hfix hbound halpha hgap_res hres_ascent hphi_bound hmono_gap
    hmonoX hprimal_at_d hdual hbias hrate n k k (Nat.le_refl k) hbudget

/--
Natural-bound companion of the topical all-in-one endpoint with zero-seed orbit control.
-/
theorem genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_from_zero_of_isTopical_of_natBound
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hzero : variationSeminorm u0 = 0)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k N : ℕ)
    (_hNk : N ≤ k)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k]) ∧
      variationSeminorm ((Psi^[k]) u0) ≤ 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget :=
  genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_from_zero_of_isTopical_of_le_index
    (Psi := Psi) (hT := hT) (uStar := uStar) (u0 := u0)
    hfix hbound hzero halpha hgap_res hres_ascent hphi_bound hmono_gap
    hmonoX hprimal_at_d hdual hbias hrate n k k (Nat.le_refl k) hbudget

/--
Natural-bound companion of the topical pre-epsilon
`dualRate + primalMass + iterate + orbit` endpoint.
-/
theorem genericBlueprint_applicationDualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical_of_natBound
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (n k N : ℕ)
    (_hNk : N ≤ k) :
    gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k]) ∧
      variationSeminorm ((Psi^[k]) u0) ≤
        variationSeminorm u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget :=
  genericBlueprint_applicationDualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical_of_le_index
    (Psi := Psi) (hT := hT) (uStar := uStar) (u0 := u0)
    hfix hbound halpha hgap_res hres_ascent hphi_bound hmono_gap
    hmonoX hprimal_at_d hdual n k k (Nat.le_refl k)

/--
Successor-index + index-relaxed wrapper for the topical pre-epsilon endpoint.
-/
theorem
    genericBlueprint_applicationDualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical_succ_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (n kMax k : ℕ)
    (hk : k + 1 ≤ kMax) :
    gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k + 1]) ∧
      variationSeminorm ((Psi^[k + 1]) u0) ≤
        variationSeminorm u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget :=
  genericBlueprint_applicationDualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical_of_le_index
    (Psi := Psi) (hT := hT) (uStar := uStar) (u0 := u0)
    hfix hbound halpha hgap_res hres_ascent hphi_bound hmono_gap
    hmonoX hprimal_at_d hdual n kMax (k + 1) hk

/--
Successor-index natural-bound wrapper for the topical pre-epsilon endpoint.
-/
theorem
    genericBlueprint_applicationDualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical_succ_of_natBound
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (n k N : ℕ)
    (_hNk : N ≤ k + 1) :
    gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k + 1]) ∧
      variationSeminorm ((Psi^[k + 1]) u0) ≤
        variationSeminorm u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget :=
  genericBlueprint_applicationDualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical_succ_of_le_index
    (Psi := Psi) (hT := hT) (uStar := uStar) (u0 := u0)
    hfix hbound halpha hgap_res hres_ascent hphi_bound hmono_gap
    hmonoX hprimal_at_d hdual n (k + 1) k (Nat.le_refl (k + 1))

/--
Successor-index + index-relaxed wrapper of the topical all-in-one endpoint
with base-orbit control.
-/
theorem
    genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical_succ_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n kMax k : ℕ)
    (hk : k + 1 ≤ kMax)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k + 1]) ∧
      variationSeminorm ((Psi^[k + 1]) u0) ≤
        variationSeminorm u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget :=
  genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical_of_le_index
    (Psi := Psi) (hT := hT) (uStar := uStar) (u0 := u0)
    hfix hbound halpha hgap_res hres_ascent hphi_bound hmono_gap
    hmonoX hprimal_at_d hdual hbias hrate n kMax (k + 1) hk hbudget

/--
Successor-index + index-relaxed wrapper of the topical all-in-one endpoint
with zero-seed orbit control.
-/
theorem
    genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_from_zero_of_isTopical_succ_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hzero : variationSeminorm u0 = 0)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n kMax k : ℕ)
    (hk : k + 1 ≤ kMax)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k + 1]) ∧
      variationSeminorm ((Psi^[k + 1]) u0) ≤ 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget :=
  genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_from_zero_of_isTopical_of_le_index
    (Psi := Psi) (hT := hT) (uStar := uStar) (u0 := u0)
    hfix hbound hzero halpha hgap_res hres_ascent hphi_bound hmono_gap
    hmonoX hprimal_at_d hdual hbias hrate n kMax (k + 1) hk hbudget

/--
Successor-index natural-bound wrapper for the topical all-in-one endpoint
with base-orbit control.
-/
theorem
    genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical_succ_of_natBound
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k N : ℕ)
    (_hNk : N ≤ k + 1)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k + 1]) ∧
      variationSeminorm ((Psi^[k + 1]) u0) ≤
        variationSeminorm u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget :=
  genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical_succ_of_le_index
    (Psi := Psi) (hT := hT) (uStar := uStar) (u0 := u0)
    hfix hbound halpha hgap_res hres_ascent hphi_bound hmono_gap
    hmonoX hprimal_at_d hdual hbias hrate n (k + 1) k (Nat.le_refl (k + 1)) hbudget

/--
Successor-index natural-bound wrapper for the topical all-in-one endpoint
with zero-seed orbit control.
-/
theorem
    genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_from_zero_of_isTopical_succ_of_natBound
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hzero : variationSeminorm u0 = 0)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k N : ℕ)
    (_hNk : N ≤ k + 1)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps ∧
      gap n ≤ (alpha * Brate) / (n + 1 : ℝ) ∧
      SeminormNonexpansive variationSeminormAsSeminorm (Psi^[k + 1]) ∧
      variationSeminorm ((Psi^[k + 1]) u0) ≤ 2 * hGammaKappaBudget kappa cost gamma hGamma ∧
      X ≤ Xbound budget :=
  genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_from_zero_of_isTopical_succ_of_le_index
    (Psi := Psi) (hT := hT) (uStar := uStar) (u0 := u0)
    hfix hbound hzero halpha hgap_res hres_ascent hphi_bound hmono_gap
    hmonoX hprimal_at_d hdual hbias hrate n (k + 1) k (Nat.le_refl (k + 1)) hbudget

/--
Concrete primal endpoint projection from the topical all-in-one bridge.
-/
theorem
    genericBlueprint_primalEndpoint_of_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {phi gap residual : ℕ → ℝ}
    {alpha Brate d budget X kappa cost gamma hGamma bias C eps : ℝ}
    (Psi : (ι → ℝ) → (ι → ℝ))
    {Xbound : ℝ → ℝ}
    {F0 FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (hT : IsTopical Psi)
    {uStar u0 : ι → ℝ}
    (hfix : Psi uStar = uStar)
    (hbound : variationSeminorm uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ j : ℕ, gap j ≤ alpha * residual j)
    (hres_ascent : ∀ j : ℕ, residual j ≤ phi (j + 1) - phi j)
    (hphi_bound : ∀ m : ℕ, phi (m + 1) - phi 0 ≤ Brate)
    (hmono_gap : Antitone gap)
    (hmonoX : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ C / (m + 1 : ℝ))
    (n k : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    X ≤ Xbound budget :=
  (genericBlueprint_applicationEpsilonAccuracy_and_dualRate_and_primalMass_and_iterate_and_orbit_with_base_of_isTopical
    (Psi := Psi) (hT := hT) (uStar := uStar) (u0 := u0)
    hfix hbound halpha hgap_res hres_ascent hphi_bound hmono_gap
    hmonoX hprimal_at_d hdual hbias hrate n k hbudget).2.2.2.2

end PrimalDualBounds
end KLProjection
end FlowSinkhorn
