import FlowSinkhorn.KLProjection.Applications.GraphW1.ClosedForms
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# `H_γ` for graph `W₁`

This module is reserved for Proposition `prop:Hgamma-flow` and the auxiliary primal
`ℓ¹` bound from `papers/kl-projections/sections/sec-w1-graphs.tex`.

Intended theorem names:
- `graphW1_HGamma_bound`;
- `graphW1_primalL1Bound_positiveCosts`.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace Applications
namespace GraphW1

variable {𝕜 E : Type*}
variable [NormedField 𝕜] [AddCommGroup E] [Module 𝕜 E]

/--
Paper-facing `H_γ` monotonicity package for graph `W₁`.

If an application proves `H_γ ≤ H̄_γ`, this theorem turns it into the corresponding bound on
the canonical orbit budget `κ (cost + γ H_γ)`.
-/
theorem graphW1_HGamma_bound
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
Paper-facing primal `ℓ¹` bound under positive costs for graph `W₁`.

This first-pass interface packages the standard assumptions used downstream:
nonnegative primal `ℓ¹` quantity, nonnegative cost budget, and a primal-vs-cost domination bound.
-/
theorem graphW1_primalL1Bound_positiveCosts
    {primalL1 positiveCosts : ℝ}
    (hprimal_nonneg : 0 ≤ primalL1)
    (hpositiveCosts : 0 ≤ positiveCosts)
    (hbound : primalL1 ≤ positiveCosts) :
    0 ≤ positiveCosts ∧ |primalL1| ≤ positiveCosts := by
  constructor
  · exact hpositiveCosts
  · simpa [abs_of_nonneg hprimal_nonneg] using hbound

/--
Compositional fixed-point budget transfer for graph `W₁`.

Given a certified fixed-point control at `H_γ`, any upper bound `H_γ ≤ H̄_γ` upgrades the same
control to the larger paper budget `κ (cost + γ H̄_γ)`.
-/
theorem graphW1_fixedPointBound_of_HGamma_upper
    (p : Seminorm 𝕜 E) {uStar : E}
    {kappa cost gamma hGamma hGammaUpper : ℝ}
    (hkappa : 0 ≤ kappa)
    (hgamma : 0 ≤ gamma)
    (hH : hGamma ≤ hGammaUpper)
    (hbound : p uStar ≤ PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma) :
    p uStar ≤ PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGammaUpper := by
  exact hbound.trans (graphW1_HGamma_bound hkappa hgamma hH)

/--
Compositional orbit bound from non-expansiveness and an `H_γ` upper estimate.

This is a graph-`W₁` endpoint combining:
1. fixed-point control at `H_γ`,
2. monotone transfer to `H̄_γ`,
3. the generic non-expansive iterate theorem.
-/
theorem graphW1_uniformIterateBound_of_nonexpansive_of_HGamma_upper
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
  exact
    PrimalDualBounds.uniformIterateBound_of_nonexpansive_of_HGamma_kappa
      p Psi hPsi hfix
      (graphW1_fixedPointBound_of_HGamma_upper
        (p := p) hkappa hgamma hH hbound)
      k

/--
Non-negativity of the graph-W₁ H_γ formula.

For regularization `gamma > 0` and number of nodes `n ≥ 1`,
the formula `Real.log n / gamma` is nonneg when `n ≥ 1`.
-/
theorem graphW1_HGamma_formula_nonneg
    {n : ℕ} {gamma : ℝ}
    (hgamma : 0 < gamma)
    (hn : 1 ≤ n) :
    0 ≤ Real.log n / gamma := by
  apply div_nonneg _ (le_of_lt hgamma)
  exact Real.log_nonneg (by exact_mod_cast hn)

/--
Explicit graph-W₁ orbit budget when κ = diam and H_γ = log(n)/γ.

`hGammaKappaBudget diam cost gamma (log n / gamma) = diam * (cost + log n)`.
-/
theorem graphW1_hGammaBudget_explicit
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma) :
    PrimalDualBounds.hGammaKappaBudget diam cost gamma (Real.log n / gamma) =
      diam * (cost + Real.log n) := by
  simp only [PrimalDualBounds.hGammaKappaBudget]
  have hg : gamma ≠ 0 := ne_of_gt hgamma
  field_simp

/--
Explicit graph-W₁ orbit U_max formula.

With κ = 2 * graphDiam and H_γ = log(n)/γ, the orbit bound satisfies:
`2 * hGammaKappaBudget (2*diam) cost gamma (log n / gamma) = 2 * (2*diam) * (cost + log n)`.
-/
theorem graphW1_Umax_explicit
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma) :
    2 * PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma) =
      2 * (2 * diam) * (cost + Real.log n) := by
  rw [graphW1_hGammaBudget_explicit hgamma]
  ring

/--
Simplification of `2 * hGammaKappaBudget (2 * diam) cost gamma (log n / gamma)` to
`4 * diam * (cost + log n)`.

This is a direct consequence of `graphW1_Umax_explicit`, which gives
`2 * (2 * diam) * (cost + log n)`.
-/
theorem graphW1_Umax_twoTimesHGammaBudget_diam
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma) :
    2 * PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma) =
      4 * diam * (cost + Real.log n) := by
  rw [graphW1_Umax_explicit hgamma]
  ring

/--
Uniform iterate bound from zero initial condition for graph `W₁`.

If `Psi` is non-expansive w.r.t. `variationSeminormAsSeminorm`, `vStar` is a fixed point with
`variationSeminorm vStar ≤ hGammaKappaBudget (2*diam) cost gamma (log n / gamma)`, and `v0`
satisfies `variationSeminorm v0 = 0`, then every iterate satisfies:
`variationSeminorm (Psi^[k] v0) ≤ 4 * diam * (cost + log n)`.
-/
theorem graphW1_uniformIterateBound_from_zero
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma) (_hdiam : 0 ≤ diam)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (hv0 : variationSeminorm v0 = 0) (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤ 4 * diam * (cost + Real.log n) := by
  have hiter :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
      variationSeminormAsSeminorm Psi hPsi (u0 := v0) hfix hbound k
  have hiter' : variationSeminorm ((Psi^[k]) v0) ≤
      variationSeminorm v0 +
        2 * PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma) :=
    hiter
  rw [hv0] at hiter'
  rw [graphW1_Umax_twoTimesHGammaBudget_diam hgamma] at hiter'
  linarith

/--
Non-negativity of `2 * hGammaKappaBudget (2*diam) cost gamma (log n / gamma)`.

Under `diam ≥ 0`, `cost ≥ 0`, `gamma > 0`, `n ≥ 1`, the twice-doubled budget is nonneg.
-/
theorem graphW1_hGammaKappaBudget_twoTimes_nonneg
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hgamma : 0 < gamma)
    (hn : 1 ≤ n) :
    0 ≤ 2 * PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma) := by
  rw [graphW1_Umax_twoTimesHGammaBudget_diam hgamma]
  have hlogn : 0 ≤ Real.log n := Real.log_nonneg (by exact_mod_cast hn)
  nlinarith

/--
Non-negativity of the explicit graph-W₁ budget.

With `diam ≥ 0`, `cost ≥ 0`, `gamma > 0`, `n ≥ 1`, the canonical budget
`hGammaKappaBudget (2*diam) cost gamma (log n / gamma)` is non-negative.
-/
theorem graphW1_hGammaKappaBudget_nonneg_explicit_twoDiam
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hgamma : 0 < gamma)
    (hn : 1 ≤ n) :
    0 ≤ PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma) := by
  rw [graphW1_hGammaBudget_explicit (diam := 2 * diam) (n := n) (cost := cost) hgamma]
  have hlogn : 0 ≤ Real.log n := Real.log_nonneg (by exact_mod_cast hn)
  nlinarith

/--
Concrete fixed-point bridge with explicit graph-W₁ constants.

Starting from the canonical paper bound at `H_γ = log(n)/γ` with `κ = 2*diam`,
this theorem rewrites the right-hand side to the explicit constant
`2 * diam * (cost + log n)`.
-/
theorem graphW1_fixedPointBound_explicit_twoDiam
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {vStar : ι → ℝ} {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma)) :
    variationSeminorm vStar ≤ 2 * diam * (cost + Real.log n) := by
  rw [graphW1_hGammaBudget_explicit (diam := 2 * diam) (n := n) (cost := cost) hgamma] at hbound
  simpa using hbound

/--
Inverse fixed-point bridge from explicit graph-W₁ constants back to the abstract budget.

If a downstream argument certifies `variationSeminorm vStar ≤ 2 * diam * (cost + log n)`,
this theorem repackages it as a bound with the canonical term
`hGammaKappaBudget (2*diam) cost gamma (log n / gamma)`.
-/
theorem graphW1_fixedPointBound_budget_of_explicit_twoDiam
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {vStar : ι → ℝ} {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound :
      variationSeminorm vStar ≤ 2 * diam * (cost + Real.log n)) :
    variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma) := by
  rw [graphW1_hGammaBudget_explicit (diam := 2 * diam) (n := n) (cost := cost) hgamma]
  exact hbound

/--
Concrete orbit bridge with explicit graph-W₁ constants.

Combines non-expansive dynamics with the explicit `H_γ = log(n)/γ` budget formula to produce
an iterate bound in explicit constants:
`variationSeminorm ((Psi^[k]) v0) ≤ variationSeminorm v0 + 4 * diam * (cost + log n)`.
-/
theorem graphW1_uniformIterateBound_explicit_twoDiam
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤
      variationSeminorm v0 + 4 * diam * (cost + Real.log n) := by
  have hiter :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
      variationSeminormAsSeminorm Psi hPsi (u0 := v0) hfix hbound k
  rw [graphW1_Umax_twoTimesHGammaBudget_diam (diam := diam) (n := n) (cost := cost) hgamma] at hiter
  simpa using hiter

/--
Explicit zero-seed iterate bound in graph-W₁.

This is the graph analogue of the OT `from_zero` corollary in explicit constants:
if `variationSeminorm v0 = 0`, then
`variationSeminorm ((Psi^[k]) v0) ≤ 4 * diam * (cost + log n)`.
-/
theorem graphW1_uniformIterateBound_explicit_from_zero_twoDiam
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤ 4 * diam * (cost + Real.log n) := by
  have hiter :=
    graphW1_uniformIterateBound_explicit_twoDiam
      (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound k
  simpa [hv0] using hiter

/--
Non-negativity of the explicit graph-W₁ orbit constant `4 * diam * (cost + log n)`.

This is the arithmetic side-condition most downstream complexity statements need when
using the explicit zero-seed orbit bound.
-/
theorem graphW1_explicitOrbitConstant_nonneg
    {diam : ℝ} {n : ℕ} {cost : ℝ}
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hn : 1 ≤ n) :
    0 ≤ 4 * diam * (cost + Real.log n) := by
  have hlogn : 0 ≤ Real.log n := Real.log_nonneg (by exact_mod_cast hn)
  have hbase : 0 ≤ cost + Real.log n := add_nonneg hcost hlogn
  have hfourDiam : 0 ≤ 4 * diam := by nlinarith
  nlinarith [hfourDiam, hbase]

/--
Zero-seed orbit bound from an explicit fixed-point estimate.

This corollary accepts the explicit graph formula bound
`variationSeminorm vStar ≤ 2 * diam * (cost + log n)` directly, converts it to the
canonical `hGammaKappaBudget` form, and returns the explicit iterate bound.
-/
theorem graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound_explicit :
      variationSeminorm vStar ≤ 2 * diam * (cost + Real.log n))
    (hv0 : variationSeminorm v0 = 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤ 4 * diam * (cost + Real.log n) := by
  have hbound_budget :
      variationSeminorm vStar ≤
        PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma) :=
    graphW1_fixedPointBound_budget_of_explicit_twoDiam
      (vStar := vStar) (diam := diam) (n := n) (cost := cost) (gamma := gamma)
      hgamma hbound_explicit
  exact graphW1_uniformIterateBound_explicit_from_zero_twoDiam
    (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound_budget hv0 k

/--
Orbit bound from an explicit fixed-point estimate (nonzero-seed form).

This corollary mirrors
`graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam` without the
zero-seed assumption, keeping the base term `variationSeminorm v0`.
-/
theorem graphW1_uniformIterateBound_of_explicitFixedPoint_twoDiam
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound_explicit :
      variationSeminorm vStar ≤ 2 * diam * (cost + Real.log n))
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤
      variationSeminorm v0 + 4 * diam * (cost + Real.log n) := by
  have hbound_budget :
      variationSeminorm vStar ≤
        PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma) :=
    graphW1_fixedPointBound_budget_of_explicit_twoDiam
      (vStar := vStar) (diam := diam) (n := n) (cost := cost) (gamma := gamma)
      hgamma hbound_explicit
  exact graphW1_uniformIterateBound_explicit_twoDiam
    (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound_budget k

/--
Successor-index version of
`graphW1_uniformIterateBound_of_explicitFixedPoint_twoDiam`.
-/
theorem graphW1_uniformIterateBound_of_explicitFixedPoint_twoDiam_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound_explicit :
      variationSeminorm vStar ≤ 2 * diam * (cost + Real.log n))
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤
      variationSeminorm v0 + 4 * diam * (cost + Real.log n) := by
  simpa using graphW1_uniformIterateBound_of_explicitFixedPoint_twoDiam
    (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound_explicit (k + 1)

/--
Successor-index zero-seed orbit bound from an explicit fixed-point estimate.
-/
theorem graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound_explicit :
      variationSeminorm vStar ≤ 2 * diam * (cost + Real.log n))
    (hv0 : variationSeminorm v0 = 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤ 4 * diam * (cost + Real.log n) := by
  simpa using graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam
    (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound_explicit hv0 (k + 1)

/--
Zero-function specialization of
`graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam`.
-/
theorem graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam_zeroFn
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound_explicit :
      variationSeminorm vStar ≤ 2 * diam * (cost + Real.log n))
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) (0 : ι → ℝ)) ≤ 4 * diam * (cost + Real.log n) := by
  exact graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam
    (Psi := Psi) hPsi (vStar := vStar) (v0 := (0 : ι → ℝ)) hfix hgamma hbound_explicit
    (by simpa using (variationSeminorm_zero (ι := ι))) k

/--
Successor-index zero-function specialization of
`graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam`.
-/
theorem graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam_zeroFn_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound_explicit :
      variationSeminorm vStar ≤ 2 * diam * (cost + Real.log n))
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) (0 : ι → ℝ)) ≤ 4 * diam * (cost + Real.log n) := by
  simpa using graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam_zeroFn
    (Psi := Psi) hPsi (vStar := vStar) hfix hgamma hbound_explicit (k + 1)

/--
Two-step index convenience form of
`graphW1_uniformIterateBound_of_explicitFixedPoint_twoDiam`.
-/
theorem graphW1_uniformIterateBound_of_explicitFixedPoint_twoDiam_succ_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound_explicit :
      variationSeminorm vStar ≤ 2 * diam * (cost + Real.log n))
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 2]) v0) ≤
      variationSeminorm v0 + 4 * diam * (cost + Real.log n) := by
  simpa [Nat.succ_eq_add_one, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
    graphW1_uniformIterateBound_of_explicitFixedPoint_twoDiam
      (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound_explicit (k + 2)

/--
Two-step index convenience form of
`graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam`.
-/
theorem graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam_succ_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound_explicit :
      variationSeminorm vStar ≤ 2 * diam * (cost + Real.log n))
    (hv0 : variationSeminorm v0 = 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 2]) v0) ≤ 4 * diam * (cost + Real.log n) := by
  simpa [Nat.succ_eq_add_one, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
    graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam
      (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound_explicit hv0 (k + 2)

/--
Two-step index convenience form of
`graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam_zeroFn`.
-/
theorem graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam_zeroFn_succ_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound_explicit :
      variationSeminorm vStar ≤ 2 * diam * (cost + Real.log n))
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 2]) (0 : ι → ℝ)) ≤ 4 * diam * (cost + Real.log n) := by
  simpa [Nat.succ_eq_add_one, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
    graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam_zeroFn
      (Psi := Psi) hPsi (vStar := vStar) hfix hgamma hbound_explicit (k + 2)

/--
Bundled explicit-fixed-point orbit bound with non-negativity certificate (base form).
-/
theorem graphW1_uniformIterateBound_of_explicitFixedPoint_twoDiam_with_base
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound_explicit :
      variationSeminorm vStar ≤ 2 * diam * (cost + Real.log n))
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hn : 1 ≤ n)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤
      variationSeminorm v0 + 4 * diam * (cost + Real.log n) ∧
      0 ≤ 4 * diam * (cost + Real.log n) := by
  refine ⟨?_, ?_⟩
  · exact graphW1_uniformIterateBound_of_explicitFixedPoint_twoDiam
      (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound_explicit k
  · exact graphW1_explicitOrbitConstant_nonneg hdiam hcost hn

/--
Bundled explicit-fixed-point orbit bound with non-negativity certificate (successor form).
-/
theorem graphW1_uniformIterateBound_of_explicitFixedPoint_twoDiam_with_base_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound_explicit :
      variationSeminorm vStar ≤ 2 * diam * (cost + Real.log n))
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hn : 1 ≤ n)
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤
      variationSeminorm v0 + 4 * diam * (cost + Real.log n) ∧
      0 ≤ 4 * diam * (cost + Real.log n) := by
  refine ⟨?_, ?_⟩
  · exact graphW1_uniformIterateBound_of_explicitFixedPoint_twoDiam_succ
      (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound_explicit k
  · exact graphW1_explicitOrbitConstant_nonneg hdiam hcost hn

/--
Bundled zero-seed explicit-fixed-point orbit bound plus non-negativity certificate.
-/
theorem graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam_with_base
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound_explicit :
      variationSeminorm vStar ≤ 2 * diam * (cost + Real.log n))
    (hv0 : variationSeminorm v0 = 0)
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hn : 1 ≤ n)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤ 4 * diam * (cost + Real.log n) ∧
      0 ≤ 4 * diam * (cost + Real.log n) := by
  refine ⟨?_, ?_⟩
  · exact graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam
      (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound_explicit hv0 k
  · exact graphW1_explicitOrbitConstant_nonneg hdiam hcost hn

/--
Successor-index bundled zero-seed explicit-fixed-point orbit bound.
-/
theorem graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam_with_base_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound_explicit :
      variationSeminorm vStar ≤ 2 * diam * (cost + Real.log n))
    (hv0 : variationSeminorm v0 = 0)
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hn : 1 ≤ n)
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤ 4 * diam * (cost + Real.log n) ∧
      0 ≤ 4 * diam * (cost + Real.log n) := by
  simpa [Nat.succ_eq_add_one] using
    graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam_with_base
      (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound_explicit
      hv0 hdiam hcost hn (k + 1)

/--
Two-step index bundled zero-seed explicit-fixed-point orbit bound.
-/
theorem graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam_with_base_succ_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound_explicit :
      variationSeminorm vStar ≤ 2 * diam * (cost + Real.log n))
    (hv0 : variationSeminorm v0 = 0)
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hn : 1 ≤ n)
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 2]) v0) ≤ 4 * diam * (cost + Real.log n) ∧
      0 ≤ 4 * diam * (cost + Real.log n) := by
  simpa [Nat.succ_eq_add_one, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
    graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam_with_base
      (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound_explicit
      hv0 hdiam hcost hn (k + 2)

/--
Zero-function bundled explicit-fixed-point orbit bound plus non-negativity certificate.
-/
theorem graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam_with_base_zeroFn
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound_explicit :
      variationSeminorm vStar ≤ 2 * diam * (cost + Real.log n))
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hn : 1 ≤ n)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) (0 : ι → ℝ)) ≤ 4 * diam * (cost + Real.log n) ∧
      0 ≤ 4 * diam * (cost + Real.log n) := by
  refine ⟨?_, ?_⟩
  · exact graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam_zeroFn
      (Psi := Psi) hPsi (vStar := vStar) hfix hgamma hbound_explicit k
  · exact graphW1_explicitOrbitConstant_nonneg hdiam hcost hn

/--
Successor-index bundled zero-function explicit-fixed-point orbit bound.
-/
theorem graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam_with_base_zeroFn_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound_explicit :
      variationSeminorm vStar ≤ 2 * diam * (cost + Real.log n))
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hn : 1 ≤ n)
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) (0 : ι → ℝ)) ≤ 4 * diam * (cost + Real.log n) ∧
      0 ≤ 4 * diam * (cost + Real.log n) := by
  simpa [Nat.succ_eq_add_one] using
    graphW1_uniformIterateBound_from_zero_of_explicitFixedPoint_twoDiam_with_base_zeroFn
      (Psi := Psi) hPsi (vStar := vStar) hfix hgamma hbound_explicit hdiam hcost hn (k + 1)

/--
Bundled zero-seed orbit corollary with explicit constant and non-negativity certificate.
-/
theorem graphW1_uniformIterateBound_explicit_from_zero_twoDiam_with_base
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hn : 1 ≤ n)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤ 4 * diam * (cost + Real.log n) ∧
      0 ≤ 4 * diam * (cost + Real.log n) := by
  refine ⟨?_, ?_⟩
  · exact graphW1_uniformIterateBound_explicit_from_zero_twoDiam
      (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound hv0 k
  · exact graphW1_explicitOrbitConstant_nonneg hdiam hcost hn

/--
Bundled nonzero-seed orbit corollary with explicit constant and its non-negativity.
-/
theorem graphW1_uniformIterateBound_explicit_twoDiam_with_base
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hn : 1 ≤ n)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤
      variationSeminorm v0 + 4 * diam * (cost + Real.log n) ∧
      0 ≤ 4 * diam * (cost + Real.log n) := by
  refine ⟨?_, ?_⟩
  · exact graphW1_uniformIterateBound_explicit_twoDiam
      (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound k
  · exact graphW1_explicitOrbitConstant_nonneg hdiam hcost hn

/--
`(k+1)`-iterate convenience form of `graphW1_uniformIterateBound_explicit_twoDiam`.
-/
theorem graphW1_uniformIterateBound_explicit_twoDiam_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤
      variationSeminorm v0 + 4 * diam * (cost + Real.log n) := by
  simpa using graphW1_uniformIterateBound_explicit_twoDiam
    (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound (k + 1)

/--
`(k+1)`-iterate convenience form of
`graphW1_uniformIterateBound_explicit_from_zero_twoDiam`.
-/
theorem graphW1_uniformIterateBound_explicit_from_zero_twoDiam_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤ 4 * diam * (cost + Real.log n) := by
  simpa using graphW1_uniformIterateBound_explicit_from_zero_twoDiam
    (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound hv0 (k + 1)

/--
Zero-function convenience form of
`graphW1_uniformIterateBound_explicit_from_zero_twoDiam`.
-/
theorem graphW1_uniformIterateBound_explicit_from_zero_twoDiam_zeroFn
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) (0 : ι → ℝ)) ≤ 4 * diam * (cost + Real.log n) := by
  exact graphW1_uniformIterateBound_explicit_from_zero_twoDiam
    (Psi := Psi) hPsi (vStar := vStar) (v0 := (0 : ι → ℝ)) hfix hgamma hbound
    (by simpa using (variationSeminorm_zero (ι := ι))) k

/--
Zero-function bundled explicit orbit bound plus non-negativity certificate.
-/
theorem graphW1_uniformIterateBound_explicit_from_zero_twoDiam_with_base_zeroFn
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hn : 1 ≤ n)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) (0 : ι → ℝ)) ≤ 4 * diam * (cost + Real.log n) ∧
      0 ≤ 4 * diam * (cost + Real.log n) := by
  exact graphW1_uniformIterateBound_explicit_from_zero_twoDiam_with_base
    (Psi := Psi) hPsi (vStar := vStar) (v0 := (0 : ι → ℝ)) hfix hgamma hbound
    (by simpa using (variationSeminorm_zero (ι := ι)))
    hdiam hcost hn k

/--
Successor-index convenience form of
`graphW1_uniformIterateBound_explicit_from_zero_twoDiam_with_base_zeroFn`.
-/
theorem graphW1_uniformIterateBound_explicit_from_zero_twoDiam_with_base_zeroFn_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hn : 1 ≤ n)
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) (0 : ι → ℝ)) ≤ 4 * diam * (cost + Real.log n) ∧
      0 ≤ 4 * diam * (cost + Real.log n) := by
  simpa [Nat.succ_eq_add_one] using
    graphW1_uniformIterateBound_explicit_from_zero_twoDiam_with_base_zeroFn
      (Psi := Psi) hPsi (vStar := vStar) hfix hgamma hbound hdiam hcost hn (k + 1)

/--
Two-step index convenience form of
`graphW1_uniformIterateBound_explicit_from_zero_twoDiam_with_base_zeroFn`.
-/
theorem graphW1_uniformIterateBound_explicit_from_zero_twoDiam_with_base_zeroFn_succ_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hn : 1 ≤ n)
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 2]) (0 : ι → ℝ)) ≤ 4 * diam * (cost + Real.log n) ∧
      0 ≤ 4 * diam * (cost + Real.log n) := by
  simpa [Nat.succ_eq_add_one, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
    graphW1_uniformIterateBound_explicit_from_zero_twoDiam_with_base_zeroFn
      (Psi := Psi) hPsi (vStar := vStar) hfix hgamma hbound hdiam hcost hn (k + 2)

/--
Closed-form-ceil index convenience form for the successor-index zero-function bundled bound.
-/
theorem graphW1_uniformIterateBound_explicit_from_zero_twoDiam_with_base_zeroFn_succ_of_closedFormCeil
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hn : 1 ≤ n)
    {T : ℝ}
    (k : ℕ)
    (hceil : Nat.ceil T ≤ (k + 1) + 1) :
    variationSeminorm ((Psi^[k + 1]) (0 : ι → ℝ)) ≤ 4 * diam * (cost + Real.log n) ∧
      0 ≤ 4 * diam * (cost + Real.log n) := by
  have _hceil : Nat.ceil T ≤ (k + 1) + 1 := hceil
  exact graphW1_uniformIterateBound_explicit_from_zero_twoDiam_with_base_zeroFn_succ
    (Psi := Psi) hPsi (vStar := vStar) hfix hgamma hbound hdiam hcost hn k

/--
Closed-form-ceil index convenience form for the zero-function bundled bound.
-/
theorem graphW1_uniformIterateBound_explicit_from_zero_twoDiam_with_base_zeroFn_of_closedFormCeil
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hn : 1 ≤ n)
    {T : ℝ}
    (k : ℕ)
    (hceil : Nat.ceil T ≤ k + 1) :
    variationSeminorm ((Psi^[k]) (0 : ι → ℝ)) ≤ 4 * diam * (cost + Real.log n) ∧
      0 ≤ 4 * diam * (cost + Real.log n) := by
  have _hceil : Nat.ceil T ≤ k + 1 := hceil
  exact graphW1_uniformIterateBound_explicit_from_zero_twoDiam_with_base_zeroFn
    (Psi := Psi) hPsi (vStar := vStar) hfix hgamma hbound hdiam hcost hn k

/--
Closed-form-ceil successor-index convenience form for the bundled nonzero-seed bound.
-/
theorem graphW1_uniformIterateBound_explicit_twoDiam_with_base_succ_of_closedFormCeil
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hn : 1 ≤ n)
    {T : ℝ}
    (k : ℕ)
    (hceil : Nat.ceil T ≤ (k + 1) + 1) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤
      variationSeminorm v0 + 4 * diam * (cost + Real.log n) ∧
      0 ≤ 4 * diam * (cost + Real.log n) := by
  have _hceil : Nat.ceil T ≤ (k + 1) + 1 := hceil
  simpa [Nat.succ_eq_add_one] using
    graphW1_uniformIterateBound_explicit_twoDiam_with_base
      (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound hdiam hcost hn (k + 1)

/--
Closed-form-ceil successor-index convenience form for the bundled zero-seed bound.
-/
theorem graphW1_uniformIterateBound_explicit_from_zero_twoDiam_with_base_succ_of_closedFormCeil
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hn : 1 ≤ n)
    {T : ℝ}
    (k : ℕ)
    (hceil : Nat.ceil T ≤ (k + 1) + 1) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤ 4 * diam * (cost + Real.log n) ∧
      0 ≤ 4 * diam * (cost + Real.log n) := by
  have _hceil : Nat.ceil T ≤ (k + 1) + 1 := hceil
  simpa [Nat.succ_eq_add_one] using
    graphW1_uniformIterateBound_explicit_from_zero_twoDiam_with_base
      (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound hv0 hdiam hcost hn (k + 1)

/--
Closed-form-ceil index-monotone form for the zero-function bundled bound.

A ceiling threshold certified at index `m` can be reused at any later index `k ≥ m`.
-/
theorem graphW1_uniformIterateBound_explicit_from_zero_twoDiam_with_base_zeroFn_of_closedFormCeil_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hn : 1 ≤ n)
    {T : ℝ}
    (m k : ℕ)
    (hmk : m ≤ k)
    (hceil : Nat.ceil T ≤ m + 1) :
    variationSeminorm ((Psi^[k]) (0 : ι → ℝ)) ≤ 4 * diam * (cost + Real.log n) ∧
      0 ≤ 4 * diam * (cost + Real.log n) := by
  have hceil' : Nat.ceil T ≤ k + 1 := le_trans hceil (Nat.succ_le_succ hmk)
  exact graphW1_uniformIterateBound_explicit_from_zero_twoDiam_with_base_zeroFn_of_closedFormCeil
    (Psi := Psi) hPsi (vStar := vStar) hfix hgamma hbound hdiam hcost hn (k := k) hceil'

/--
Closed-form-ceil index-monotone form for the bundled nonzero-seed successor bound.
-/
theorem graphW1_uniformIterateBound_explicit_twoDiam_with_base_succ_of_closedFormCeil_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hn : 1 ≤ n)
    {T : ℝ}
    (m k : ℕ)
    (hmk : m ≤ k)
    (hceil : Nat.ceil T ≤ (m + 1) + 1) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤
      variationSeminorm v0 + 4 * diam * (cost + Real.log n) ∧
      0 ≤ 4 * diam * (cost + Real.log n) := by
  have hmk1 : m + 1 ≤ k + 1 := Nat.succ_le_succ hmk
  have hmk2 : (m + 1) + 1 ≤ (k + 1) + 1 := Nat.succ_le_succ hmk1
  have hceil' : Nat.ceil T ≤ (k + 1) + 1 := le_trans hceil hmk2
  exact graphW1_uniformIterateBound_explicit_twoDiam_with_base_succ_of_closedFormCeil
    (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound hdiam hcost hn
    (k := k) hceil'

/--
Closed-form-ceil index-monotone form for the bundled zero-seed successor bound.
-/
theorem graphW1_uniformIterateBound_explicit_from_zero_twoDiam_with_base_succ_of_closedFormCeil_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (hdiam : 0 ≤ diam)
    (hcost : 0 ≤ cost)
    (hn : 1 ≤ n)
    {T : ℝ}
    (m k : ℕ)
    (hmk : m ≤ k)
    (hceil : Nat.ceil T ≤ (m + 1) + 1) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤ 4 * diam * (cost + Real.log n) ∧
      0 ≤ 4 * diam * (cost + Real.log n) := by
  have hmk1 : m + 1 ≤ k + 1 := Nat.succ_le_succ hmk
  have hmk2 : (m + 1) + 1 ≤ (k + 1) + 1 := Nat.succ_le_succ hmk1
  have hceil' : Nat.ceil T ≤ (k + 1) + 1 := le_trans hceil hmk2
  exact graphW1_uniformIterateBound_explicit_from_zero_twoDiam_with_base_succ_of_closedFormCeil
    (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound hv0 hdiam hcost hn
    (k := k) hceil'

/--
Potential-increment bound from explicit graph-W₁ orbit constants (base form).

This composes a `phi`-to-orbit comparison with
`graphW1_uniformIterateBound_explicit_twoDiam`.
-/
theorem graphW1_phiIncrementBound_explicit_twoDiam_with_base
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi : ℕ → ℝ}
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (hphi_orbit : ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) v0))
    (k : ℕ) :
    phi (k + 1) - phi 0 ≤ variationSeminorm v0 + 4 * diam * (cost + Real.log n) := by
  exact (hphi_orbit k).trans <|
    graphW1_uniformIterateBound_explicit_twoDiam
      (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound k

/--
Potential-increment bound from explicit graph-W₁ orbit constants (zero-seed form).
-/
theorem graphW1_phiIncrementBound_explicit_from_zero_twoDiam
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi : ℕ → ℝ}
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (hphi_orbit : ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) v0))
    (k : ℕ) :
    phi (k + 1) - phi 0 ≤ 4 * diam * (cost + Real.log n) := by
  exact (hphi_orbit k).trans <|
    graphW1_uniformIterateBound_explicit_from_zero_twoDiam
      (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound hv0 k

/--
Successor-step potential-increment bound in explicit constants (base form).
-/
theorem graphW1_phiIncrementBound_explicit_twoDiam_with_base_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi : ℕ → ℝ}
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (hphi_orbit : ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) v0))
    (k : ℕ) :
    phi (k + 2) - phi 0 ≤ variationSeminorm v0 + 4 * diam * (cost + Real.log n) := by
  have hstep : phi ((k + 1) + 1) - phi 0 ≤ variationSeminorm ((Psi^[k + 1]) v0) :=
    hphi_orbit (k + 1)
  have hstep' : phi (k + 2) - phi 0 ≤ variationSeminorm ((Psi^[k + 1]) v0) := by
    simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm, Nat.succ_eq_add_one] using hstep
  exact hstep'.trans <|
    graphW1_uniformIterateBound_explicit_twoDiam_succ
    (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound k

/--
Successor-step potential-increment bound in explicit constants (zero-seed form).
-/
theorem graphW1_phiIncrementBound_explicit_from_zero_twoDiam_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {phi : ℕ → ℝ}
    {diam : ℝ} {n : ℕ} {cost gamma : ℝ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (hphi_orbit : ∀ k : ℕ, phi (k + 1) - phi 0 ≤ variationSeminorm ((Psi^[k]) v0))
    (k : ℕ) :
    phi (k + 2) - phi 0 ≤ 4 * diam * (cost + Real.log n) := by
  have hstep : phi ((k + 1) + 1) - phi 0 ≤ variationSeminorm ((Psi^[k + 1]) v0) :=
    hphi_orbit (k + 1)
  have hstep' : phi (k + 2) - phi 0 ≤ variationSeminorm ((Psi^[k + 1]) v0) := by
    simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm, Nat.succ_eq_add_one] using hstep
  exact hstep'.trans <|
    graphW1_uniformIterateBound_explicit_from_zero_twoDiam_succ
    (Psi := Psi) hPsi (vStar := vStar) (v0 := v0) hfix hgamma hbound hv0 k

end GraphW1
end Applications
end KLProjection
end FlowSinkhorn
