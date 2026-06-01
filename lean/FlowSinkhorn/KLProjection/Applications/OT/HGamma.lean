import FlowSinkhorn.KLProjection.PrimalDualBounds.Blueprint
import FlowSinkhorn.KLProjection.Applications.OT.HGammaVocabulary
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# `H_γ` for balanced optimal transport

This module is reserved for Proposition `prop:Hgamma-ot` from
the OT material in `neurips/paper.tex`.

Intended theorem names:
- `ot_HGamma_bound`;
- `ot_HGamma_formula_uniform_logRatio_bound`;
- `ot_dualPotential_bound`.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace Applications
namespace OT

open scoped BigOperators

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
Proof-core statement for Proposition `app-prop:hgamma-ot`.

For a balanced OT minimizer, the paper proves the two scalar estimates
`log(Pγᵢⱼ) - log(zᵢⱼ) ≤ |log(min_b)|` and
`-(2*C_max/gamma) ≤ log(Pγᵢⱼ) - log(zᵢⱼ)`.  Once those two estimates have been derived
from the Sinkhorn marginal equations, this theorem packages the exact uniform `H_γ` formula.
-/
theorem ot_HGamma_formula_uniform_logRatio_bound
    {ι₁ ι₂ : Type*}
    (logRatio : ι₁ → ι₂ → ℝ)
    {min_b C_max gamma : ℝ}
    (hgamma : 0 < gamma)
    (_hmin_b : 0 < min_b)
    (hC_max : 0 ≤ C_max)
    (hupper : ∀ i j, logRatio i j ≤ |Real.log min_b|)
    (hlower : ∀ i j, -(2 * C_max / gamma) ≤ logRatio i j) :
    (∀ i j, |logRatio i j| ≤ |Real.log min_b| + 2 * C_max / gamma) ∧
      0 ≤ |Real.log min_b| + 2 * C_max / gamma := by
  have hB : 0 ≤ 2 * C_max / gamma := by positivity
  have hH : 0 ≤ |Real.log min_b| + 2 * C_max / gamma :=
    add_nonneg (abs_nonneg _) hB
  refine ⟨?_, hH⟩
  intro i j
  apply abs_le.mpr
  constructor
  · calc
      -(|Real.log min_b| + 2 * C_max / gamma)
          ≤ -(2 * C_max / gamma) := by
            linarith [abs_nonneg (Real.log min_b)]
      _ ≤ logRatio i j := hlower i j
  · exact (hupper i j).trans (le_add_of_nonneg_right hB)

/--
Min-mass version of Proposition `app-prop:hgamma-ot`.

This strengthens `ot_HGamma_formula_uniform_logRatio_bound`: the paper's upper log-ratio estimate
is naturally obtained as `logRatio <= -log(min_b)`.  Under the marginal-mass fact
`min_b <= 1`, Lean proves `|log(min_b)| = -log(min_b)` and converts that raw estimate into the
absolute-value form used by the displayed `H_γ` formula.
-/
theorem ot_HGamma_formula_uniform_logRatio_bound_from_minMass_le_one
    {ι₁ ι₂ : Type*}
    (logRatio : ι₁ → ι₂ → ℝ)
    {min_b C_max gamma : ℝ}
    (hgamma : 0 < gamma)
    (hmin_b : 0 < min_b)
    (hmin_le_one : min_b ≤ 1)
    (hC_max : 0 ≤ C_max)
    (hupper : ∀ i j, logRatio i j ≤ -Real.log min_b)
    (hlower : ∀ i j, -(2 * C_max / gamma) ≤ logRatio i j) :
    (∀ i j, |logRatio i j| ≤ |Real.log min_b| + 2 * C_max / gamma) ∧
      0 ≤ |Real.log min_b| + 2 * C_max / gamma := by
  have hlog_nonpos : Real.log min_b ≤ 0 :=
    Real.log_nonpos (le_of_lt hmin_b) hmin_le_one
  have hupper_abs : ∀ i j, logRatio i j ≤ |Real.log min_b| := by
    intro i j
    simpa [abs_of_nonpos hlog_nonpos] using hupper i j
  exact ot_HGamma_formula_uniform_logRatio_bound
    logRatio hgamma hmin_b hC_max hupper_abs hlower

/--
Probability-marginal version of Proposition `app-prop:hgamma-ot`.

This strengthens `ot_HGamma_formula_uniform_logRatio_bound_from_minMass_le_one`: the hypothesis
`min_b <= 1` is no longer supplied as a scalar fact.  Lean derives it from a nonempty finite
probability marginal `b`, coordinatewise nonnegativity, total mass `sum_i b_i = 1`, and the
certificate that `min_b` is a lower bound for all coordinates of that marginal.  This matches the
classical OT specialization, where `min_b` is the minimum of the positive marginals.
-/
theorem ot_HGamma_formula_uniform_logRatio_bound_from_probabilityMarginal
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (b : ι₁ → ℝ)
    (logRatio : ι₁ → ι₂ → ℝ)
    {min_b C_max gamma : ℝ}
    (hgamma : 0 < gamma)
    (hmin_b : 0 < min_b)
    (hb_nonneg : ∀ i, 0 ≤ b i)
    (hb_mass : (∑ i, b i) = 1)
    (hmin_le_marginal : ∀ i, min_b ≤ b i)
    (hC_max : 0 ≤ C_max)
    (hupper : ∀ i j, logRatio i j ≤ -Real.log min_b)
    (hlower : ∀ i j, -(2 * C_max / gamma) ≤ logRatio i j) :
    (∀ i j, |logRatio i j| ≤ |Real.log min_b| + 2 * C_max / gamma) ∧
      0 ≤ |Real.log min_b| + 2 * C_max / gamma := by
  classical
  obtain ⟨i₀⟩ := ‹Nonempty ι₁›
  have hb_i_le_sum : b i₀ ≤ ∑ i, b i := by
    simpa using
      (Finset.single_le_sum
        (s := Finset.univ) (f := b)
        (by intro i _hi; exact hb_nonneg i)
        (by simp : i₀ ∈ (Finset.univ : Finset ι₁)))
  have hmin_le_one : min_b ≤ 1 := by
    calc
      min_b ≤ b i₀ := hmin_le_marginal i₀
      _ ≤ ∑ i, b i := hb_i_le_sum
      _ = 1 := hb_mass
  exact ot_HGamma_formula_uniform_logRatio_bound_from_minMass_le_one
    logRatio hgamma hmin_b hmin_le_one hC_max hupper hlower

/--
Upper Sinkhorn log-ratio estimate from a row-scaling equation.

This is the finite algebraic step used in the OT proof: if the positive column weights satisfy
`min_b <= b_j` and each normalized row obeys
`sum_j b_j * exp(r_ij) = 1`, then every term of the nonnegative sum is at most one, hence
`exp(r_ij) <= 1 / min_b` and `r_ij <= -log(min_b)`.
-/
theorem ot_logRatio_upper_from_rowScaling
    {ι₁ ι₂ : Type*} [Fintype ι₂]
    (b : ι₂ → ℝ)
    (logRatio : ι₁ → ι₂ → ℝ)
    {min_b : ℝ}
    (hmin_b : 0 < min_b)
    (hmin_le_marginal : ∀ j, min_b ≤ b j)
    (hrowScaling : ∀ i, (∑ j, b j * Real.exp (logRatio i j)) = 1) :
    ∀ i j, logRatio i j ≤ -Real.log min_b := by
  classical
  intro i j
  have hbpos : ∀ j, 0 < b j := fun j => lt_of_lt_of_le hmin_b (hmin_le_marginal j)
  have hterm_nonneg :
      ∀ k, 0 ≤ b k * Real.exp (logRatio i k) := by
    intro k
    exact mul_nonneg (le_of_lt (hbpos k)) (le_of_lt (Real.exp_pos _))
  have hterm_le_sum :
      b j * Real.exp (logRatio i j) ≤ ∑ k, b k * Real.exp (logRatio i k) := by
    simpa using
      (Finset.single_le_sum
        (s := Finset.univ)
        (f := fun k => b k * Real.exp (logRatio i k))
        (by intro k _hk; exact hterm_nonneg k)
        (by simp : j ∈ (Finset.univ : Finset ι₂)))
  have hterm_le_one : b j * Real.exp (logRatio i j) ≤ 1 := by
    simpa [hrowScaling i] using hterm_le_sum
  have hexp_le_inv_b : Real.exp (logRatio i j) ≤ 1 / b j := by
    exact (le_div_iff₀ (hbpos j)).2 (by simpa [mul_comm] using hterm_le_one)
  have hinv_le_inv_min : 1 / b j ≤ 1 / min_b :=
    one_div_le_one_div_of_le hmin_b (hmin_le_marginal j)
  have hexp_le_inv_min : Real.exp (logRatio i j) ≤ 1 / min_b :=
    hexp_le_inv_b.trans hinv_le_inv_min
  have hlog : logRatio i j ≤ Real.log (1 / min_b) :=
    (Real.le_log_iff_exp_le (by positivity : 0 < 1 / min_b)).2 hexp_le_inv_min
  simpa [one_div, Real.log_inv] using hlog

/--
Lower Sinkhorn log-ratio estimate from the finite `K_min` denominator certificate.

This is the algebraic core of the lower-bound paragraph in Proposition `app-prop:hgamma-ot`.
If `K_ij >= exp(-C_max/gamma)`, the denominator is bounded by `exp(C_max/gamma)`, and
`exp(r_ij) = K_ij / denominator_ij`, then
`r_ij >= -2*C_max/gamma`.
-/
theorem ot_logRatio_lower_from_kernel_denominator_bound
    {ι₁ ι₂ : Type*}
    (K denominator : ι₁ → ι₂ → ℝ)
    (logRatio : ι₁ → ι₂ → ℝ)
    {C_max gamma : ℝ}
    (hgamma : 0 < gamma)
    (_hC_max : 0 ≤ C_max)
    (hK_lower : ∀ i j, Real.exp (-(C_max / gamma)) ≤ K i j)
    (hden_upper : ∀ i j, denominator i j ≤ Real.exp (C_max / gamma))
    (hlogRatio_exp :
      ∀ i j, Real.exp (logRatio i j) = K i j / denominator i j) :
    ∀ i j, -(2 * C_max / gamma) ≤ logRatio i j := by
  intro i j
  have hK_pos : 0 < K i j :=
    lt_of_lt_of_le (Real.exp_pos _) (hK_lower i j)
  have hratio_pos : 0 < K i j / denominator i j := by
    rw [← hlogRatio_exp i j]
    exact Real.exp_pos _
  have hden_pos : 0 < denominator i j := by
    rcases (div_pos_iff.mp hratio_pos) with hpos | hneg
    · exact hpos.2
    · exact False.elim ((not_lt_of_ge (le_of_lt hK_pos)) hneg.1)
  have hscale_nonneg : 0 ≤ Real.exp (-(2 * C_max / gamma)) :=
    le_of_lt (Real.exp_pos _)
  have hmul_den :
      Real.exp (-(2 * C_max / gamma)) * denominator i j ≤
        Real.exp (-(2 * C_max / gamma)) * Real.exp (C_max / gamma) :=
    mul_le_mul_of_nonneg_left (hden_upper i j) hscale_nonneg
  have hmul_exp :
      Real.exp (-(2 * C_max / gamma)) * Real.exp (C_max / gamma) =
        Real.exp (-(C_max / gamma)) := by
    rw [← Real.exp_add]
    congr
    field_simp [ne_of_gt hgamma]
    ring
  have hmul_le_K :
      Real.exp (-(2 * C_max / gamma)) * denominator i j ≤ K i j := by
    calc
      Real.exp (-(2 * C_max / gamma)) * denominator i j
          ≤ Real.exp (-(2 * C_max / gamma)) * Real.exp (C_max / gamma) := hmul_den
      _ = Real.exp (-(C_max / gamma)) := hmul_exp
      _ ≤ K i j := hK_lower i j
  have hexp_le_ratio :
      Real.exp (-(2 * C_max / gamma)) ≤ K i j / denominator i j :=
    (le_div_iff₀ hden_pos).2 hmul_le_K
  have hexp_le :
      Real.exp (-(2 * C_max / gamma)) ≤ Real.exp (logRatio i j) := by
    simpa [hlogRatio_exp i j] using hexp_le_ratio
  exact Real.exp_le_exp.mp hexp_le

/--
Finite Sinkhorn denominator upper bound from scaling data.

This formalizes the denominator estimate in the lower-bound paragraph of Proposition
`app-prop:hgamma-ot`.  If `0 <= u`, `0 <= v`, the kernel satisfies
`K_ij <= 1` and `K_ij >= exp(-C_max/gamma)`, and the scaled coupling has total mass one,
then the Sinkhorn denominator `(Kv)_i (K^T u)_j` is bounded by `exp(C_max/gamma)`.
-/
theorem ot_sinkhorn_denominator_upper_from_scaling
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    (u : ι₁ → ℝ) (v : ι₂ → ℝ) (K : ι₁ → ι₂ → ℝ)
    {C_max gamma : ℝ}
    (hK_upper : ∀ i j, K i j ≤ 1)
    (hK_lower : ∀ i j, Real.exp (-(C_max / gamma)) ≤ K i j)
    (hu_nonneg : ∀ i, 0 ≤ u i)
    (hv_nonneg : ∀ j, 0 ≤ v j)
    (hmass : (∑ i, ∑ j, u i * K i j * v j) = 1) :
    ∀ i j,
      (∑ j', K i j' * v j') * (∑ i', K i' j * u i') ≤
        Real.exp (C_max / gamma) := by
  classical
  intro i j
  let a : ℝ := Real.exp (-(C_max / gamma))
  have ha_pos : 0 < a := Real.exp_pos _
  have hK_nonneg : ∀ i j, 0 ≤ K i j := by
    intro i j
    exact le_trans (le_of_lt ha_pos) (hK_lower i j)
  have hsumv_nonneg : 0 ≤ ∑ j, v j :=
    Finset.sum_nonneg (fun j _ => hv_nonneg j)
  have hrow_le_sumv : (∑ j', K i j' * v j') ≤ ∑ j', v j' := by
    refine Finset.sum_le_sum ?_
    intro j' _
    simpa using mul_le_mul_of_nonneg_right (hK_upper i j') (hv_nonneg j')
  have hcol_le_sumu : (∑ i', K i' j * u i') ≤ ∑ i', u i' := by
    refine Finset.sum_le_sum ?_
    intro i' _
    simpa [mul_comm] using mul_le_mul_of_nonneg_right (hK_upper i' j) (hu_nonneg i')
  have hcol_nonneg : 0 ≤ ∑ i', K i' j * u i' :=
    Finset.sum_nonneg (fun i' _ => mul_nonneg (hK_nonneg i' j) (hu_nonneg i'))
  have hden_le_sums :
      (∑ j', K i j' * v j') * (∑ i', K i' j * u i') ≤
        (∑ j', v j') * (∑ i', u i') := by
    exact mul_le_mul hrow_le_sumv hcol_le_sumu hcol_nonneg hsumv_nonneg
  have hterm_lower :
      ∀ i j, a * (u i * v j) ≤ u i * K i j * v j := by
    intro i j
    have h1 : u i * a ≤ u i * K i j :=
      mul_le_mul_of_nonneg_left (hK_lower i j) (hu_nonneg i)
    have h2 : (u i * a) * v j ≤ (u i * K i j) * v j :=
      mul_le_mul_of_nonneg_right h1 (hv_nonneg j)
    calc
      a * (u i * v j) = (u i * a) * v j := by ring
      _ ≤ (u i * K i j) * v j := h2
      _ = u i * K i j * v j := by ring
  have hsum_lower :
      (∑ i, ∑ j, a * (u i * v j)) ≤ ∑ i, ∑ j, u i * K i j * v j := by
    refine Finset.sum_le_sum ?_
    intro i _
    refine Finset.sum_le_sum ?_
    intro j _
    exact hterm_lower i j
  have hleft_eq :
      (∑ i, ∑ j, a * (u i * v j)) = a * (∑ i, u i) * (∑ j, v j) := by
    simp [Finset.mul_sum, Finset.sum_mul, mul_assoc, mul_left_comm, mul_comm]
  have hsums_le_one : a * (∑ i, u i) * (∑ j, v j) ≤ 1 := by
    calc
      a * (∑ i, u i) * (∑ j, v j) = ∑ i, ∑ j, a * (u i * v j) := hleft_eq.symm
      _ ≤ ∑ i, ∑ j, u i * K i j * v j := hsum_lower
      _ = 1 := hmass
  have hsums_le_one' : ((∑ i, u i) * (∑ j, v j)) * a ≤ 1 := by
    convert hsums_le_one using 1
    ring
  have hprod_le : (∑ i, u i) * (∑ j, v j) ≤ 1 / a :=
    (le_div_iff₀ ha_pos).2 hsums_le_one'
  have hinv : 1 / a = Real.exp (C_max / gamma) := by
    simp [a, Real.exp_neg]
  calc
    (∑ j', K i j' * v j') * (∑ i', K i' j * u i')
        ≤ (∑ j', v j') * (∑ i', u i') := hden_le_sums
    _ = (∑ i', u i') * (∑ j', v j') := by ring
    _ ≤ 1 / a := hprod_le
    _ = Real.exp (C_max / gamma) := hinv

/--
Total scaled mass from a left Sinkhorn marginal equation.

This replaces the raw certificate `sum_ij u_i K_ij v_j = 1` by the structural finite
Sinkhorn equation `u_i (Kv)_i = a_i` together with the row marginal normalization
`sum_i a_i = 1`.
-/
theorem ot_scaled_mass_from_left_marginal_scaling
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    (a u : ι₁ → ℝ) (v : ι₂ → ℝ) (K : ι₁ → ι₂ → ℝ)
    (ha_mass : (∑ i, a i) = 1)
    (hleftScaling : ∀ i, u i * (∑ j, K i j * v j) = a i) :
    (∑ i, ∑ j, u i * K i j * v j) = 1 := by
  classical
  have hrow : ∀ i, (∑ j, u i * K i j * v j) = a i := by
    intro i
    calc
      (∑ j, u i * K i j * v j) = u i * (∑ j, K i j * v j) := by
        simp [Finset.mul_sum, mul_assoc]
      _ = a i := hleftScaling i
  calc
    (∑ i, ∑ j, u i * K i j * v j) = ∑ i, a i := by
      exact Finset.sum_congr rfl (fun i _ => hrow i)
    _ = 1 := ha_mass

/--
Total scaled mass from a right Sinkhorn marginal equation.

This is the column-marginal analogue of `ot_scaled_mass_from_left_marginal_scaling`.  It is the
cleaner certificate for Proposition `app-prop:hgamma-ot`, because the row-scaling normalization
already uses the column marginal `b`.  Thus the total scaled mass follows from
`v_j (Kᵀu)_j = b_j` and `sum_j b_j = 1`, without introducing an additional row marginal.
-/
theorem ot_scaled_mass_from_right_marginal_scaling
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    (b : ι₂ → ℝ) (u : ι₁ → ℝ) (v : ι₂ → ℝ) (K : ι₁ → ι₂ → ℝ)
    (hb_mass : (∑ j, b j) = 1)
    (hrightScaling : ∀ j, v j * (∑ i, K i j * u i) = b j) :
    (∑ i, ∑ j, u i * K i j * v j) = 1 := by
  classical
  have hcol : ∀ j, (∑ i, u i * K i j * v j) = b j := by
    intro j
    calc
      (∑ i, u i * K i j * v j) = v j * (∑ i, K i j * u i) := by
          simp [Finset.mul_sum, mul_comm, mul_left_comm]
      _ = b j := hrightScaling j
  calc
    (∑ i, ∑ j, u i * K i j * v j) =
        ∑ j, ∑ i, u i * K i j * v j := by
      rw [Finset.sum_comm]
    _ = ∑ j, b j := by
      exact Finset.sum_congr rfl (fun j _ => hcol j)
    _ = 1 := hb_mass

/--
Gibbs-kernel bounds from bounded nonnegative costs.

For the OT specialization, the kernel is `K_ij = exp(-C_ij / gamma)`.  The assumptions
`0 <= C_ij <= C_max` and `gamma > 0` imply both kernel bounds used by the finite Sinkhorn
denominator estimate: `K_ij <= 1` and `exp(-C_max/gamma) <= K_ij`.
-/
theorem ot_gibbsKernel_bounds_from_cost_bounds
    {ι₁ ι₂ : Type*}
    (C : ι₁ → ι₂ → ℝ)
    {C_max gamma : ℝ}
    (hgamma : 0 < gamma)
    (hC_nonneg : ∀ i j, 0 ≤ C i j)
    (hC_le : ∀ i j, C i j ≤ C_max) :
    (∀ i j, Real.exp (-(C i j / gamma)) ≤ 1) ∧
      (∀ i j, Real.exp (-(C_max / gamma)) ≤ Real.exp (-(C i j / gamma))) := by
  constructor
  · intro i j
    have hdiv_nonneg : 0 ≤ C i j / gamma :=
      div_nonneg (hC_nonneg i j) (le_of_lt hgamma)
    have hle : -(C i j / gamma) ≤ 0 := by
      linarith
    simpa using (Real.exp_le_exp.mpr hle)
  · intro i j
    have hdiv_le : C i j / gamma ≤ C_max / gamma :=
      div_le_div_of_nonneg_right (hC_le i j) (le_of_lt hgamma)
    have hle : -(C_max / gamma) ≤ -(C i j / gamma) := by
      linarith
    exact Real.exp_le_exp.mpr hle

/--
Row-scaling version of Proposition `app-prop:hgamma-ot`.

Compared with `ot_HGamma_formula_uniform_logRatio_bound_from_probabilityMarginal`, this endpoint
does not assume the upper log-ratio estimate.  It derives it from the concrete finite
normalization equation `sum_j b_j * exp(r_ij) = 1`, which is exactly the Sinkhorn row-scaling
identity after writing `r_ij = log(Pγ_ij) - log(z_ij)`.
-/
theorem ot_HGamma_formula_uniform_logRatio_bound_from_rowScaling
    {ι₁ ι₂ : Type*} [Fintype ι₂] [Nonempty ι₂]
    (b : ι₂ → ℝ)
    (logRatio : ι₁ → ι₂ → ℝ)
    {min_b C_max gamma : ℝ}
    (hgamma : 0 < gamma)
    (hmin_b : 0 < min_b)
    (hb_mass : (∑ j, b j) = 1)
    (hmin_le_marginal : ∀ j, min_b ≤ b j)
    (hC_max : 0 ≤ C_max)
    (hrowScaling : ∀ i, (∑ j, b j * Real.exp (logRatio i j)) = 1)
    (hlower : ∀ i j, -(2 * C_max / gamma) ≤ logRatio i j) :
    (∀ i j, |logRatio i j| ≤ |Real.log min_b| + 2 * C_max / gamma) ∧
      0 ≤ |Real.log min_b| + 2 * C_max / gamma := by
  classical
  obtain ⟨j₀⟩ := ‹Nonempty ι₂›
  have hbpos : ∀ j, 0 < b j := fun j => lt_of_lt_of_le hmin_b (hmin_le_marginal j)
  have hb_j_le_sum : b j₀ ≤ ∑ j, b j := by
    simpa using
      (Finset.single_le_sum
        (s := Finset.univ) (f := b)
        (by intro j _hj; exact le_of_lt (hbpos j))
        (by simp : j₀ ∈ (Finset.univ : Finset ι₂)))
  have hmin_le_one : min_b ≤ 1 := by
    calc
      min_b ≤ b j₀ := hmin_le_marginal j₀
      _ ≤ ∑ j, b j := hb_j_le_sum
      _ = 1 := hb_mass
  have hupper : ∀ i j, logRatio i j ≤ -Real.log min_b :=
    ot_logRatio_upper_from_rowScaling b logRatio hmin_b hmin_le_marginal hrowScaling
  exact ot_HGamma_formula_uniform_logRatio_bound_from_minMass_le_one
    logRatio hgamma hmin_b hmin_le_one hC_max hupper hlower

/--
Row-scaling and `K_min`-denominator version of Proposition `app-prop:hgamma-ot`.

Compared with `ot_HGamma_formula_uniform_logRatio_bound_from_rowScaling`, this endpoint no
longer assumes the lower log-ratio estimate directly.  Lean derives the upper estimate from the
row-scaling equation and the lower estimate from the finite kernel/denominator certificate.
-/
theorem ot_HGamma_formula_uniform_logRatio_bound_from_rowScaling_kernelDenominator
    {ι₁ ι₂ : Type*} [Fintype ι₂] [Nonempty ι₂]
    (b : ι₂ → ℝ)
    (K denominator logRatio : ι₁ → ι₂ → ℝ)
    {min_b C_max gamma : ℝ}
    (hgamma : 0 < gamma)
    (hmin_b : 0 < min_b)
    (hb_mass : (∑ j, b j) = 1)
    (hmin_le_marginal : ∀ j, min_b ≤ b j)
    (hC_max : 0 ≤ C_max)
    (hrowScaling : ∀ i, (∑ j, b j * Real.exp (logRatio i j)) = 1)
    (hK_lower : ∀ i j, Real.exp (-(C_max / gamma)) ≤ K i j)
    (hden_upper : ∀ i j, denominator i j ≤ Real.exp (C_max / gamma))
    (hlogRatio_exp :
      ∀ i j, Real.exp (logRatio i j) = K i j / denominator i j) :
    (∀ i j, |logRatio i j| ≤ |Real.log min_b| + 2 * C_max / gamma) ∧
      0 ≤ |Real.log min_b| + 2 * C_max / gamma := by
  have hlower : ∀ i j, -(2 * C_max / gamma) ≤ logRatio i j :=
    ot_logRatio_lower_from_kernel_denominator_bound
      K denominator logRatio hgamma hC_max hK_lower hden_upper hlogRatio_exp
  exact ot_HGamma_formula_uniform_logRatio_bound_from_rowScaling
    b logRatio hgamma hmin_b hb_mass hmin_le_marginal hC_max hrowScaling hlower

/--
Row-scaling and concrete scaling-vector version of Proposition `app-prop:hgamma-ot`.

This endpoint internalizes the finite denominator estimate used by the classical Sinkhorn
specialization.  Instead of assuming `(Kv)_i (K^T u)_j <= exp(C_max/gamma)`, Lean derives it from
`K <= 1`, the `K_min` lower bound, nonnegative scaling vectors, and total mass one.
-/
theorem ot_HGamma_formula_uniform_logRatio_bound_from_rowScaling_sinkhornScaling
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂] [Nonempty ι₂]
    (b : ι₂ → ℝ)
    (u : ι₁ → ℝ) (v : ι₂ → ℝ)
    (K logRatio : ι₁ → ι₂ → ℝ)
    {min_b C_max gamma : ℝ}
    (hgamma : 0 < gamma)
    (hmin_b : 0 < min_b)
    (hb_mass : (∑ j, b j) = 1)
    (hmin_le_marginal : ∀ j, min_b ≤ b j)
    (hC_max : 0 ≤ C_max)
    (hrowScaling : ∀ i, (∑ j, b j * Real.exp (logRatio i j)) = 1)
    (hK_upper : ∀ i j, K i j ≤ 1)
    (hK_lower : ∀ i j, Real.exp (-(C_max / gamma)) ≤ K i j)
    (hu_nonneg : ∀ i, 0 ≤ u i)
    (hv_nonneg : ∀ j, 0 ≤ v j)
    (hscaled_mass : (∑ i, ∑ j, u i * K i j * v j) = 1)
    (hlogRatio_exp :
      ∀ i j, Real.exp (logRatio i j) =
        K i j / ((∑ j', K i j' * v j') * (∑ i', K i' j * u i'))) :
    (∀ i j, |logRatio i j| ≤ |Real.log min_b| + 2 * C_max / gamma) ∧
      0 ≤ |Real.log min_b| + 2 * C_max / gamma := by
  let denominator : ι₁ → ι₂ → ℝ :=
    fun i j => (∑ j', K i j' * v j') * (∑ i', K i' j * u i')
  have hden_upper : ∀ i j, denominator i j ≤ Real.exp (C_max / gamma) :=
    ot_sinkhorn_denominator_upper_from_scaling
      u v K hK_upper hK_lower hu_nonneg hv_nonneg hscaled_mass
  have hlogRatio_exp' :
      ∀ i j, Real.exp (logRatio i j) = K i j / denominator i j := by
    intro i j
    exact hlogRatio_exp i j
  exact ot_HGamma_formula_uniform_logRatio_bound_from_rowScaling_kernelDenominator
    b K denominator logRatio hgamma hmin_b hb_mass hmin_le_marginal hC_max
    hrowScaling hK_lower hden_upper hlogRatio_exp'

/--
Row-scaling and cost-kernel scaling version of Proposition `app-prop:hgamma-ot`.

This is the paper-facing OT endpoint closest to the classical Sinkhorn specialization.  The kernel
is not an abstract input: it is the Gibbs kernel `exp(-C_ij/gamma)`.  Lean derives the kernel
bounds `K <= 1` and `K >= exp(-C_max/gamma)` from `0 <= C_ij <= C_max`, then reuses the finite
scaling-vector denominator theorem.
-/
theorem ot_HGamma_formula_uniform_logRatio_bound_from_rowScaling_costScaling
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂] [Nonempty ι₁] [Nonempty ι₂]
    (b : ι₂ → ℝ)
    (u : ι₁ → ℝ) (v : ι₂ → ℝ)
    (C logRatio : ι₁ → ι₂ → ℝ)
    {min_b C_max gamma : ℝ}
    (hgamma : 0 < gamma)
    (hmin_b : 0 < min_b)
    (hb_mass : (∑ j, b j) = 1)
    (hmin_le_marginal : ∀ j, min_b ≤ b j)
    (hC_nonneg : ∀ i j, 0 ≤ C i j)
    (hC_le : ∀ i j, C i j ≤ C_max)
    (hrowScaling : ∀ i, (∑ j, b j * Real.exp (logRatio i j)) = 1)
    (hu_nonneg : ∀ i, 0 ≤ u i)
    (hv_nonneg : ∀ j, 0 ≤ v j)
    (hscaled_mass :
      (∑ i, ∑ j, u i * Real.exp (-(C i j / gamma)) * v j) = 1)
    (hlogRatio_exp :
      ∀ i j, Real.exp (logRatio i j) =
        Real.exp (-(C i j / gamma)) /
          ((∑ j', Real.exp (-(C i j' / gamma)) * v j') *
            (∑ i', Real.exp (-(C i' j / gamma)) * u i'))) :
    (∀ i j, |logRatio i j| ≤ |Real.log min_b| + 2 * C_max / gamma) ∧
      0 ≤ |Real.log min_b| + 2 * C_max / gamma := by
  classical
  obtain ⟨i₀⟩ := ‹Nonempty ι₁›
  obtain ⟨j₀⟩ := ‹Nonempty ι₂›
  have hC_max : 0 ≤ C_max := by
    exact le_trans (hC_nonneg i₀ j₀) (hC_le i₀ j₀)
  have hbounds :=
    ot_gibbsKernel_bounds_from_cost_bounds C hgamma hC_nonneg hC_le
  exact ot_HGamma_formula_uniform_logRatio_bound_from_rowScaling_sinkhornScaling
    b u v (fun i j => Real.exp (-(C i j / gamma))) logRatio
    hgamma hmin_b hb_mass hmin_le_marginal hC_max hrowScaling
    hbounds.1 hbounds.2 hu_nonneg hv_nonneg hscaled_mass hlogRatio_exp

/--
Row-scaling, cost-kernel, and left-marginal scaling version of Proposition
`app-prop:hgamma-ot`.

Compared with `ot_HGamma_formula_uniform_logRatio_bound_from_rowScaling_costScaling`, this
endpoint no longer assumes the total scaled mass directly.  Lean derives it from the left
Sinkhorn marginal equation `u_i (Kv)_i = a_i` and the normalization `sum_i a_i = 1`.
-/
theorem ot_HGamma_formula_uniform_logRatio_bound_from_rowScaling_costLeftScaling
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂] [Nonempty ι₁] [Nonempty ι₂]
    (a : ι₁ → ℝ) (b : ι₂ → ℝ)
    (u : ι₁ → ℝ) (v : ι₂ → ℝ)
    (C logRatio : ι₁ → ι₂ → ℝ)
    {min_b C_max gamma : ℝ}
    (hgamma : 0 < gamma)
    (hmin_b : 0 < min_b)
    (ha_mass : (∑ i, a i) = 1)
    (hb_mass : (∑ j, b j) = 1)
    (hmin_le_marginal : ∀ j, min_b ≤ b j)
    (hC_nonneg : ∀ i j, 0 ≤ C i j)
    (hC_le : ∀ i j, C i j ≤ C_max)
    (hrowScaling : ∀ i, (∑ j, b j * Real.exp (logRatio i j)) = 1)
    (hu_nonneg : ∀ i, 0 ≤ u i)
    (hv_nonneg : ∀ j, 0 ≤ v j)
    (hleftScaling :
      ∀ i, u i * (∑ j, Real.exp (-(C i j / gamma)) * v j) = a i)
    (hlogRatio_exp :
      ∀ i j, Real.exp (logRatio i j) =
        Real.exp (-(C i j / gamma)) /
          ((∑ j', Real.exp (-(C i j' / gamma)) * v j') *
            (∑ i', Real.exp (-(C i' j / gamma)) * u i'))) :
    (∀ i j, |logRatio i j| ≤ |Real.log min_b| + 2 * C_max / gamma) ∧
      0 ≤ |Real.log min_b| + 2 * C_max / gamma := by
  classical
  have hscaled_mass :
      (∑ i, ∑ j, u i * Real.exp (-(C i j / gamma)) * v j) = 1 :=
    ot_scaled_mass_from_left_marginal_scaling
      a u v (fun i j => Real.exp (-(C i j / gamma))) ha_mass hleftScaling
  exact ot_HGamma_formula_uniform_logRatio_bound_from_rowScaling_costScaling
    b u v C logRatio hgamma hmin_b hb_mass hmin_le_marginal
    hC_nonneg hC_le hrowScaling hu_nonneg hv_nonneg hscaled_mass hlogRatio_exp

/--
Row-scaling, cost-kernel, and right-marginal scaling version of Proposition
`app-prop:hgamma-ot`.

Compared with `ot_HGamma_formula_uniform_logRatio_bound_from_rowScaling_costLeftScaling`, this
endpoint no longer introduces a separate row marginal `a`.  Lean derives the total scaled mass
from the right Sinkhorn marginal equation `v_j (Kᵀu)_j = b_j` and the existing normalization
`sum_j b_j = 1`.
-/
theorem ot_HGamma_formula_uniform_logRatio_bound_from_rowScaling_costRightScaling
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂] [Nonempty ι₁] [Nonempty ι₂]
    (b : ι₂ → ℝ)
    (u : ι₁ → ℝ) (v : ι₂ → ℝ)
    (C logRatio : ι₁ → ι₂ → ℝ)
    {min_b C_max gamma : ℝ}
    (hgamma : 0 < gamma)
    (hmin_b : 0 < min_b)
    (hb_mass : (∑ j, b j) = 1)
    (hmin_le_marginal : ∀ j, min_b ≤ b j)
    (hC_nonneg : ∀ i j, 0 ≤ C i j)
    (hC_le : ∀ i j, C i j ≤ C_max)
    (hrowScaling : ∀ i, (∑ j, b j * Real.exp (logRatio i j)) = 1)
    (hu_nonneg : ∀ i, 0 ≤ u i)
    (hv_nonneg : ∀ j, 0 ≤ v j)
    (hrightScaling :
      ∀ j, v j * (∑ i, Real.exp (-(C i j / gamma)) * u i) = b j)
    (hlogRatio_exp :
      ∀ i j, Real.exp (logRatio i j) =
        Real.exp (-(C i j / gamma)) /
          ((∑ j', Real.exp (-(C i j' / gamma)) * v j') *
            (∑ i', Real.exp (-(C i' j / gamma)) * u i'))) :
    (∀ i j, |logRatio i j| ≤ |Real.log min_b| + 2 * C_max / gamma) ∧
      0 ≤ |Real.log min_b| + 2 * C_max / gamma := by
  classical
  have hscaled_mass :
      (∑ i, ∑ j, u i * Real.exp (-(C i j / gamma)) * v j) = 1 :=
    ot_scaled_mass_from_right_marginal_scaling
      b u v (fun i j => Real.exp (-(C i j / gamma))) hb_mass hrightScaling
  exact ot_HGamma_formula_uniform_logRatio_bound_from_rowScaling_costScaling
    b u v C logRatio hgamma hmin_b hb_mass hmin_le_marginal
    hC_nonneg hC_le hrowScaling hu_nonneg hv_nonneg hscaled_mass hlogRatio_exp

/--
Typed finite-data version of Proposition `app-prop:hgamma-ot`.

The paper states the OT certificate using a probability vector `b`, nonnegative
scaling vectors `u,v`, and a cost field bounded by `0 <= C_ij <= C_max`.  This
endpoint stores those facts in data records and delegates to the real-valued
right-scaling theorem above.
-/
theorem ot_HGamma_formula_uniform_logRatio_bound_from_typedRightScaling
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂] [Nonempty ι₁] [Nonempty ι₂]
    (b : ProbabilityVector ι₂)
    (u : NonnegativeField ι₁) (v : NonnegativeField ι₂)
    {C_max : ℝ}
    (C : BoundedCostField ι₁ ι₂ C_max)
    (logRatio : ι₁ → ι₂ → ℝ)
    {min_b gamma : ℝ}
    (hgamma : 0 < gamma)
    (hmin_b : 0 < min_b)
    (hmin_le_marginal : ∀ j, min_b ≤ b j)
    (hrowScaling : ∀ i, (∑ j, b j * Real.exp (logRatio i j)) = 1)
    (hrightScaling :
      ∀ j, v j * (∑ i, Real.exp (-(C i j / gamma)) * u i) = b j)
    (hlogRatio_exp :
      ∀ i j, Real.exp (logRatio i j) =
        Real.exp (-(C i j / gamma)) /
          ((∑ j', Real.exp (-(C i j' / gamma)) * v j') *
            (∑ i', Real.exp (-(C i' j / gamma)) * u i'))) :
    (∀ i j, |logRatio i j| ≤ |Real.log min_b| + 2 * C_max / gamma) ∧
      0 ≤ |Real.log min_b| + 2 * C_max / gamma := by
  exact
    ot_HGamma_formula_uniform_logRatio_bound_from_rowScaling_costRightScaling
      (b := fun j => b j)
      (u := fun i => u i)
      (v := fun j => v j)
      (C := fun i j => C i j)
      (logRatio := logRatio)
      (min_b := min_b)
      (C_max := C_max)
      (gamma := gamma)
      hgamma hmin_b b.mass hmin_le_marginal C.nonneg C.le_bound hrowScaling
      u.nonneg v.nonneg hrightScaling hlogRatio_exp

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
