import FlowSinkhorn.KLProjection.DualConvergence.GapResidual
import FlowSinkhorn.KLProjection.DualConvergence.Vocabulary
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# Robust `O(1/k)` rate

This module is reserved for the Lean formalization of Theorem `thm:KL-dual-rate` and the
regularization/complexity corollaries from
the dual-convergence material in `neurips/paper.tex`.

Paper targets:
- Theorem `thm:KL-dual-rate`;
- Lemma `lem:bias-T`;
- Theorem `thm:approx-linprog`.

Intended theorem names:
- `dualRate_O_one_over_k`;
- `dualRate_iterationThreshold_of_masterAbstractRate`;
- `dualRate_iterationThreshold_of_ratioBound`;
- `dualRate_iterationThreshold_of_closedFormCeil`;
- `regularizedApproximation_stoppingRule_of_closedFormIterationThreshold`;
- `klBias_bound`;
- `klBias_regularizedGap_and_constantReference`;
- `regularizedApproximation_error_le_eps_of_closedFormIterationThreshold`;
- `regularizedApproximation_finalEpsilon_of_stoppingRule`;
- `regularizedApproximation_error_le_eps_of_iterationThreshold`;
- `regularizedApproximation_complexity_of_closedFormIterationThreshold`;
- `regularizedApproximation_paperEpsilon_of_KLRate_closedFormIterationThreshold`;
- `regularizedApproximation_targetAccuracy_of_closedFormIterationThreshold`;
- `regularizedApproximation_complexity`.

Design note:
this file should expose a paper-faithful top theorem whose hypotheses are exactly the constants
later discharged in `PrimalDualBounds` and the application modules.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace DualConvergence

open scoped BigOperators

/--
Sequence lemma behind an `O(1/k)` bound:
if one has a per-iterate estimate `(k+1) * gap_k ≤ S_k` and a uniform bound `S_k ≤ C`,
then `gap_k ≤ C/(k+1)`.
-/
theorem dualRate_O_one_over_k_of_antitone_of_partialSum_bound
    {gap S : ℕ → ℝ} {C : ℝ}
    (hstep : ∀ n : ℕ, ((n + 1 : ℝ) * gap n) ≤ S n)
    (hS : ∀ n : ℕ, S n ≤ C)
    (n : ℕ) :
    gap n ≤ C / (n + 1 : ℝ) := by
  have hmulC : ((n + 1 : ℝ) * gap n) ≤ C := (hstep n).trans (hS n)
  have hpos : 0 < (n + 1 : ℝ) := by
    exact_mod_cast Nat.succ_pos n
  have hmulC' : gap n * (n + 1 : ℝ) ≤ C := by
    simpa [mul_comm, mul_left_comm, mul_assoc] using hmulC
  have hden : (n + 1 : ℝ) ≠ 0 := ne_of_gt hpos
  field_simp [hden]
  simpa [mul_comm, mul_left_comm, mul_assoc] using hmulC'

/--
Paper-facing wrapper for Theorem `thm:KL-dual-rate` once the partial-sum control has been proved.
-/
theorem dualRate_O_one_over_k
    {gap S : ℕ → ℝ} {U : ℝ}
    (hstep : ∀ n : ℕ, ((n + 1 : ℝ) * gap n) ≤ S n)
    (hS : ∀ n : ℕ, S n ≤ U)
    (n : ℕ) :
    gap n ≤ U / (n + 1 : ℝ) :=
  dualRate_O_one_over_k_of_antitone_of_partialSum_bound hstep hS n

/--
Convenient specialization: antitonicity plus a uniform cumulative bound directly yield `O(1/k)`.
-/
theorem dualRate_O_one_over_k_of_antitone_of_cumulativeBound
    {gap : ℕ → ℝ} {U : ℝ}
    (hmono : Antitone gap)
    (hcum : ∀ n : ℕ, cumulative gap (n + 1) ≤ U)
    (n : ℕ) :
    gap n ≤ U / (n + 1 : ℝ) := by
  apply dualRate_O_one_over_k (gap := gap) (S := fun k => cumulative gap (k + 1))
  · intro k
    exact mul_le_cumulative_of_antitone hmono k
  · intro k
    exact hcum k

/--
Linear cumulative-budget corollary for pointwise bounded residuals.

This is the arithmetic companion to `cumulative_le_linear_of_uniform_bound`, exposed with a more
paper-facing name so applications can refer to it directly.
-/
theorem cumulativeBudget_linear_of_uniformResidualBound
    {residual : ℕ → ℝ} {B : ℝ}
    (hres : ∀ k : ℕ, residual k ≤ B) :
    ∀ n : ℕ, cumulative residual n ≤ (n : ℝ) * B :=
  cumulative_le_linear_of_uniform_bound hres

/--
Section-3 master abstract rate statement.

This is the canonical rate theorem in the dual-convergence track:
scaled gap control + per-step ascent + bounded objective growth + antitonicity imply an explicit
`O(1/k)` bound with constant `alpha * B`.
-/
theorem dualRate_masterAbstractRateStatement
    {phi gap residual : ℕ → ℝ}
    {alpha B : ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (n : ℕ) :
    gap n ≤ (alpha * B) / (n + 1 : ℝ) := by
  have hcum_gap : ∀ m : ℕ, cumulative gap (m + 1) ≤ alpha * B := by
    intro m
    have hcum_gap_le :
        cumulative gap (m + 1) ≤ cumulative (fun k => alpha * residual k) (m + 1) :=
      cumulative_le_of_le hgap_res (m + 1)
    have hcum_res_le : cumulative residual (m + 1) ≤ B := by
      exact (perStepAscent_twoStep hres_ascent (m + 1)).trans (hphi_bound m)
    have hmul_le : alpha * cumulative residual (m + 1) ≤ alpha * B := by
      nlinarith [halpha, hcum_res_le]
    calc
      cumulative gap (m + 1) ≤ cumulative (fun k => alpha * residual k) (m + 1) := hcum_gap_le
      _ = alpha * cumulative residual (m + 1) := by
            simpa using cumulative_mul_left alpha residual (m + 1)
      _ ≤ alpha * B := hmul_le
  simpa using dualRate_O_one_over_k_of_antitone_of_cumulativeBound hmono_gap hcum_gap n

/--
Backwards-compatible wrapper for the master Section-3 rate theorem.

This keeps the paper-facing theorem name used by earlier iterations while making the canonical path
explicit in the file.
-/
theorem dualRate_O_one_over_k_of_ascent_gap_control
    {phi gap residual : ℕ → ℝ}
    {alpha B : ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (n : ℕ) :
    gap n ≤ (alpha * B) / (n + 1 : ℝ) :=
  dualRate_masterAbstractRateStatement
    (phi := phi) (gap := gap) (residual := residual)
    (alpha := alpha) (B := B)
    halpha hgap_res hres_ascent hphi_bound hmono_gap n

/--
Paper-constant version of Theorem `thm:kl-dual-rate`.

This theorem keeps the master-rate hypotheses explicit, but no longer hides the manuscript
constant behind abstract symbols.  With the paper specialization
`alpha = 2 * Umax` and
`B = 4 * Xmax * Umax * Anorm^2 / gamma`, the master rate gives
`8 * Xmax * Umax^2 * Anorm^2 / gamma`.

The conclusion is written in zero-based Lean indexing:
`gap n ≤ C / (n+1)`, corresponding to the paper's `1/k` rate for positive iteration indices.
-/
theorem dualRate_KL_paperConstant_from_masterAbstractRate
    {phi gap residual : ℕ → ℝ}
    {gamma Xmax Umax Anorm : ℝ}
    (hgamma_pos : 0 < gamma)
    (hUmax_nonneg : 0 ≤ Umax)
    (hgap_nonneg : ∀ k : ℕ, 0 ≤ gap k)
    (hgap_res : ∀ k : ℕ, gap k ≤ (2 * Umax) * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound :
      ∀ n : ℕ, phi (n + 1) - phi 0 ≤ 4 * Xmax * Umax * Anorm ^ 2 / gamma)
    (hmono_gap : Antitone gap)
    (n : ℕ) :
    0 ≤ gap n ∧
      gap n ≤ (8 * Xmax * Umax ^ 2 * Anorm ^ 2 / gamma) / (n + 1 : ℝ) := by
  have _hgamma_ne : gamma ≠ 0 := ne_of_gt hgamma_pos
  refine ⟨hgap_nonneg n, ?_⟩
  have halpha : 0 ≤ 2 * Umax := by nlinarith
  have hrate :=
    dualRate_masterAbstractRateStatement
      (phi := phi) (gap := gap) (residual := residual)
      (alpha := 2 * Umax) (B := 4 * Xmax * Umax * Anorm ^ 2 / gamma)
      halpha hgap_res hres_ascent hphi_bound hmono_gap n
  have hconst :
      (2 * Umax) * (4 * Xmax * Umax * Anorm ^ 2 / gamma) =
        8 * Xmax * Umax ^ 2 * Anorm ^ 2 / gamma := by
    ring
  simpa [hconst] using hrate

/--
One-step reciprocal-growth lemma used in the paper proof of Theorem `thm:kl-dual-rate`.

If a positive gap sequence satisfies a quadratic descent step
`alpha * a^2 ≤ a - b` and the next gap is no larger than the current one, then
the reciprocal gap grows by at least `alpha`.
-/
theorem reciprocalStep_of_quadraticDescent
    {a b alpha : ℝ}
    (ha : 0 < a) (hb : 0 < b)
    (hmono : b ≤ a)
    (hdescent : alpha * a ^ 2 ≤ a - b) :
    alpha ≤ 1 / b - 1 / a := by
  have ha2_pos : 0 < a ^ 2 := sq_pos_of_pos ha
  have hab_pos : 0 < a * b := mul_pos ha hb
  have h_alpha_frac : alpha ≤ (a - b) / a ^ 2 := by
    exact (le_div_iff₀ ha2_pos).2 hdescent
  have hden_le : a * b ≤ a ^ 2 := by
    nlinarith
  have hfrac_mono : (a - b) / a ^ 2 ≤ (a - b) / (a * b) := by
    rw [div_le_div_iff₀ ha2_pos hab_pos]
    nlinarith
  have hfrac_eq : (a - b) / (a * b) = 1 / b - 1 / a := by
    field_simp [ne_of_gt ha, ne_of_gt hb]
  exact h_alpha_frac.trans (hfrac_mono.trans_eq hfrac_eq)

/--
Telescoping form of the reciprocal-growth argument.

This lemma is deliberately independent of KL geometry: it only says that if
`1/gap` increases by at least `alpha` each iteration, then after `n` steps the
reciprocal has increased by at least `n * alpha`.
-/
theorem reciprocalGrowth_of_step
    {gap : ℕ → ℝ} {alpha : ℝ}
    (hstep : ∀ k : ℕ, alpha ≤ 1 / gap (k + 1) - 1 / gap k) :
    ∀ n : ℕ, (n : ℝ) * alpha ≤ 1 / gap n - 1 / gap 0 := by
  intro n
  induction n with
  | zero => simp
  | succ n ih =>
      have hs := hstep n
      have hsum : (n : ℝ) * alpha + alpha ≤
          (1 / gap n - 1 / gap 0) + (1 / gap (n + 1) - 1 / gap n) :=
        add_le_add ih hs
      calc
        ((n + 1 : ℕ) : ℝ) * alpha = (n : ℝ) * alpha + alpha := by
          norm_num [Nat.cast_add, Nat.cast_one]
          ring
        _ ≤ (1 / gap n - 1 / gap 0) + (1 / gap (n + 1) - 1 / gap n) := hsum
        _ = 1 / gap (n + 1) - 1 / gap 0 := by ring

/--
Rate theorem from the reciprocal-growth proof pattern.

The conclusion is stated at iterate `n + 1`, exactly matching the manuscript's
positive-index statement `k ≥ 1` after converting Lean's zero-based naturals.
-/
theorem dualRate_from_reciprocalStep
    {gap : ℕ → ℝ} {alpha : ℝ}
    (halpha_pos : 0 < alpha)
    (hgap_pos : ∀ k : ℕ, 0 < gap k)
    (hstep : ∀ k : ℕ, alpha ≤ 1 / gap (k + 1) - 1 / gap k)
    (n : ℕ) :
    gap (n + 1) ≤ (1 / alpha) / (n + 1 : ℝ) := by
  have hgrowth := reciprocalGrowth_of_step (gap := gap) (alpha := alpha) hstep (n + 1)
  have hgrowth' : alpha * (n + 1 : ℝ) ≤ 1 / gap (n + 1) - 1 / gap 0 := by
    simpa [mul_comm] using hgrowth
  have hinit_nonneg : 0 ≤ 1 / gap 0 := by
    exact one_div_nonneg.mpr (le_of_lt (hgap_pos 0))
  have halphaN_pos : 0 < alpha * (n + 1 : ℝ) := by
    exact mul_pos halpha_pos (by exact_mod_cast Nat.succ_pos n)
  have hmain : alpha * (n + 1 : ℝ) ≤ 1 / gap (n + 1) := by
    linarith
  have hmul : (alpha * (n + 1 : ℝ)) * gap (n + 1) ≤ 1 := by
    have hg_nonneg : 0 ≤ gap (n + 1) := le_of_lt (hgap_pos (n + 1))
    have h := mul_le_mul_of_nonneg_right hmain hg_nonneg
    have hright : (gap (n + 1))⁻¹ * gap (n + 1) = 1 := by
      field_simp [ne_of_gt (hgap_pos (n + 1))]
    simpa [one_div, hright, mul_assoc] using h
  have htarget : gap (n + 1) ≤ 1 / (alpha * (n + 1 : ℝ)) := by
    exact (le_div_iff₀ halphaN_pos).2
      (by simpa [mul_comm, mul_left_comm, mul_assoc] using hmul)
  have heq : 1 / (alpha * (n + 1 : ℝ)) = (1 / alpha) / (n + 1 : ℝ) := by
    field_simp [ne_of_gt halpha_pos,
      ne_of_gt (by exact_mod_cast Nat.succ_pos n : 0 < (n + 1 : ℝ))]
  simpa [heq] using htarget

/--
Finite-horizon variant of `reciprocalGrowth_of_step`.

Only the reciprocal steps strictly before `N` are needed to telescope up to time `N`.
-/
theorem reciprocalGrowth_of_step_upto
    {gap : ℕ → ℝ} {alpha : ℝ} (N : ℕ)
    (hstep : ∀ k : ℕ, k < N → alpha ≤ 1 / gap (k + 1) - 1 / gap k) :
    (N : ℝ) * alpha ≤ 1 / gap N - 1 / gap 0 := by
  induction N with
  | zero => simp
  | succ N ih =>
      have ih' : (N : ℝ) * alpha ≤ 1 / gap N - 1 / gap 0 := by
        exact ih (by
          intro k hk
          exact hstep k (Nat.lt_trans hk (Nat.lt_succ_self N)))
      have hs : alpha ≤ 1 / gap (N + 1) - 1 / gap N :=
        hstep N (Nat.lt_succ_self N)
      have hsum : (N : ℝ) * alpha + alpha ≤
          (1 / gap N - 1 / gap 0) + (1 / gap (N + 1) - 1 / gap N) :=
        add_le_add ih' hs
      calc
        ((N + 1 : ℕ) : ℝ) * alpha = (N : ℝ) * alpha + alpha := by
          norm_num [Nat.cast_add, Nat.cast_one]
          ring
        _ ≤ (1 / gap N - 1 / gap 0) + (1 / gap (N + 1) - 1 / gap N) := hsum
        _ = 1 / gap (N + 1) - 1 / gap 0 := by ring

/--
Finite-horizon reciprocal-rate theorem.

This version only asks positivity of the gaps that appear in the telescoping proof.  It is used
below to handle the paper-natural nonnegative-gap case by splitting on whether the target gap is
zero.
-/
theorem dualRate_from_reciprocalStep_upto
    {gap : ℕ → ℝ} {alpha : ℝ}
    (halpha_pos : 0 < alpha)
    (n : ℕ)
    (hgap_pos : ∀ k : ℕ, k ≤ n + 1 → 0 < gap k)
    (hstep : ∀ k : ℕ, k ≤ n → alpha ≤ 1 / gap (k + 1) - 1 / gap k) :
    gap (n + 1) ≤ (1 / alpha) / (n + 1 : ℝ) := by
  have hgrowth := reciprocalGrowth_of_step_upto (gap := gap) (alpha := alpha) (n + 1)
    (by
      intro k hk
      exact hstep k (Nat.lt_succ_iff.mp hk))
  have hgrowth' : alpha * (n + 1 : ℝ) ≤ 1 / gap (n + 1) - 1 / gap 0 := by
    simpa [mul_comm] using hgrowth
  have hinit_nonneg : 0 ≤ 1 / gap 0 := by
    exact one_div_nonneg.mpr (le_of_lt (hgap_pos 0 (Nat.zero_le _)))
  have halphaN_pos : 0 < alpha * (n + 1 : ℝ) := by
    exact mul_pos halpha_pos (by exact_mod_cast Nat.succ_pos n)
  have hmain : alpha * (n + 1 : ℝ) ≤ 1 / gap (n + 1) := by
    linarith
  have hmul : (alpha * (n + 1 : ℝ)) * gap (n + 1) ≤ 1 := by
    have hg_nonneg : 0 ≤ gap (n + 1) := le_of_lt (hgap_pos (n + 1) le_rfl)
    have h := mul_le_mul_of_nonneg_right hmain hg_nonneg
    have hright : (gap (n + 1))⁻¹ * gap (n + 1) = 1 := by
      field_simp [ne_of_gt (hgap_pos (n + 1) le_rfl)]
    simpa [one_div, hright, mul_assoc] using h
  have htarget : gap (n + 1) ≤ 1 / (alpha * (n + 1 : ℝ)) := by
    exact (le_div_iff₀ halphaN_pos).2
      (by simpa [mul_comm, mul_left_comm, mul_assoc] using hmul)
  have heq : 1 / (alpha * (n + 1 : ℝ)) = (1 / alpha) / (n + 1 : ℝ) := by
    field_simp [ne_of_gt halpha_pos,
      ne_of_gt (by exact_mod_cast Nat.succ_pos n : 0 < (n + 1 : ℝ))]
  simpa [heq] using htarget

/--
Paper-constant reciprocal-growth version of Theorem `thm:kl-dual-rate`.

This is closer to Appendix A than the cumulative master-rate wrapper: the hypothesis
is the reciprocal growth obtained after combining per-step ascent with the
gap-vs-residual estimate, and the conclusion is the paper constant
`8 * Xmax * Umax^2 * ||A||^2 / gamma` with positive iteration index.
-/
theorem dualRate_KL_paperConstant_from_reciprocalGapGrowth
    {gap : ℕ → ℝ} {gamma Xmax Umax Anorm : ℝ}
    (hgamma_pos : 0 < gamma)
    (hXmax_pos : 0 < Xmax)
    (hUmax_pos : 0 < Umax)
    (hAnorm_pos : 0 < Anorm)
    (hgap_pos : ∀ k : ℕ, 0 < gap k)
    (hreciprocal_step :
      ∀ k : ℕ,
        gamma / (8 * Xmax * Umax ^ 2 * Anorm ^ 2) ≤
          1 / gap (k + 1) - 1 / gap k)
    (n : ℕ) :
    0 ≤ gap (n + 1) ∧
      gap (n + 1) ≤
        (8 * Xmax * Umax ^ 2 * Anorm ^ 2 / gamma) / (n + 1 : ℝ) := by
  have hden_pos : 0 < 8 * Xmax * Umax ^ 2 * Anorm ^ 2 := by positivity
  have halpha_pos : 0 < gamma / (8 * Xmax * Umax ^ 2 * Anorm ^ 2) :=
    div_pos hgamma_pos hden_pos
  refine ⟨le_of_lt (hgap_pos (n + 1)), ?_⟩
  have hrate := dualRate_from_reciprocalStep
    (gap := gap) (alpha := gamma / (8 * Xmax * Umax ^ 2 * Anorm ^ 2))
    halpha_pos hgap_pos hreciprocal_step n
  have hconst : 1 / (gamma / (8 * Xmax * Umax ^ 2 * Anorm ^ 2)) =
      8 * Xmax * Umax ^ 2 * Anorm ^ 2 / gamma := by
    field_simp [ne_of_gt hgamma_pos, ne_of_gt hden_pos]
  simpa [hconst] using hrate

/--
Quadratic-descent version of the paper dual-rate theorem.

This theorem internalizes the final algebraic step of the appendix proof:
from `Delta_k - Delta_{k+1} ≥ gamma/(8 Xmax Umax^2 ||A||^2) * Delta_k^2`,
the reciprocal argument gives the advertised `O(1/k)` rate.
-/
theorem dualRate_KL_paperConstant_from_quadraticGapDescent
    {gap : ℕ → ℝ} {gamma Xmax Umax Anorm : ℝ}
    (hgamma_pos : 0 < gamma)
    (hXmax_pos : 0 < Xmax)
    (hUmax_pos : 0 < Umax)
    (hAnorm_pos : 0 < Anorm)
    (hgap_nonneg : ∀ k : ℕ, 0 ≤ gap k)
    (hdescent :
      ∀ k : ℕ,
        (gamma / (8 * Xmax * Umax ^ 2 * Anorm ^ 2)) * gap k ^ 2 ≤
          gap k - gap (k + 1))
    (n : ℕ) :
    0 ≤ gap (n + 1) ∧
      gap (n + 1) ≤
        (8 * Xmax * Umax ^ 2 * Anorm ^ 2 / gamma) / (n + 1 : ℝ) := by
  have hden_pos : 0 < 8 * Xmax * Umax ^ 2 * Anorm ^ 2 := by positivity
  have halpha_pos : 0 < gamma / (8 * Xmax * Umax ^ 2 * Anorm ^ 2) :=
    div_pos hgamma_pos hden_pos
  have hstep_mono : ∀ k : ℕ, gap (k + 1) ≤ gap k := by
    intro k
    have hterm_nonneg : 0 ≤ (gamma / (8 * Xmax * Umax ^ 2 * Anorm ^ 2)) * gap k ^ 2 := by
      exact mul_nonneg (le_of_lt halpha_pos) (sq_nonneg (gap k))
    have hdiff_nonneg : 0 ≤ gap k - gap (k + 1) := hterm_nonneg.trans (hdescent k)
    exact sub_nonneg.mp hdiff_nonneg
  have hmono : Antitone gap := antitone_nat_of_succ_le hstep_mono
  refine ⟨hgap_nonneg (n + 1), ?_⟩
  by_cases hzero : gap (n + 1) = 0
  · have hconst_nonneg : 0 ≤ (8 * Xmax * Umax ^ 2 * Anorm ^ 2 / gamma) / (n + 1 : ℝ) := by
      positivity
    simpa [hzero] using hconst_nonneg
  · have hpos_n1 : 0 < gap (n + 1) := lt_of_le_of_ne (hgap_nonneg (n + 1)) (Ne.symm hzero)
    have hgap_pos_to : ∀ k : ℕ, k ≤ n + 1 → 0 < gap k := by
      intro k hk
      have hle : gap (n + 1) ≤ gap k := hmono hk
      exact lt_of_lt_of_le hpos_n1 hle
    have hrate := dualRate_from_reciprocalStep_upto
      (gap := gap) (alpha := gamma / (8 * Xmax * Umax ^ 2 * Anorm ^ 2))
      halpha_pos n hgap_pos_to
      (by
        intro k hk
        have hk_to : k ≤ n + 1 := Nat.le_trans hk (Nat.le_succ n)
        have hk1_to : k + 1 ≤ n + 1 := Nat.succ_le_succ hk
        exact reciprocalStep_of_quadraticDescent
          (a := gap k) (b := gap (k + 1))
          (alpha := gamma / (8 * Xmax * Umax ^ 2 * Anorm ^ 2))
          (hgap_pos_to k hk_to) (hgap_pos_to (k + 1) hk1_to)
          (hstep_mono k) (hdescent k))
    have hconst : 1 / (gamma / (8 * Xmax * Umax ^ 2 * Anorm ^ 2)) =
        8 * Xmax * Umax ^ 2 * Anorm ^ 2 / gamma := by
      field_simp [ne_of_gt hgamma_pos, ne_of_gt hden_pos]
    simpa [hconst] using hrate

/--
Algebraic bridge from the two paper proof ingredients to quadratic gap descent.

The appendix combines the gap-vs-residual estimate
`Delta_k <= 2 * Umax * residual_k` with the per-step ascent inequality
`gamma/(2*Xmax*||A||^2) * residual_k^2 <= Delta_k - Delta_{k+1}`.  This theorem
formalizes the scalar algebra that turns those two inequalities into
`gamma/(8*Xmax*Umax^2*||A||^2) * Delta_k^2 <= Delta_k - Delta_{k+1}`.
-/
theorem quadraticGapDescent_of_gapResidual_ascent
    {gap residual : ℕ → ℝ} {gamma Xmax Umax Anorm : ℝ}
    (hgamma_pos : 0 < gamma)
    (hXmax_pos : 0 < Xmax)
    (hUmax_pos : 0 < Umax)
    (hAnorm_pos : 0 < Anorm)
    (hgap_nonneg : ∀ k : ℕ, 0 ≤ gap k)
    (hgap_res : ∀ k : ℕ, gap k ≤ (2 * Umax) * residual k)
    (hascent : ∀ k : ℕ,
      (gamma / (2 * Xmax * Anorm ^ 2)) * residual k ^ 2 ≤ gap k - gap (k + 1))
    (k : ℕ) :
    (gamma / (8 * Xmax * Umax ^ 2 * Anorm ^ 2)) * gap k ^ 2 ≤
      gap k - gap (k + 1) := by
  have halpha_nonneg : 0 ≤ gamma / (8 * Xmax * Umax ^ 2 * Anorm ^ 2) := by
    positivity
  have hsq : gap k ^ 2 ≤ ((2 * Umax) * residual k) ^ 2 := by
    nlinarith [hgap_nonneg k, hUmax_pos, hgap_res k]
  have hscaled :
      (gamma / (8 * Xmax * Umax ^ 2 * Anorm ^ 2)) * gap k ^ 2 ≤
        (gamma / (8 * Xmax * Umax ^ 2 * Anorm ^ 2)) *
          ((2 * Umax) * residual k) ^ 2 :=
    mul_le_mul_of_nonneg_left hsq halpha_nonneg
  have hconst :
      (gamma / (8 * Xmax * Umax ^ 2 * Anorm ^ 2)) *
          ((2 * Umax) * residual k) ^ 2 =
        (gamma / (2 * Xmax * Anorm ^ 2)) * residual k ^ 2 := by
    field_simp [ne_of_gt hXmax_pos, ne_of_gt hUmax_pos, ne_of_gt hAnorm_pos]
    ring
  exact hscaled.trans (by simpa [hconst] using hascent k)

/--
Paper-constant rate theorem from the two Section-3 proof ingredients.

Compared with `dualRate_KL_paperConstant_from_quadraticGapDescent`, this endpoint no longer
takes the quadratic descent inequality as primitive: Lean derives it from the displayed
gap-vs-residual and per-step-ascent hypotheses, then applies the certified reciprocal-rate proof.
-/
theorem dualRate_KL_paperConstant_from_ascentGapResidual
    {gap residual : ℕ → ℝ} {gamma Xmax Umax Anorm : ℝ}
    (hgamma_pos : 0 < gamma)
    (hXmax_pos : 0 < Xmax)
    (hUmax_pos : 0 < Umax)
    (hAnorm_pos : 0 < Anorm)
    (hgap_nonneg : ∀ k : ℕ, 0 ≤ gap k)
    (hgap_res : ∀ k : ℕ, gap k ≤ (2 * Umax) * residual k)
    (hascent : ∀ k : ℕ,
      (gamma / (2 * Xmax * Anorm ^ 2)) * residual k ^ 2 ≤ gap k - gap (k + 1))
    (n : ℕ) :
    0 ≤ gap (n + 1) ∧
      gap (n + 1) ≤
        (8 * Xmax * Umax ^ 2 * Anorm ^ 2 / gamma) / (n + 1 : ℝ) := by
  exact dualRate_KL_paperConstant_from_quadraticGapDescent
    hgamma_pos hXmax_pos hUmax_pos hAnorm_pos hgap_nonneg
    (quadraticGapDescent_of_gapResidual_ascent
      hgamma_pos hXmax_pos hUmax_pos hAnorm_pos hgap_nonneg hgap_res hascent)
    n

/--
Paper-facing vocabulary-certificate version of Theorem `thm:kl-dual-rate`.

The two hypotheses are named statement-level certificates: positivity of the scalar constants and
the exact Appendix-A scalar ingredients.  Lean unfolds the certificates, derives the quadratic
descent inequality, and applies the certified reciprocal-growth rate theorem.
-/
theorem dualRate_KL_paperConstant_from_scalarCertificates
    {gap residual : ℕ → ℝ} {gamma Xmax Umax Anorm : ℝ}
    (hconst : PositiveKLRateConstants gamma Xmax Umax Anorm)
    (hingredients : KLRateScalarIngredients gap residual gamma Xmax Umax Anorm)
    (n : ℕ) :
    0 ≤ gap (n + 1) ∧
      gap (n + 1) ≤
        (8 * Xmax * Umax ^ 2 * Anorm ^ 2 / gamma) / (n + 1 : ℝ) := by
  exact
    dualRate_KL_paperConstant_from_ascentGapResidual
      hconst.gamma_pos hconst.xmax_pos hconst.umax_pos hconst.anorm_pos
      hingredients.gap_nonneg hingredients.gap_residual hingredients.per_step_ascent n

/--
Structured-certificate version of Theorem `thm:kl-dual-rate`.

The challenge-facing statement now exposes a single named certificate for the
scalar proof interface.  Lean unfolds that certificate and reuses the
paper-constant rate proof, so the remaining bridge is precisely the construction
of `KLDualRateCertificate` from the concrete cyclic KL block iterates.
-/
theorem dualRate_KL_paperConstant_from_rateCertificate
    {gap residual : ℕ → ℝ} {gamma Xmax Umax Anorm : ℝ}
    (hcert : KLDualRateCertificate gap residual gamma Xmax Umax Anorm)
    (n : ℕ) :
    0 ≤ gap (n + 1) ∧
      gap (n + 1) ≤
        (8 * Xmax * Umax ^ 2 * Anorm ^ 2 / gamma) / (n + 1 : ℝ) := by
  exact
    dualRate_KL_paperConstant_from_scalarCertificates
      hcert.constants hcert.scalar_ingredients n

/--
Ready-to-use iteration threshold extracted from the Section-3 master rate.

If the iteration count is large enough so that `C ≤ eps * (n + 1)`, then the master `O(1/k)`
bound immediately yields `gap n ≤ eps`.
-/
theorem dualRate_iterationThreshold_of_masterAbstractRate
    {gap : ℕ → ℝ} {C eps : ℝ}
    (hmaster : ∀ n : ℕ, gap n ≤ C / (n + 1 : ℝ))
    (n : ℕ)
    (hthreshold : C ≤ eps * (n + 1 : ℝ)) :
    gap n ≤ eps := by
  have hpos : 0 < (n + 1 : ℝ) := by
    exact_mod_cast Nat.succ_pos n
  have hbound : C / (n + 1 : ℝ) ≤ eps := by
    exact (div_le_iff₀ hpos).2 hthreshold
  exact (hmaster n).trans hbound

/--
Canonical closed-form iteration threshold in ratio form.

If `C / eps ≤ n + 1`, then the master rate already implies `gap n ≤ eps`.
-/
theorem dualRate_iterationThreshold_of_ratioBound
    {gap : ℕ → ℝ} {C eps : ℝ}
    (hmaster : ∀ n : ℕ, gap n ≤ C / (n + 1 : ℝ))
    (heps : 0 < eps)
    (n : ℕ)
    (hratio : C / eps ≤ (n + 1 : ℝ)) :
    gap n ≤ eps := by
  have hthreshold : C ≤ eps * (n + 1 : ℝ) := by
    have hthreshold' : C ≤ (n + 1 : ℝ) * eps := (div_le_iff₀ heps).1 hratio
    simpa [mul_comm] using hthreshold'
  exact dualRate_iterationThreshold_of_masterAbstractRate hmaster n hthreshold

/--
Ceiling-form wrapper for the iteration threshold.

This is the paper-readable version; the canonical proof route is the ratio bound above.
-/
theorem dualRate_iterationThreshold_of_closedFormCeil
    {gap : ℕ → ℝ} {C eps : ℝ}
    (hmaster : ∀ n : ℕ, gap n ≤ C / (n + 1 : ℝ))
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil (C / eps) ≤ n + 1) :
    gap n ≤ eps := by
  have hratio : C / eps ≤ (n + 1 : ℝ) := by
    simpa [Nat.cast_add, Nat.cast_one] using (Nat.le_of_ceil_le hn : C / eps ≤ (n + 1 : ℕ))
  exact dualRate_iterationThreshold_of_ratioBound hmaster heps n hratio

/--
Master-rate-to-threshold bridge in ratio form.

This packages the common pipeline
`dualRate_masterAbstractRateStatement -> dualRate_iterationThreshold_of_ratioBound`
with the explicit constant `alpha * B`.
-/
theorem dualRate_iterationThreshold_of_masterAbstractRate_ratioBound
    {phi gap residual : ℕ → ℝ}
    {alpha B eps : ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (heps : 0 < eps)
    (n : ℕ)
    (hratio : (alpha * B) / eps ≤ (n + 1 : ℝ)) :
    gap n ≤ eps := by
  have hmaster : ∀ m : ℕ, gap m ≤ (alpha * B) / (m + 1 : ℝ) := by
    intro m
    exact dualRate_masterAbstractRateStatement
      halpha hgap_res hres_ascent hphi_bound hmono_gap m
  exact dualRate_iterationThreshold_of_ratioBound hmaster heps n hratio

/--
Ratio-threshold bridge with the stricter index condition `C/eps ≤ n`.

This avoids recurring `+1` index bookkeeping in downstream threshold calls.
-/
theorem dualRate_iterationThreshold_of_ratioBound_of_le_index
    {gap : ℕ → ℝ} {C eps : ℝ}
    (hmaster : ∀ n : ℕ, gap n ≤ C / (n + 1 : ℝ))
    (heps : 0 < eps)
    (n : ℕ)
    (hratio : C / eps ≤ (n : ℝ)) :
    gap n ≤ eps := by
  have hratio' : C / eps ≤ (n + 1 : ℝ) := by
    calc
      C / eps ≤ (n : ℝ) := hratio
      _ ≤ (n + 1 : ℝ) := by linarith
  exact dualRate_iterationThreshold_of_ratioBound hmaster heps n hratio'

/--
Ratio-threshold bridge using an intermediate natural-number bound.

This is convenient when applications first produce an integer bound `N` with
`C / eps ≤ N` and then separately prove `N ≤ n + 1`.
-/
theorem dualRate_iterationThreshold_of_ratioBound_of_natBound
    {gap : ℕ → ℝ} {C eps : ℝ}
    (hmaster : ∀ n : ℕ, gap n ≤ C / (n + 1 : ℝ))
    (heps : 0 < eps)
    (n N : ℕ)
    (hratioN : C / eps ≤ (N : ℝ))
    (hN : N ≤ n + 1) :
    gap n ≤ eps := by
  have hratio : C / eps ≤ (n + 1 : ℝ) := by
    calc
      C / eps ≤ (N : ℝ) := hratioN
      _ ≤ (n + 1 : ℝ) := by exact_mod_cast hN
  exact dualRate_iterationThreshold_of_ratioBound hmaster heps n hratio

/--
Master-rate ratio-threshold bridge with the stricter index condition
`(alpha * B)/eps ≤ n`.
-/
theorem dualRate_iterationThreshold_of_masterAbstractRate_ratioBound_of_le_index
    {phi gap residual : ℕ → ℝ}
    {alpha B eps : ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (heps : 0 < eps)
    (n : ℕ)
    (hratio : (alpha * B) / eps ≤ (n : ℝ)) :
    gap n ≤ eps := by
  have hratio' : (alpha * B) / eps ≤ (n + 1 : ℝ) := by
    calc
      (alpha * B) / eps ≤ (n : ℝ) := hratio
      _ ≤ (n + 1 : ℝ) := by linarith
  exact dualRate_iterationThreshold_of_masterAbstractRate_ratioBound
    halpha hgap_res hres_ascent hphi_bound hmono_gap heps n hratio'

/--
Successor-index ratio-threshold bridge.

This is the `n+1` convenience form of
`dualRate_iterationThreshold_of_ratioBound_of_le_index`.
-/
theorem dualRate_iterationThreshold_of_ratioBound_succ
    {gap : ℕ → ℝ} {C eps : ℝ}
    (hmaster : ∀ n : ℕ, gap n ≤ C / (n + 1 : ℝ))
    (heps : 0 < eps)
    (n : ℕ)
    (hratio : C / eps ≤ (n + 1 : ℝ)) :
    gap (n + 1) ≤ eps := by
  have hratio' : C / eps ≤ ((n + 1 : ℕ) : ℝ) := by
    simpa [Nat.cast_add, Nat.cast_one] using hratio
  simpa [Nat.succ_eq_add_one] using
    dualRate_iterationThreshold_of_ratioBound_of_le_index
      hmaster heps (n + 1) hratio'

/--
Successor-index master-rate ratio-threshold bridge.

This is the `n+1` convenience form of
`dualRate_iterationThreshold_of_masterAbstractRate_ratioBound_of_le_index`.
-/
theorem dualRate_iterationThreshold_of_masterAbstractRate_ratioBound_succ
    {phi gap residual : ℕ → ℝ}
    {alpha B eps : ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (heps : 0 < eps)
    (n : ℕ)
    (hratio : (alpha * B) / eps ≤ (n + 1 : ℝ)) :
    gap (n + 1) ≤ eps := by
  have hratio' : (alpha * B) / eps ≤ ((n + 1 : ℕ) : ℝ) := by
    simpa [Nat.cast_add, Nat.cast_one] using hratio
  simpa [Nat.succ_eq_add_one] using
    dualRate_iterationThreshold_of_masterAbstractRate_ratioBound_of_le_index
      halpha hgap_res hres_ascent hphi_bound hmono_gap
      heps (n + 1) hratio'

/--
Master-rate-to-threshold bridge in ceiling form.

This is the closed-form stopping-rule version of
`dualRate_iterationThreshold_of_masterAbstractRate_ratioBound`.
-/
theorem dualRate_iterationThreshold_of_masterAbstractRate_closedFormCeil
    {phi gap residual : ℕ → ℝ}
    {alpha B eps : ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil ((alpha * B) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  have hmaster : ∀ m : ℕ, gap m ≤ (alpha * B) / (m + 1 : ℝ) := by
    intro m
    exact dualRate_masterAbstractRateStatement
      halpha hgap_res hres_ascent hphi_bound hmono_gap m
  exact dualRate_iterationThreshold_of_closedFormCeil hmaster heps n hn

/--
Closed-form threshold bridge with the stricter index condition `ceil(C/eps) ≤ n`.

This avoids the recurring `+ 1` bookkeeping in downstream callers.
-/
theorem dualRate_iterationThreshold_of_closedFormCeil_of_le_index
    {gap : ℕ → ℝ} {C eps : ℝ}
    (hmaster : ∀ n : ℕ, gap n ≤ C / (n + 1 : ℝ))
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil (C / eps) ≤ n) :
    gap n ≤ eps := by
  exact dualRate_iterationThreshold_of_closedFormCeil
    hmaster heps n (hn.trans (Nat.le_succ n))

/--
Master-rate threshold bridge with the stricter index condition
`ceil((alpha * B) / eps) ≤ n`.

This is the `masterAbstractRate` counterpart of
`dualRate_iterationThreshold_of_closedFormCeil_of_le_index`.
-/
theorem dualRate_iterationThreshold_of_masterAbstractRate_closedFormCeil_of_le_index
    {phi gap residual : ℕ → ℝ}
    {alpha B eps : ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil ((alpha * B) / eps) ≤ n) :
    gap n ≤ eps := by
  exact dualRate_iterationThreshold_of_masterAbstractRate_closedFormCeil
    halpha hgap_res hres_ascent hphi_bound hmono_gap heps n
    (hn.trans (Nat.le_succ n))

/--
Master-rate closed-form threshold bridge using an intermediate natural-number bound.

This variant is convenient when applications first establish
`ceil((alpha * B) / eps) ≤ N` and then separately prove `N ≤ n`.
-/
theorem dualRate_iterationThreshold_of_masterAbstractRate_closedFormCeil_of_natBound
    {phi gap residual : ℕ → ℝ}
    {alpha B eps : ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (heps : 0 < eps)
    (n N : ℕ)
    (hceilN : Nat.ceil ((alpha * B) / eps) ≤ N)
    (hN : N ≤ n) :
    gap n ≤ eps := by
  exact dualRate_iterationThreshold_of_masterAbstractRate_closedFormCeil_of_le_index
    halpha hgap_res hres_ascent hphi_bound hmono_gap heps n (hceilN.trans hN)

/--
Successor-index master-rate closed-form-threshold bridge.

This is the `n+1` convenience form of
`dualRate_iterationThreshold_of_masterAbstractRate_closedFormCeil_of_le_index`.
-/
theorem dualRate_iterationThreshold_of_masterAbstractRate_closedFormCeil_succ
    {phi gap residual : ℕ → ℝ}
    {alpha B eps : ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil ((alpha * B) / eps) ≤ n + 1) :
    gap (n + 1) ≤ eps := by
  simpa [Nat.succ_eq_add_one] using
    dualRate_iterationThreshold_of_masterAbstractRate_closedFormCeil_of_le_index
      halpha hgap_res hres_ascent hphi_bound hmono_gap
      heps (n + 1) hn

/--
Successor-index closed-form-threshold bridge.

This is the `n+1` convenience form of
`dualRate_iterationThreshold_of_closedFormCeil_of_le_index`.
-/
theorem dualRate_iterationThreshold_of_closedFormCeil_succ
    {gap : ℕ → ℝ} {C eps : ℝ}
    (hmaster : ∀ n : ℕ, gap n ≤ C / (n + 1 : ℝ))
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil (C / eps) ≤ n + 1) :
    gap (n + 1) ≤ eps := by
  simpa [Nat.succ_eq_add_one] using
    dualRate_iterationThreshold_of_closedFormCeil_of_le_index
      hmaster heps (n + 1) hn

/--
Convenience ceiling-index statement at `n = ceil((alpha * B)/eps)`.

This packages the common “use the canonical closed-form iterate count directly” step.
-/
theorem dualRate_iterationThreshold_of_masterAbstractRate_at_ceiled_index
    {phi gap residual : ℕ → ℝ}
    {alpha B eps : ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (heps : 0 < eps) :
    gap (Nat.ceil ((alpha * B) / eps)) ≤ eps := by
  exact dualRate_iterationThreshold_of_masterAbstractRate_closedFormCeil
    halpha hgap_res hres_ascent hphi_bound hmono_gap heps
    (Nat.ceil ((alpha * B) / eps)) (Nat.le_succ _)

/--
Scaled bridge theorem combining gap scaling, ascent, and the cumulative budget in explicit
`O(1/k)` form.

This is the version to use when the application naturally provides a multiplicative gap-to-residual
comparison and wants the final bound written directly for the scaled gap.
-/
theorem dualRate_O_one_over_k_of_scaledGap_ascent_budget
    {phi gap residual : ℕ → ℝ}
    {alpha B : ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (n : ℕ) :
    alpha * gap n ≤ alpha * B / (n + 1 : ℝ) := by
  let gap' : ℕ → ℝ := fun k => alpha * gap k
  let residual' : ℕ → ℝ := fun k => alpha * residual k
  let phi' : ℕ → ℝ := fun k => alpha * phi k
  have hgap' : ∀ k : ℕ, gap' k ≤ residual' k := by
    intro k
    simpa [gap', residual'] using
      dualGap_le_scaledResidual (gap := gap) (residual := residual) (c := alpha)
        hgap_res halpha k
  have hres' : ∀ k : ℕ, residual' k ≤ phi' (k + 1) - phi' k := by
    intro k
    have hk : alpha * residual k ≤ alpha * (phi (k + 1) - phi k) :=
      mul_le_mul_of_nonneg_left (hres_ascent k) halpha
    calc
      residual' k = alpha * residual k := by rfl
      _ ≤ alpha * (phi (k + 1) - phi k) := hk
      _ = phi' (k + 1) - phi' k := by
            unfold phi'
            ring
  have hphi' : ∀ n : ℕ, phi' (n + 1) - phi' 0 ≤ alpha * B := by
    intro n
    have hb : alpha * (phi (n + 1) - phi 0) ≤ alpha * B :=
      mul_le_mul_of_nonneg_left (hphi_bound n) halpha
    calc
      phi' (n + 1) - phi' 0 = alpha * (phi (n + 1) - phi 0) := by
        unfold phi'
        ring
      _ ≤ alpha * B := hb
  have hmono_gap' : Antitone gap' := by
    intro i j hij
    exact mul_le_mul_of_nonneg_left (hmono_gap hij) halpha
  have hrate' :=
    dualRate_masterAbstractRateStatement
      (gap := gap') (residual := residual') (phi := phi')
      (alpha := (1 : ℝ)) (B := alpha * B)
      (by simp)
      (by simpa using hgap') hres' hphi' hmono_gap' n
  simpa [gap', residual', phi'] using hrate'

/--
Bridge theorem from the Section-3 master rate statement to the final objective approximation
bound.  This is the canonical handoff into the bias/complexity layer.
-/
theorem regularizedApproximation_bound_from_masterAbstractRate
    {F0 FgammaStar bias C : ℝ}
    {Fgamma gap : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (hmaster : ∀ n : ℕ, gap n ≤ C / (n + 1 : ℝ))
    (n : ℕ) :
    |F0 - Fgamma n| ≤ bias + C / (n + 1 : ℝ) := by
  have hrate : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ C / (n + 1 : ℝ) := by
    intro m
    exact (hobj_gap m).trans (hmaster m)
  have hsplit : F0 - Fgamma n = (F0 - FgammaStar) + (FgammaStar - Fgamma n) := by ring
  calc
    |F0 - Fgamma n|
        = |(F0 - FgammaStar) + (FgammaStar - Fgamma n)| := by rw [hsplit]
    _ ≤ |F0 - FgammaStar| + |FgammaStar - Fgamma n| := abs_add_le _ _
    _ ≤ bias + C / (n + 1 : ℝ) := add_le_add hbias (hrate n)

/--
Bridge from the master iteration threshold to the regularized approximation error.

This is the bias/complexity-facing companion to the new threshold theorem.
-/
theorem regularizedApproximation_error_le_eps_of_iterationThreshold
    {FgammaStar C eps : ℝ}
    {Fgamma : ℕ → ℝ}
    (hmaster : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ C / (n + 1 : ℝ))
    (n : ℕ)
    (hthreshold : C ≤ eps * (n + 1 : ℝ)) :
    |FgammaStar - Fgamma n| ≤ eps :=
  dualRate_iterationThreshold_of_masterAbstractRate
    (gap := fun k => |FgammaStar - Fgamma k|)
    (C := C) (eps := eps)
    hmaster n hthreshold

/--
Closed-form bridge from a ceiling bound to the regularized approximation error.

This is the optimization-side companion to `dualRate_iterationThreshold_of_closedFormCeil`.
-/
theorem regularizedApproximation_error_le_eps_of_closedFormIterationThreshold
    {FgammaStar C eps : ℝ}
    {Fgamma : ℕ → ℝ}
    (hmaster : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ C / (n + 1 : ℝ))
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil (C / eps) ≤ n + 1) :
    |FgammaStar - Fgamma n| ≤ eps :=
  dualRate_iterationThreshold_of_closedFormCeil
    (gap := fun k => |FgammaStar - Fgamma k|)
    (C := C) (eps := eps)
    hmaster heps n hn

/--
Master-rate-to-regularized-error bridge with an explicit closed-form threshold.

This theorem removes the intermediate boilerplate of creating an optimization-gap
master bound before applying the regularized approximation stopping rule.
-/
theorem regularizedApproximation_error_le_eps_of_masterAbstractRate_closedFormCeil
    {phi gap residual : ℕ → ℝ}
    {alpha B eps FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil ((alpha * B) / eps) ≤ n + 1) :
    |FgammaStar - Fgamma n| ≤ eps := by
  have hmaster_gap : ∀ m : ℕ, gap m ≤ (alpha * B) / (m + 1 : ℝ) := by
    intro m
    exact dualRate_masterAbstractRateStatement
      halpha hgap_res hres_ascent hphi_bound hmono_gap m
  have hmaster_obj : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ (alpha * B) / (m + 1 : ℝ) := by
    intro m
    exact (hobj_gap m).trans (hmaster_gap m)
  exact regularizedApproximation_error_le_eps_of_closedFormIterationThreshold
    (FgammaStar := FgammaStar) (C := alpha * B) (eps := eps) (Fgamma := Fgamma)
    hmaster_obj heps n hn

/--
Master-rate-to-regularized-error bridge with a ratio-form threshold.

This is the ratio counterpart of
`regularizedApproximation_error_le_eps_of_masterAbstractRate_closedFormCeil`.
-/
theorem regularizedApproximation_error_le_eps_of_masterAbstractRate_ratioBound
    {phi gap residual : ℕ → ℝ}
    {alpha B eps FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (n : ℕ)
    (hratio : (alpha * B) / eps ≤ (n + 1 : ℝ)) :
    |FgammaStar - Fgamma n| ≤ eps := by
  have hmaster_gap : ∀ m : ℕ, gap m ≤ (alpha * B) / (m + 1 : ℝ) := by
    intro m
    exact dualRate_masterAbstractRateStatement
      halpha hgap_res hres_ascent hphi_bound hmono_gap m
  have hmaster_obj : ∀ m : ℕ, |FgammaStar - Fgamma m| ≤ (alpha * B) / (m + 1 : ℝ) := by
    intro m
    exact (hobj_gap m).trans (hmaster_gap m)
  exact dualRate_iterationThreshold_of_ratioBound
    (gap := fun k => |FgammaStar - Fgamma k|)
    (C := alpha * B) (eps := eps)
    hmaster_obj heps n hratio

/--
Ratio-threshold regularized-error bridge with stricter index condition
`(alpha * B)/eps ≤ n`.
-/
theorem regularizedApproximation_error_le_eps_of_masterAbstractRate_ratioBound_of_le_index
    {phi gap residual : ℕ → ℝ}
    {alpha B eps FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (n : ℕ)
    (hratio : (alpha * B) / eps ≤ (n : ℝ)) :
    |FgammaStar - Fgamma n| ≤ eps := by
  have hratio' : (alpha * B) / eps ≤ (n + 1 : ℝ) := by
    calc
      (alpha * B) / eps ≤ (n : ℝ) := hratio
      _ ≤ (n + 1 : ℝ) := by linarith
  exact regularizedApproximation_error_le_eps_of_masterAbstractRate_ratioBound
    halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap heps n hratio'

/--
Ratio-threshold regularized-error bridge using an intermediate natural-number bound.

This variant is convenient when a stopping rule is first certified as an integer
bound `N` satisfying `(alpha * B)/eps ≤ N`, then related to the iterate count by
`N ≤ n + 1`.
-/
theorem regularizedApproximation_error_le_eps_of_masterAbstractRate_ratioBound_of_natBound
    {phi gap residual : ℕ → ℝ}
    {alpha B eps FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (n N : ℕ)
    (hratioN : (alpha * B) / eps ≤ (N : ℝ))
    (hN : N ≤ n + 1) :
    |FgammaStar - Fgamma n| ≤ eps := by
  have hratio : (alpha * B) / eps ≤ (n + 1 : ℝ) := by
    calc
      (alpha * B) / eps ≤ (N : ℝ) := hratioN
      _ ≤ (n + 1 : ℝ) := by exact_mod_cast hN
  exact regularizedApproximation_error_le_eps_of_masterAbstractRate_ratioBound
    halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap heps n hratio

/--
Successor-index ratio-threshold bridge for regularized approximation error.

This is the `n+1` convenience form of
`regularizedApproximation_error_le_eps_of_masterAbstractRate_ratioBound_of_le_index`.
-/
theorem regularizedApproximation_error_le_eps_of_masterAbstractRate_ratioBound_succ
    {phi gap residual : ℕ → ℝ}
    {alpha B eps FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (n : ℕ)
    (hratio : (alpha * B) / eps ≤ (n + 1 : ℝ)) :
    |FgammaStar - Fgamma (n + 1)| ≤ eps := by
  have hratio' : (alpha * B) / eps ≤ ((n + 1 : ℕ) : ℝ) := by
    simpa [Nat.cast_add, Nat.cast_one] using hratio
  simpa [Nat.succ_eq_add_one] using
    regularizedApproximation_error_le_eps_of_masterAbstractRate_ratioBound_of_le_index
      halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap
      heps (n + 1) hratio'

/--
Stricter-index companion of
`regularizedApproximation_error_le_eps_of_masterAbstractRate_closedFormCeil`.
-/
theorem regularizedApproximation_error_le_eps_of_masterAbstractRate_closedFormCeil_of_le_index
    {phi gap residual : ℕ → ℝ}
    {alpha B eps FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil ((alpha * B) / eps) ≤ n) :
    |FgammaStar - Fgamma n| ≤ eps := by
  exact regularizedApproximation_error_le_eps_of_masterAbstractRate_closedFormCeil
    halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap heps n
    (hn.trans (Nat.le_succ n))

/--
Closed-form regularized-error bridge using an intermediate natural-number bound.

This packages the common two-step index reasoning:
`ceil((alpha * B) / eps) ≤ N` together with `N ≤ n`.
-/
theorem regularizedApproximation_error_le_eps_of_masterAbstractRate_closedFormCeil_of_natBound
    {phi gap residual : ℕ → ℝ}
    {alpha B eps FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (n N : ℕ)
    (hceilN : Nat.ceil ((alpha * B) / eps) ≤ N)
    (hN : N ≤ n) :
    |FgammaStar - Fgamma n| ≤ eps := by
  exact regularizedApproximation_error_le_eps_of_masterAbstractRate_closedFormCeil_of_le_index
    halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap
    heps n (hceilN.trans hN)

/--
Successor-index closed-form-threshold bridge for regularized approximation error.

This is the `n+1` convenience form of
`regularizedApproximation_error_le_eps_of_masterAbstractRate_closedFormCeil_of_le_index`.
-/
theorem regularizedApproximation_error_le_eps_of_masterAbstractRate_closedFormCeil_succ
    {phi gap residual : ℕ → ℝ}
    {alpha B eps FgammaStar : ℝ}
    {Fgamma : ℕ → ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil ((alpha * B) / eps) ≤ n + 1) :
    |FgammaStar - Fgamma (n + 1)| ≤ eps := by
  simpa [Nat.succ_eq_add_one] using
    regularizedApproximation_error_le_eps_of_masterAbstractRate_closedFormCeil_of_le_index
      halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap
      heps (n + 1) hn

/--
Canonical stopping-rule alias.

This is the paper-facing name for the threshold step: once the closed-form stopping rule is met,
the optimization error is at most `eps`.
-/
theorem regularizedApproximation_stoppingRule_of_closedFormIterationThreshold
    {FgammaStar C eps : ℝ}
    {Fgamma : ℕ → ℝ}
    (hmaster : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ C / (n + 1 : ℝ))
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil (C / eps) ≤ n + 1) :
    |FgammaStar - Fgamma n| ≤ eps :=
  regularizedApproximation_error_le_eps_of_closedFormIterationThreshold
    (FgammaStar := FgammaStar) (C := C) (eps := eps) (Fgamma := Fgamma)
    hmaster heps n hn

/--
Final epsilon guarantee from a stopping rule and a bias budget.

This is the closest abstract analogue of Theorem `thm:approx-linprog`: once the optimization
error has been forced below `eps` and the bias budget is compatible with the target, the total
error is below the target.
-/
theorem regularizedApproximation_finalEpsilon_of_stoppingRule
    {F0 FgammaStar bias eps target : ℝ}
    {Fgamma : ℕ → ℝ}
    (n : ℕ)
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hstop : |FgammaStar - Fgamma n| ≤ eps)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma n| ≤ target := by
  have hsplit : F0 - Fgamma n = (F0 - FgammaStar) + (FgammaStar - Fgamma n) := by ring
  have hsum : |F0 - Fgamma n| ≤ bias + eps := by
    calc
      |F0 - Fgamma n|
          = |(F0 - FgammaStar) + (FgammaStar - Fgamma n)| := by rw [hsplit]
      _ ≤ |F0 - FgammaStar| + |FgammaStar - Fgamma n| := abs_add_le _ _
      _ ≤ bias + eps := add_le_add hbias hstop
  exact hsum.trans hbudget

/--
Canonical abstract complexity recipe.

Dependency flow:
master rate → closed-form threshold → stopping rule → bias budget → final guarantee.
-/
theorem regularizedApproximation_complexity_of_closedFormIterationThreshold
    {F0 FgammaStar bias C eps target : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hmaster : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ C / (n + 1 : ℝ))
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil (C / eps) ≤ n + 1)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma n| ≤ target := by
  have hreg : |FgammaStar - Fgamma n| ≤ eps :=
    regularizedApproximation_stoppingRule_of_closedFormIterationThreshold
      (FgammaStar := FgammaStar) (C := C) (eps := eps) (Fgamma := Fgamma)
      hmaster heps n hn
  exact regularizedApproximation_finalEpsilon_of_stoppingRule
    (F0 := F0) (FgammaStar := FgammaStar) (bias := bias) (eps := eps) (target := target)
    (Fgamma := Fgamma) n hbias hreg hbudget

/--
Paper-style epsilon guarantee for Theorem `thm:approx-linprog`.

This endpoint exposes the constants that appear in the manuscript:

* the temperature choice `γ = ε / (2 * X₀ * log d)`;
* the Section-3 regularized dual-rate constant
  `8 * Xmax * Umax^2 * ||A||^2 / γ`;
* the paper's larger closed-form iteration budget
  `ceil(64 * Xmax * Umax^2 * ||A||^2 * maxMass * log d / ε^2)`;
* the usual `ε/2 + ε/2` split between regularization bias and optimization error.

The result is still stated at the scalar objective-error level.  Upstream hypotheses represent the
model-specific facts that the LP optimum gives the `ε/2` bias bound and that the generated KL
iterates satisfy the displayed dual-rate estimate.
-/
theorem regularizedApproximation_paperEpsilon_of_KLRate_closedFormIterationThreshold
    {F0 FgammaStar gamma Xmax Umax Anorm XmaxZero maxMass logd eps : ℝ}
    {Fgamma : ℕ → ℝ}
    (heps : 0 < eps)
    (hXmaxZero_logd_pos : 0 < XmaxZero * logd)
    (hAlog_nonneg : 0 ≤ Xmax * Umax ^ 2 * Anorm ^ 2 * logd)
    (hXmaxZero_nonneg : 0 ≤ XmaxZero)
    (hXmaxZero_le_maxMass : XmaxZero ≤ maxMass)
    (hgamma_choice : gamma = eps / (2 * XmaxZero * logd))
    (hbias : |F0 - FgammaStar| ≤ eps / 2)
    (hrate :
      ∀ n : ℕ,
        |FgammaStar - Fgamma n| ≤
          (8 * Xmax * Umax ^ 2 * Anorm ^ 2 / gamma) / (n + 1 : ℝ))
    (n : ℕ)
    (hn :
      Nat.ceil (64 * Xmax * Umax ^ 2 * Anorm ^ 2 * maxMass * logd / eps ^ 2) ≤
        n + 1) :
    |F0 - Fgamma n| ≤ eps := by
  let C := 16 * Xmax * Umax ^ 2 * Anorm ^ 2 * XmaxZero * logd / eps
  let paperN := 64 * Xmax * Umax ^ 2 * Anorm ^ 2 * maxMass * logd / eps ^ 2
  have hconst_eq :
      8 * Xmax * Umax ^ 2 * Anorm ^ 2 / gamma = C := by
    have heps_ne : eps ≠ 0 := ne_of_gt heps
    have hden_ne : 2 * XmaxZero * logd ≠ 0 := by nlinarith
    rw [hgamma_choice]
    dsimp [C]
    field_simp [heps_ne, hden_ne]
    ring_nf
  have hmaster : ∀ k : ℕ, |FgammaStar - Fgamma k| ≤ C / (k + 1 : ℝ) := by
    intro k
    simpa [hconst_eq] using hrate k
  have hratio : C / (eps / 2) ≤ paperN := by
    let A := Xmax * Umax ^ 2 * Anorm ^ 2 * logd
    have hA_nonneg : 0 ≤ A := by simpa [A] using hAlog_nonneg
    have hAmul : A * XmaxZero ≤ A * maxMass :=
      mul_le_mul_of_nonneg_left hXmaxZero_le_maxMass hA_nonneg
    have hmax_nonneg : 0 ≤ maxMass :=
      hXmaxZero_nonneg.trans hXmaxZero_le_maxMass
    have hAmax_nonneg : 0 ≤ A * maxMass := mul_nonneg hA_nonneg hmax_nonneg
    have h32 : 32 * (A * XmaxZero) ≤ 32 * (A * maxMass) := by
      exact mul_le_mul_of_nonneg_left hAmul (by norm_num)
    have h3264 : 32 * (A * maxMass) ≤ 64 * (A * maxMass) := by nlinarith
    have hmain : 32 * (A * XmaxZero) ≤ 64 * (A * maxMass) := h32.trans h3264
    have heps_ne : eps ≠ 0 := ne_of_gt heps
    dsimp [C, paperN]
    field_simp [heps_ne]
    ring_nf at hmain ⊢
    nlinarith
  have hceil : Nat.ceil (C / (eps / 2)) ≤ Nat.ceil paperN := Nat.ceil_le_ceil hratio
  have hnC : Nat.ceil (C / (eps / 2)) ≤ n + 1 := hceil.trans hn
  have heps_half : 0 < eps / 2 := by positivity
  have hbudget : eps / 2 + eps / 2 ≤ eps := by linarith
  exact regularizedApproximation_complexity_of_closedFormIterationThreshold
    (F0 := F0) (FgammaStar := FgammaStar) (bias := eps / 2) (C := C)
    (eps := eps / 2) (target := eps) (Fgamma := Fgamma)
    hbias hmaster heps_half n hnC hbudget

/--
Master-rate-to-final-target bridge with explicit ceiling threshold.

This packages the full non-application-specific pipeline:
Section-3 master rate hypotheses -> closed-form iterate threshold -> regularized error bound ->
bias handoff -> final target guarantee.
-/
theorem regularizedApproximation_complexity_of_masterAbstractRate_closedFormCeil
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias alpha B eps target : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil ((alpha * B) / eps) ≤ n + 1)
    (hbudget : bias + eps ≤ target) :
    |F0 - Fgamma n| ≤ target := by
  have hstop : |FgammaStar - Fgamma n| ≤ eps :=
    regularizedApproximation_error_le_eps_of_masterAbstractRate_closedFormCeil
      (phi := phi) (gap := gap) (residual := residual)
      (alpha := alpha) (B := B) (eps := eps)
      (FgammaStar := FgammaStar) (Fgamma := Fgamma)
      halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap heps n hn
  exact regularizedApproximation_finalEpsilon_of_stoppingRule
    (F0 := F0) (FgammaStar := FgammaStar) (bias := bias) (eps := eps) (target := target)
    (Fgamma := Fgamma) n hbias hstop hbudget

/--
Target-accuracy stopping rule in paper style.

Canonical flow:
master rate → ratio threshold → closed-form threshold → target guarantee.

If the iteration count is large enough to make the optimization budget at most `eps - bias`,
then the final objective error is at most `eps`.
-/
theorem regularizedApproximation_targetAccuracy_of_closedFormIterationThreshold
    {F0 FgammaStar bias C eps : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hmaster : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ C / (n + 1 : ℝ))
    (heps : bias < eps)
    (n : ℕ)
    (hn : Nat.ceil (C / (eps - bias)) ≤ n + 1) :
    |F0 - Fgamma n| ≤ eps := by
  have hopt : |FgammaStar - Fgamma n| ≤ eps - bias := by
    have hpos : 0 < eps - bias := sub_pos.mpr heps
    exact regularizedApproximation_error_le_eps_of_closedFormIterationThreshold
      (FgammaStar := FgammaStar) (C := C) (eps := eps - bias) (Fgamma := Fgamma)
      hmaster hpos n hn
  have hsplit : F0 - Fgamma n = (F0 - FgammaStar) + (FgammaStar - Fgamma n) := by ring
  have hsum : |F0 - Fgamma n| ≤ bias + (eps - bias) := by
    calc
      |F0 - Fgamma n|
          = |(F0 - FgammaStar) + (FgammaStar - Fgamma n)| := by rw [hsplit]
      _ ≤ |F0 - FgammaStar| + |FgammaStar - Fgamma n| := abs_add_le _ _
      _ ≤ bias + (eps - bias) := add_le_add hbias hopt
  have hbudget : bias + (eps - bias) ≤ eps := by linarith
  exact hsum.trans hbudget

/--
Paper-facing arithmetic core of Lemma `lem:bias-T`:
once the regularized gap is known nonnegative, an upper bound directly controls the absolute bias.
-/
theorem klBias_bound
    {F0 FgammaStar bias : ℝ}
    (hnonneg : 0 ≤ FgammaStar - F0)
    (hbias : FgammaStar - F0 ≤ bias) :
    |FgammaStar - F0| ≤ bias := by
  simpa [abs_of_nonneg hnonneg] using hbias

/--
Paper-facing arithmetic core of Appendix Lemma `app-lem:kl-bias`.

The theorem exposes the two displayed bias inequalities from the manuscript.  The deeper
optimization argument proving `Fγ⋆ - F0⋆ ≤ γ KL(x0⋆‖z)` and the entropy estimate for the
constant reference vector are passed as explicit hypotheses here, so Comparator can see exactly
which scalar consequences have been certified and which model-specific facts remain upstream.
-/
theorem klBias_regularizedGap_and_constantReference
    {F0 FgammaStar gamma klX0Z XmaxZero logd : ℝ}
    (hnonneg : 0 ≤ FgammaStar - F0)
    (hklBias : FgammaStar - F0 ≤ gamma * klX0Z)
    (hklConst : klX0Z ≤ XmaxZero * logd)
    (hgamma : 0 ≤ gamma) :
    0 ≤ FgammaStar - F0 ∧
      FgammaStar - F0 ≤ gamma * klX0Z ∧
        FgammaStar - F0 ≤ gamma * XmaxZero * logd := by
  refine ⟨hnonneg, hklBias, ?_⟩
  calc
    FgammaStar - F0 ≤ gamma * klX0Z := hklBias
    _ ≤ gamma * (XmaxZero * logd) := mul_le_mul_of_nonneg_left hklConst hgamma
    _ = gamma * XmaxZero * logd := by simp [mul_assoc]

/--
Coordinate-sum version of Appendix Lemma `app-lem:kl-bias`.

The previous paper-facing endpoint accepted the constant-reference entropy estimate as the single
scalar hypothesis `KL(x₀⋆‖z) <= XmaxZero*log d`.  This version exposes one more proof layer:
the KL quantity is represented as a finite sum of coordinate terms, each coordinate term is bounded
by an envelope term, and Lean verifies the summation step before applying the regularization bias
inequality.
-/
theorem klBias_regularizedGap_from_coordinateConstantReference
    {coord : Type*} [Fintype coord]
    {klTerm klEnvelope : coord → ℝ}
    {F0 FgammaStar gamma XmaxZero logd : ℝ}
    (hnonneg : 0 ≤ FgammaStar - F0)
    (hklBias : FgammaStar - F0 ≤ gamma * (∑ i : coord, klTerm i))
    (hterm : ∀ i : coord, klTerm i ≤ klEnvelope i)
    (henvelope : (∑ i : coord, klEnvelope i) ≤ XmaxZero * logd)
    (hgamma : 0 ≤ gamma) :
    0 ≤ FgammaStar - F0 ∧
      FgammaStar - F0 ≤ gamma * (∑ i : coord, klTerm i) ∧
        FgammaStar - F0 ≤ gamma * XmaxZero * logd := by
  have hklConst : (∑ i : coord, klTerm i) ≤ XmaxZero * logd := by
    exact (Finset.sum_le_sum fun i _hi => hterm i).trans henvelope
  refine ⟨hnonneg, hklBias, ?_⟩
  calc
    FgammaStar - F0 ≤ gamma * (∑ i : coord, klTerm i) := hklBias
    _ ≤ gamma * (XmaxZero * logd) := mul_le_mul_of_nonneg_left hklConst hgamma
    _ = gamma * XmaxZero * logd := by simp [mul_assoc]

/--
Feasible-minimizer version of Appendix Lemma `app-lem:kl-bias`.

This endpoint derives the optimization comparison
`Fγ⋆ - F0⋆ <= γ KL(x0⋆‖z)` from two explicit minimizer predicates: `x0` minimizes the
linear objective on the feasible set, while `xgamma` minimizes the entropic objective on the same
set.  It then reuses the coordinate-sum certificate for the constant-reference KL estimate.
-/
theorem klBias_regularizedGap_from_minimizers_coordinateConstantReference
    {coord : Type*} [Fintype coord]
    (C x0 xgamma : coord → ℝ)
    (Feasible : (coord → ℝ) → Prop)
    (KL : (coord → ℝ) → ℝ)
    {klTerm klEnvelope : coord → ℝ}
    {gamma XmaxZero logd : ℝ}
    (hgamma : 0 ≤ gamma)
    (hlinearMin : IsLinearMinimizer Feasible C x0)
    (hregMin : IsRegularizedMinimizer Feasible C gamma KL xgamma)
    (hKLnonneg : NonnegativeOn Feasible KL)
    (hKLsum : KL x0 = ∑ i : coord, klTerm i)
    (hterm : ∀ i : coord, klTerm i ≤ klEnvelope i)
    (henvelope : (∑ i : coord, klEnvelope i) ≤ XmaxZero * logd) :
    0 ≤ regularizedObjective C gamma KL xgamma - linearObjective C x0 ∧
      regularizedObjective C gamma KL xgamma - linearObjective C x0 ≤ gamma * KL x0 ∧
        regularizedObjective C gamma KL xgamma - linearObjective C x0 ≤
          gamma * (∑ i : coord, klTerm i) ∧
          regularizedObjective C gamma KL xgamma - linearObjective C x0 ≤
            gamma * XmaxZero * logd := by
  have hx0Feasible : Feasible x0 := hlinearMin.1
  have hxgammaFeasible : Feasible xgamma := hregMin.1
  have hlinear_le_xgamma : linearObjective C x0 ≤ linearObjective C xgamma :=
    hlinearMin.2 xgamma hxgammaFeasible
  have hKL_xgamma : 0 ≤ KL xgamma := hKLnonneg xgamma hxgammaFeasible
  have hreg_ge_linear :
      linearObjective C xgamma ≤ regularizedObjective C gamma KL xgamma := by
    dsimp [regularizedObjective]
    have hmul : 0 ≤ gamma * KL xgamma := mul_nonneg hgamma hKL_xgamma
    linarith
  have hnonneg :
      0 ≤ regularizedObjective C gamma KL xgamma - linearObjective C x0 :=
    sub_nonneg.mpr (hlinear_le_xgamma.trans hreg_ge_linear)
  have hreg_le_x0 :
      regularizedObjective C gamma KL xgamma ≤ regularizedObjective C gamma KL x0 :=
    hregMin.2 x0 hx0Feasible
  have hupperKL :
      regularizedObjective C gamma KL xgamma - linearObjective C x0 ≤ gamma * KL x0 := by
    dsimp [regularizedObjective] at hreg_le_x0 ⊢
    linarith
  have hupperSum :
      regularizedObjective C gamma KL xgamma - linearObjective C x0 ≤
        gamma * (∑ i : coord, klTerm i) := by
    simpa [hKLsum] using hupperKL
  have hklConst : (∑ i : coord, klTerm i) ≤ XmaxZero * logd := by
    exact (Finset.sum_le_sum fun i _hi => hterm i).trans henvelope
  refine ⟨hnonneg, hupperKL, hupperSum, ?_⟩
  calc
    regularizedObjective C gamma KL xgamma - linearObjective C x0
        ≤ gamma * (∑ i : coord, klTerm i) := hupperSum
    _ ≤ gamma * (XmaxZero * logd) := mul_le_mul_of_nonneg_left hklConst hgamma
    _ = gamma * XmaxZero * logd := by simp [mul_assoc]

/--
Mass-coordinate version of Appendix Lemma `app-lem:kl-bias`.

Compared with `klBias_regularizedGap_from_minimizers_coordinateConstantReference`, this endpoint
does not assume the summed constant-reference envelope
`sum_i klEnvelope_i <= XmaxZero * log d`.  It proves that finite summation step from coordinate
bounds `klTerm_i <= x0_i * logd`, the mass certificate `sum_i x0_i <= XmaxZero`, and
`0 <= logd`.
-/
theorem klBias_regularizedGap_from_minimizers_massCoordinateReference
    {coord : Type*} [Fintype coord]
    (C x0 xgamma : coord → ℝ)
    (Feasible : (coord → ℝ) → Prop)
    (KL : (coord → ℝ) → ℝ)
    {klTerm : coord → ℝ}
    {gamma XmaxZero logd : ℝ}
    (hgamma : 0 ≤ gamma)
    (hlinearMin : IsLinearMinimizer Feasible C x0)
    (hregMin : IsRegularizedMinimizer Feasible C gamma KL xgamma)
    (hKLnonneg : NonnegativeOn Feasible KL)
    (hKLsum : KL x0 = ∑ i : coord, klTerm i)
    (hterm : ∀ i : coord, klTerm i ≤ x0 i * logd)
    (hmass : (∑ i : coord, x0 i) ≤ XmaxZero)
    (hlogd : 0 ≤ logd) :
    0 ≤ regularizedObjective C gamma KL xgamma - linearObjective C x0 ∧
      regularizedObjective C gamma KL xgamma - linearObjective C x0 ≤ gamma * KL x0 ∧
        regularizedObjective C gamma KL xgamma - linearObjective C x0 ≤
          gamma * (∑ i : coord, klTerm i) ∧
          regularizedObjective C gamma KL xgamma - linearObjective C x0 ≤
            gamma * XmaxZero * logd := by
  have henvelope :
      (∑ i : coord, x0 i * logd) ≤ XmaxZero * logd := by
    have hmassLog : (∑ i : coord, x0 i) * logd ≤ XmaxZero * logd :=
      mul_le_mul_of_nonneg_right hmass hlogd
    simpa [Finset.sum_mul] using hmassLog
  exact klBias_regularizedGap_from_minimizers_coordinateConstantReference
    (C := C)
    (x0 := x0)
    (xgamma := xgamma)
    (Feasible := Feasible)
    (KL := KL)
    (hgamma := hgamma)
    (hlinearMin := hlinearMin)
    (hregMin := hregMin)
    (hKLnonneg := hKLnonneg)
    (hKLsum := hKLsum)
    (hterm := hterm)
    (henvelope := henvelope)

/--
Finite-sum KL version of Appendix Lemma `app-lem:kl-bias`.

This is the paper-facing refinement of
`klBias_regularizedGap_from_minimizers_massCoordinateReference`.  The KL functional is no longer
an arbitrary function equipped with a separate identity `KL x0 = sum_i klTerm_i`; it is
definitionally the finite coordinate sum.  Likewise the logarithmic dimension factor is
definitionally `log(card coord)`, whose nonnegativity is derived from `[Nonempty coord]`.
-/
theorem klBias_regularizedGap_from_minimizers_finiteSumKL_cardLogReference
    {coord : Type*} [Fintype coord] [Nonempty coord]
    (C x0 xgamma : coord → ℝ)
    (Feasible : (coord → ℝ) → Prop)
    (klTerm : (coord → ℝ) → coord → ℝ)
    {gamma XmaxZero : ℝ}
    (hgamma : 0 ≤ gamma)
    (hlinearMin : IsLinearMinimizer Feasible C x0)
    (hregMin :
      IsRegularizedMinimizer Feasible C gamma (coordinateSumKL klTerm) xgamma)
    (hKLnonneg : NonnegativeOn Feasible (coordinateSumKL klTerm))
    (hterm :
      ∀ i : coord, klTerm x0 i ≤ x0 i * Real.log (Fintype.card coord : ℝ))
    (hmass : (∑ i : coord, x0 i) ≤ XmaxZero) :
    0 ≤
        regularizedObjective C gamma (coordinateSumKL klTerm) xgamma -
          linearObjective C x0 ∧
      regularizedObjective C gamma (coordinateSumKL klTerm) xgamma -
          linearObjective C x0 ≤
        gamma * coordinateSumKL klTerm x0 ∧
        regularizedObjective C gamma (coordinateSumKL klTerm) xgamma -
            linearObjective C x0 ≤
          gamma * (∑ i : coord, klTerm x0 i) ∧
          regularizedObjective C gamma (coordinateSumKL klTerm) xgamma -
              linearObjective C x0 ≤
            gamma * XmaxZero * Real.log (Fintype.card coord : ℝ) := by
  have hcard_one : (1 : ℝ) ≤ (Fintype.card coord : ℝ) := by
    exact_mod_cast (Nat.succ_le_of_lt (Fintype.card_pos : 0 < Fintype.card coord))
  have hlog_card : 0 ≤ Real.log (Fintype.card coord : ℝ) :=
    Real.log_nonneg hcard_one
  exact
    klBias_regularizedGap_from_minimizers_massCoordinateReference
      (C := C)
      (x0 := x0)
      (xgamma := xgamma)
      (Feasible := Feasible)
      (KL := coordinateSumKL klTerm)
      (klTerm := fun i : coord => klTerm x0 i)
      (gamma := gamma)
      (XmaxZero := XmaxZero)
      (logd := Real.log (Fintype.card coord : ℝ))
      hgamma hlinearMin hregMin hKLnonneg (by rfl) hterm hmass hlog_card

/--
Bias-envelope refinement of Theorem `thm:approx-linprog`.

This endpoint removes the opaque input `|F₀⋆ - Fγ⋆| ≤ ε/2` from
`regularizedApproximation_paperEpsilon_of_KLRate_closedFormIterationThreshold`.
Instead it takes the paper's bias envelope
`0 ≤ Fγ⋆ - F₀⋆ ≤ γ * X₀ * log d` and derives the `ε/2` budget internally from the
displayed temperature choice `γ = ε/(2*X₀*log d)`.
-/
theorem regularizedApproximation_paperEpsilon_of_KLRate_biasEnvelope_closedFormIterationThreshold
    {F0 FgammaStar gamma Xmax Umax Anorm XmaxZero maxMass logd eps : ℝ}
    {Fgamma : ℕ → ℝ}
    (heps : 0 < eps)
    (hXmaxZero_logd_pos : 0 < XmaxZero * logd)
    (hAlog_nonneg : 0 ≤ Xmax * Umax ^ 2 * Anorm ^ 2 * logd)
    (hXmaxZero_nonneg : 0 ≤ XmaxZero)
    (hXmaxZero_le_maxMass : XmaxZero ≤ maxMass)
    (hgamma_choice : gamma = eps / (2 * XmaxZero * logd))
    (hbias_nonneg : 0 ≤ FgammaStar - F0)
    (hbias_envelope : FgammaStar - F0 ≤ gamma * XmaxZero * logd)
    (hrate :
      ∀ n : ℕ,
        |FgammaStar - Fgamma n| ≤
          (8 * Xmax * Umax ^ 2 * Anorm ^ 2 / gamma) / (n + 1 : ℝ))
    (n : ℕ)
    (hn :
      Nat.ceil (64 * Xmax * Umax ^ 2 * Anorm ^ 2 * maxMass * logd / eps ^ 2) ≤
        n + 1) :
    |F0 - Fgamma n| ≤ eps := by
  have hbias_abs_raw : |FgammaStar - F0| ≤ gamma * XmaxZero * logd :=
    klBias_bound hbias_nonneg hbias_envelope
  have hbias_budget : gamma * XmaxZero * logd = eps / 2 := by
    have hX_ne : XmaxZero ≠ 0 := by
      intro h
      rw [h] at hXmaxZero_logd_pos
      norm_num at hXmaxZero_logd_pos
    have hlog_ne : logd ≠ 0 := by
      intro h
      rw [h] at hXmaxZero_logd_pos
      norm_num at hXmaxZero_logd_pos
    rw [hgamma_choice]
    field_simp [hX_ne, hlog_ne]
  have hbias_abs : |F0 - FgammaStar| ≤ eps / 2 := by
    rw [abs_sub_comm]
    simpa [hbias_budget] using hbias_abs_raw
  exact
    regularizedApproximation_paperEpsilon_of_KLRate_closedFormIterationThreshold
      (F0 := F0) (FgammaStar := FgammaStar) (gamma := gamma)
      (Xmax := Xmax) (Umax := Umax) (Anorm := Anorm)
      (XmaxZero := XmaxZero) (maxMass := maxMass) (logd := logd)
      (eps := eps) (Fgamma := Fgamma)
      heps hXmaxZero_logd_pos hAlog_nonneg hXmaxZero_nonneg hXmaxZero_le_maxMass
      hgamma_choice hbias_abs hrate n hn

/--
Finite-minimizer version of Theorem `thm:approx-linprog`.

This is the strongest current paper-facing endpoint for the approximation theorem.  It derives
the regularization-bias half of the proof from the formal Appendix-B KL-bias theorem:
`x0` is a finite linear minimizer, `xgamma` is a finite regularized minimizer, the KL functional is
the finite coordinate sum, and the constant-reference entropy envelope is summed in Lean.  The
remaining input is the Section-3 regularized dual-rate estimate for the generated iterates.
-/
theorem regularizedApproximation_paperEpsilon_of_KLRate_finiteBias_closedFormIterationThreshold
    {coord : Type*} [Fintype coord] [Nonempty coord]
    (C x0 xgamma : coord → ℝ)
    (Feasible : (coord → ℝ) → Prop)
    (klTerm : (coord → ℝ) → coord → ℝ)
    {gamma Xmax Umax Anorm XmaxZero maxMass eps : ℝ}
    {Fgamma : ℕ → ℝ}
    (heps : 0 < eps)
    (hXmaxZero_logcard_pos : 0 < XmaxZero * Real.log (Fintype.card coord : ℝ))
    (hAlog_nonneg :
      0 ≤ Xmax * Umax ^ 2 * Anorm ^ 2 * Real.log (Fintype.card coord : ℝ))
    (hXmaxZero_nonneg : 0 ≤ XmaxZero)
    (hXmaxZero_le_maxMass : XmaxZero ≤ maxMass)
    (hgamma_choice :
      gamma = eps / (2 * XmaxZero * Real.log (Fintype.card coord : ℝ)))
    (hbiasCert :
      FiniteKLBiasApproximationCertificate C x0 xgamma Feasible klTerm gamma XmaxZero)
    (hrate :
      ∀ n : ℕ,
        |regularizedObjective C gamma (coordinateSumKL klTerm) xgamma - Fgamma n| ≤
          (8 * Xmax * Umax ^ 2 * Anorm ^ 2 / gamma) / (n + 1 : ℝ))
    (n : ℕ)
    (hn :
      Nat.ceil
          (64 * Xmax * Umax ^ 2 * Anorm ^ 2 * maxMass *
            Real.log (Fintype.card coord : ℝ) / eps ^ 2) ≤
        n + 1) :
    |linearObjective C x0 - Fgamma n| ≤ eps := by
  have hgamma_nonneg : 0 ≤ gamma := by
    have hden_pos : 0 < 2 * XmaxZero * Real.log (Fintype.card coord : ℝ) := by
      nlinarith
    rw [hgamma_choice]
    positivity
  have hbias :=
    klBias_regularizedGap_from_minimizers_finiteSumKL_cardLogReference
      (C := C) (x0 := x0) (xgamma := xgamma)
      (Feasible := Feasible) (klTerm := klTerm)
      (gamma := gamma) (XmaxZero := XmaxZero)
      hgamma_nonneg hbiasCert.linear_min hbiasCert.regularized_min hbiasCert.kl_nonneg
      hbiasCert.coordinate_envelope hbiasCert.mass_bound
  exact
    regularizedApproximation_paperEpsilon_of_KLRate_biasEnvelope_closedFormIterationThreshold
      (F0 := linearObjective C x0)
      (FgammaStar := regularizedObjective C gamma (coordinateSumKL klTerm) xgamma)
      (gamma := gamma) (Xmax := Xmax) (Umax := Umax) (Anorm := Anorm)
      (XmaxZero := XmaxZero) (maxMass := maxMass)
      (logd := Real.log (Fintype.card coord : ℝ))
      (eps := eps) (Fgamma := Fgamma)
      heps hXmaxZero_logcard_pos hAlog_nonneg hXmaxZero_nonneg hXmaxZero_le_maxMass
      hgamma_choice hbias.1 hbias.2.2.2 hrate n hn

/--
Theorem `thm:approx-linprog` with the Section-3 rate hypothesis derived internally.

Compared with
`regularizedApproximation_paperEpsilon_of_KLRate_finiteBias_closedFormIterationThreshold`, this
endpoint no longer assumes the regularized optimization estimate as a black-box bound.  It takes a
gap sequence linked to the displayed dual values and the two scalar proof ingredients of
Theorem `thm:kl-dual-rate`; Lean applies
`dualRate_KL_paperConstant_from_ascentGapResidual` to recover the required rate and then combines
it with the finite KL-bias certificate.
-/
theorem regularizedApproximation_paperEpsilon_of_rateIngredients_finiteBias_closedFormThreshold
    {coord : Type*} [Fintype coord] [Nonempty coord]
    (C x0 xgamma : coord → ℝ)
    (Feasible : (coord → ℝ) → Prop)
    (klTerm : (coord → ℝ) → coord → ℝ)
    {gamma Xmax Umax Anorm XmaxZero maxMass eps : ℝ}
    {Fgamma gap residual : ℕ → ℝ}
    (heps : 0 < eps)
    (hXmax_pos : 0 < Xmax)
    (hUmax_pos : 0 < Umax)
    (hAnorm_pos : 0 < Anorm)
    (hXmaxZero_logcard_pos : 0 < XmaxZero * Real.log (Fintype.card coord : ℝ))
    (hXmaxZero_le_maxMass : XmaxZero ≤ maxMass)
    (hgamma_choice :
      gamma = eps / (2 * XmaxZero * Real.log (Fintype.card coord : ℝ)))
    (hbiasCert :
      FiniteKLBiasApproximationCertificate C x0 xgamma Feasible klTerm gamma XmaxZero)
    (hgap_eval :
      ∀ n : ℕ,
        gap (n + 1) =
          regularizedObjective C gamma (coordinateSumKL klTerm) xgamma - Fgamma n)
    (hgap_nonneg : ∀ k : ℕ, 0 ≤ gap k)
    (hgap_res : ∀ k : ℕ, gap k ≤ (2 * Umax) * residual k)
    (hascent :
      ∀ k : ℕ,
        (gamma / (2 * Xmax * Anorm ^ 2)) * residual k ^ 2 ≤ gap k - gap (k + 1))
    (n : ℕ)
    (hn :
      Nat.ceil
          (64 * Xmax * Umax ^ 2 * Anorm ^ 2 * maxMass *
            Real.log (Fintype.card coord : ℝ) / eps ^ 2) ≤
        n + 1) :
    |linearObjective C x0 - Fgamma n| ≤ eps := by
  have hlog_nonneg : 0 ≤ Real.log (Fintype.card coord : ℝ) := by
    have hcard_one : (1 : ℝ) ≤ (Fintype.card coord : ℝ) := by
      exact_mod_cast (Nat.succ_le_of_lt (Fintype.card_pos : 0 < Fintype.card coord))
    exact Real.log_nonneg hcard_one
  have hXmaxZero_nonneg : 0 ≤ XmaxZero := by
    by_contra hnot
    have hXmaxZero_neg : XmaxZero < 0 := lt_of_not_ge hnot
    have hprod_nonpos :
        XmaxZero * Real.log (Fintype.card coord : ℝ) ≤ 0 :=
      mul_nonpos_of_nonpos_of_nonneg (le_of_lt hXmaxZero_neg) hlog_nonneg
    linarith
  have hAlog_nonneg :
      0 ≤ Xmax * Umax ^ 2 * Anorm ^ 2 * Real.log (Fintype.card coord : ℝ) := by
    positivity
  have hden_pos : 0 < 2 * XmaxZero * Real.log (Fintype.card coord : ℝ) := by
    nlinarith
  have hgamma_pos : 0 < gamma := by
    rw [hgamma_choice]
    positivity
  have hrate_from_gap :
      ∀ m : ℕ,
        0 ≤ gap (m + 1) ∧
          gap (m + 1) ≤
            (8 * Xmax * Umax ^ 2 * Anorm ^ 2 / gamma) / (m + 1 : ℝ) :=
    dualRate_KL_paperConstant_from_ascentGapResidual
      (gamma := gamma) (Xmax := Xmax) (Umax := Umax) (Anorm := Anorm)
      hgamma_pos hXmax_pos hUmax_pos hAnorm_pos hgap_nonneg hgap_res hascent
  have hrate :
      ∀ m : ℕ,
        |regularizedObjective C gamma (coordinateSumKL klTerm) xgamma - Fgamma m| ≤
          (8 * Xmax * Umax ^ 2 * Anorm ^ 2 / gamma) / (m + 1 : ℝ) := by
    intro m
    have hm := hrate_from_gap m
    have hdiff_nonneg :
        0 ≤ regularizedObjective C gamma (coordinateSumKL klTerm) xgamma - Fgamma m := by
      simpa [hgap_eval m] using hm.1
    have habs :
        |regularizedObjective C gamma (coordinateSumKL klTerm) xgamma - Fgamma m| =
          gap (m + 1) := by
      rw [abs_of_nonneg hdiff_nonneg, hgap_eval m]
    simpa [habs] using hm.2
  exact
    regularizedApproximation_paperEpsilon_of_KLRate_finiteBias_closedFormIterationThreshold
      (C := C) (x0 := x0) (xgamma := xgamma)
      (Feasible := Feasible) (klTerm := klTerm)
      (gamma := gamma) (Xmax := Xmax) (Umax := Umax) (Anorm := Anorm)
      (XmaxZero := XmaxZero) (maxMass := maxMass) (eps := eps)
      (Fgamma := Fgamma)
      heps hXmaxZero_logcard_pos hAlog_nonneg hXmaxZero_nonneg hXmaxZero_le_maxMass
      hgamma_choice hbiasCert hrate n hn

/--
Paper-facing certificate version of Theorem `thm:approx-linprog`.

The long list of scalar, finite-bias, gap-evaluation, and rate hypotheses is
exposed as the named proof-free certificate `ApproxLinprogCertificate`.  Lean
unfolds this certificate, derives the Section-3 KL rate from the bundled
`KLDualRateCertificate`, derives the regularization-bias half-budget from the
finite KL-bias certificate, and applies the closed-form iteration threshold
theorem.
-/
theorem regularizedApproximation_paperEpsilon_of_certificate_closedFormThreshold
    {coord : Type*} [Fintype coord] [Nonempty coord]
    (C x0 xgamma : coord → ℝ)
    (Feasible : (coord → ℝ) → Prop)
    (klTerm : (coord → ℝ) → coord → ℝ)
    {gamma Xmax Umax Anorm XmaxZero maxMass eps : ℝ}
    {Fgamma gap residual : ℕ → ℝ}
    (hcert :
      ApproxLinprogCertificate C x0 xgamma Feasible klTerm
        gamma Xmax Umax Anorm XmaxZero maxMass eps Fgamma gap residual)
    (n : ℕ)
    (hn :
      Nat.ceil
          (64 * Xmax * Umax ^ 2 * Anorm ^ 2 * maxMass *
            Real.log (Fintype.card coord : ℝ) / eps ^ 2) ≤
        n + 1) :
    |linearObjective C x0 - Fgamma n| ≤ eps := by
  exact
    regularizedApproximation_paperEpsilon_of_rateIngredients_finiteBias_closedFormThreshold
      (C := C)
      (x0 := x0)
      (xgamma := xgamma)
      (Feasible := Feasible)
      (klTerm := klTerm)
      (gamma := gamma)
      (Xmax := Xmax)
      (Umax := Umax)
      (Anorm := Anorm)
      (XmaxZero := XmaxZero)
      (maxMass := maxMass)
      (eps := eps)
      (Fgamma := Fgamma)
      (gap := gap)
      (residual := residual)
      hcert.eps_pos
      hcert.rate_certificate.constants.xmax_pos
      hcert.rate_certificate.constants.umax_pos
      hcert.rate_certificate.constants.anorm_pos
      hcert.xmaxZero_logcard_pos
      hcert.xmaxZero_le_maxMass
      hcert.gamma_choice
      hcert.bias
      hcert.gap_eval
      hcert.rate_certificate.scalar_ingredients.gap_nonneg
      hcert.rate_certificate.scalar_ingredients.gap_residual
      hcert.rate_certificate.scalar_ingredients.per_step_ascent
      n hn

/--
Combine bias and regularized optimization error into a total unregularized error bound.

This is the abstract decomposition used before inserting explicit constants and a rate.
-/
theorem regularizedApproximation_bound_of_bias_and_dualRate
    {F0 FgammaStar bias C : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ C / (n + 1 : ℝ))
    (n : ℕ) :
    |F0 - Fgamma n| ≤ bias + C / (n + 1 : ℝ) := by
  have hsplit : F0 - Fgamma n = (F0 - FgammaStar) + (FgammaStar - Fgamma n) := by ring
  calc
    |F0 - Fgamma n|
        = |(F0 - FgammaStar) + (FgammaStar - Fgamma n)| := by rw [hsplit]
    _ ≤ |F0 - FgammaStar| + |FgammaStar - Fgamma n| := abs_add_le _ _
    _ ≤ bias + C / (n + 1 : ℝ) := add_le_add hbias (hrate n)

/--
Paper-facing abstract complexity step:
if the sum of bias and optimization budget is at most `eps`, then the total error is at most `eps`.
-/
theorem regularizedApproximation_complexity
    {F0 FgammaStar bias C eps : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ C / (n + 1 : ℝ))
    (n : ℕ)
    (hbudget : bias + C / (n + 1 : ℝ) ≤ eps) :
    |F0 - Fgamma n| ≤ eps :=
  -- Legacy direct-budget wrapper; the canonical paper route is the theorem above.
  (regularizedApproximation_bound_from_masterAbstractRate
    (Fgamma := Fgamma)
    (gap := fun m => C / (m + 1 : ℝ))
    hbias
    hobj_gap
    (by intro m; exact le_rfl)
    n).trans hbudget

/--
First-iterate specialization of the O(1/k) rate bound.

If `gap n ≤ U / (n + 1)` for all n and U ≥ 0, then at n = 0 we have `gap 0 ≤ U`.
-/
theorem dualRate_first_iterate_bound
    {gap : ℕ → ℝ} {U : ℝ}
    (hrate : ∀ n : ℕ, gap n ≤ U / (n + 1 : ℝ))
    (hU : 0 ≤ U) :
    gap 0 ≤ U := by
  have h : gap 0 ≤ U := by
    simpa using hrate 0
  exact (And.left ⟨h, hU⟩)

/--
Halving-iterate specialization of the O(1/k) rate bound.

At iterate n = 1, the bound U / (1 + 1) = U / 2 gives `gap 1 ≤ U / 2`.
-/
theorem dualRate_halving_iterate
    {gap : ℕ → ℝ} {U : ℝ}
    (hrate : ∀ n : ℕ, gap n ≤ U / (n + 1 : ℝ))
    (hU : 0 ≤ U) :
    gap 1 ≤ U / 2 := by
  have h := hrate 1
  norm_num at h ⊢
  have hhalf_nonneg : 0 ≤ U / 2 := by
    nlinarith [hU]
  linarith [h, hhalf_nonneg]

/--
Antitone monotonicity bound: if `gap` is antitone, then at any iterate n, `gap n ≤ gap 0`.

This is trivially true from antitonicity, but is useful as a paper-facing interface theorem.
-/
theorem dualRate_bound_of_antitone_and_rate
    {gap : ℕ → ℝ} {U : ℝ}
    (hmono : Antitone gap)
    (hrate : ∀ n : ℕ, gap n ≤ U / (n + 1 : ℝ))
    (n : ℕ) :
    gap n ≤ gap 0 := by
  exact (And.left ⟨hmono (Nat.zero_le n), hrate n⟩)

/--
The O(1/k) rate constant can be extracted from the master ascent theorem.

If the Section-3 master conditions hold with parameters alpha, B, then the
O(1/k) rate constant is exactly `alpha * B`.
-/
theorem dualRate_constant_from_master
    {phi gap residual : ℕ → ℝ} {alpha B : ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap) :
    ∀ n : ℕ, gap n ≤ (alpha * B) / (n + 1 : ℝ) :=
  fun n => dualRate_masterAbstractRateStatement halpha hgap_res hres_ascent hphi_bound hmono_gap n

/--
Explicit iteration count lower bound: to achieve `gap n ≤ eps`, it suffices to have
`n + 1 ≥ alpha * B / eps`.

This is the concrete stopping criterion derivable from the O(1/k) rate.
-/
theorem dualRate_iterationCount_suffices
    {gap : ℕ → ℝ} {C eps : ℝ}
    (hmaster : ∀ n : ℕ, gap n ≤ C / (n + 1 : ℝ))
    (heps : 0 < eps)
    (hC : 0 ≤ C)
    (n : ℕ)
    (hn : C / eps ≤ (n + 1 : ℝ)) :
    gap n ≤ eps := by
  exact (And.left ⟨dualRate_iterationThreshold_of_ratioBound hmaster heps n hn, hC⟩)

/--
Abstract regularized approximation pipeline.

This packages the entire Section-3+bias pipeline: from abstract rate to final
`|F0 - Fgamma n| ≤ target` under minimal hypotheses.
-/
theorem regularizedApproximation_pipeline
    {F0 FgammaStar bias target C : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (hrate : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ C / (n + 1 : ℝ))
    (heps : 0 < target - bias)
    (n : ℕ)
    (hn : C / (target - bias) ≤ (n + 1 : ℝ)) :
    |F0 - Fgamma n| ≤ target := by
  have hopt : |FgammaStar - Fgamma n| ≤ target - bias :=
    dualRate_iterationThreshold_of_ratioBound hrate (by linarith) n hn
  have hsplit : F0 - Fgamma n = (F0 - FgammaStar) + (FgammaStar - Fgamma n) := by ring
  calc |F0 - Fgamma n|
      = |(F0 - FgammaStar) + (FgammaStar - Fgamma n)| := by rw [hsplit]
    _ ≤ |F0 - FgammaStar| + |FgammaStar - Fgamma n| := abs_add_le _ _
    _ ≤ bias + (target - bias) := add_le_add hbias hopt
    _ = target := by ring

/--
Master-rate-to-final-target pipeline with closed-form ceiling threshold.

This is the ergonomic one-shot route where the optimization tolerance is chosen as
`target - bias` and discharged via the master Section-3 hypotheses.
-/
theorem regularizedApproximation_pipeline_of_masterAbstractRate_closedFormCeil
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias target alpha B : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (hmargin : bias < target)
    (n : ℕ)
    (hn : Nat.ceil ((alpha * B) / (target - bias)) ≤ n + 1) :
    |F0 - Fgamma n| ≤ target := by
  have hpos : 0 < target - bias := sub_pos.mpr hmargin
  have hstop : |FgammaStar - Fgamma n| ≤ target - bias :=
    regularizedApproximation_error_le_eps_of_masterAbstractRate_closedFormCeil
      (phi := phi) (gap := gap) (residual := residual)
      (alpha := alpha) (B := B) (eps := target - bias)
      (FgammaStar := FgammaStar) (Fgamma := Fgamma)
      halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap hpos n hn
  have hbudget : bias + (target - bias) ≤ target := by linarith
  exact regularizedApproximation_finalEpsilon_of_stoppingRule
    (F0 := F0) (FgammaStar := FgammaStar)
    (bias := bias) (eps := target - bias) (target := target)
    (Fgamma := Fgamma) n hbias hstop hbudget

/--
Master-rate-to-final-target pipeline with ratio-form threshold.

This is the ratio counterpart of
`regularizedApproximation_pipeline_of_masterAbstractRate_closedFormCeil`.
-/
theorem regularizedApproximation_pipeline_of_masterAbstractRate_ratioBound
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias target alpha B : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (hmargin : bias < target)
    (n : ℕ)
    (hratio : (alpha * B) / (target - bias) ≤ (n + 1 : ℝ)) :
    |F0 - Fgamma n| ≤ target := by
  have hpos : 0 < target - bias := sub_pos.mpr hmargin
  have hstop : |FgammaStar - Fgamma n| ≤ target - bias :=
    regularizedApproximation_error_le_eps_of_masterAbstractRate_ratioBound
      (phi := phi) (gap := gap) (residual := residual)
      (alpha := alpha) (B := B) (eps := target - bias)
      (FgammaStar := FgammaStar) (Fgamma := Fgamma)
      halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap hpos n hratio
  have hbudget : bias + (target - bias) ≤ target := by linarith
  exact regularizedApproximation_finalEpsilon_of_stoppingRule
    (F0 := F0) (FgammaStar := FgammaStar)
    (bias := bias) (eps := target - bias) (target := target)
    (Fgamma := Fgamma) n hbias hstop hbudget

/--
Ratio-threshold final-pipeline bridge with stricter index condition
`(alpha * B)/(target - bias) ≤ n`.
-/
theorem regularizedApproximation_pipeline_of_masterAbstractRate_ratioBound_of_le_index
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias target alpha B : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (hmargin : bias < target)
    (n : ℕ)
    (hratio : (alpha * B) / (target - bias) ≤ (n : ℝ)) :
    |F0 - Fgamma n| ≤ target := by
  have hratio' : (alpha * B) / (target - bias) ≤ (n + 1 : ℝ) := by
    calc
      (alpha * B) / (target - bias) ≤ (n : ℝ) := hratio
      _ ≤ (n + 1 : ℝ) := by linarith
  exact regularizedApproximation_pipeline_of_masterAbstractRate_ratioBound
    hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap hmargin n hratio'

/--
Successor-index ratio-threshold bridge for the final-target pipeline.

This is the `n+1` convenience form of
`regularizedApproximation_pipeline_of_masterAbstractRate_ratioBound_of_le_index`.
-/
theorem regularizedApproximation_pipeline_of_masterAbstractRate_ratioBound_succ
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias target alpha B : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (hmargin : bias < target)
    (n : ℕ)
    (hratio : (alpha * B) / (target - bias) ≤ (n + 1 : ℝ)) :
    |F0 - Fgamma (n + 1)| ≤ target := by
  have hratio' : (alpha * B) / (target - bias) ≤ ((n + 1 : ℕ) : ℝ) := by
    simpa [Nat.cast_add, Nat.cast_one] using hratio
  simpa [Nat.succ_eq_add_one] using
    regularizedApproximation_pipeline_of_masterAbstractRate_ratioBound_of_le_index
      hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap
      hmargin (n + 1) hratio'

/--
Stricter-index companion of
`regularizedApproximation_pipeline_of_masterAbstractRate_closedFormCeil`.
-/
theorem regularizedApproximation_pipeline_of_masterAbstractRate_closedFormCeil_of_le_index
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias target alpha B : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (hmargin : bias < target)
    (n : ℕ)
    (hn : Nat.ceil ((alpha * B) / (target - bias)) ≤ n) :
    |F0 - Fgamma n| ≤ target := by
  exact regularizedApproximation_pipeline_of_masterAbstractRate_closedFormCeil
    hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap hmargin n
    (hn.trans (Nat.le_succ n))

/--
Closed-form final-pipeline bridge using an intermediate natural-number bound.

This wraps the frequent two-step pattern:
`ceil((alpha * B) / (target - bias)) ≤ N` followed by `N ≤ n`.
-/
theorem regularizedApproximation_pipeline_of_masterAbstractRate_closedFormCeil_of_natBound
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias target alpha B : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (hmargin : bias < target)
    (n N : ℕ)
    (hceilN : Nat.ceil ((alpha * B) / (target - bias)) ≤ N)
    (hN : N ≤ n) :
    |F0 - Fgamma n| ≤ target := by
  exact regularizedApproximation_pipeline_of_masterAbstractRate_closedFormCeil_of_le_index
    hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap
    hmargin n (hceilN.trans hN)

/--
Successor-index closed-form-threshold bridge for the final-target pipeline.

This is the `n+1` convenience form of
`regularizedApproximation_pipeline_of_masterAbstractRate_closedFormCeil_of_le_index`.
-/
theorem regularizedApproximation_pipeline_of_masterAbstractRate_closedFormCeil_succ
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias target alpha B : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (hmargin : bias < target)
    (n : ℕ)
    (hn : Nat.ceil ((alpha * B) / (target - bias)) ≤ n + 1) :
    |F0 - Fgamma (n + 1)| ≤ target := by
  simpa [Nat.succ_eq_add_one] using
    regularizedApproximation_pipeline_of_masterAbstractRate_closedFormCeil_of_le_index
      hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap
      hmargin (n + 1) hn

/--
Successor-index master-rate closed-form-threshold bridge from an intermediate natural bound.
-/
theorem dualRate_iterationThreshold_of_masterAbstractRate_closedFormCeil_succ_of_natBound
    {phi gap residual : ℕ → ℝ}
    {alpha B eps : ℝ}
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (heps : 0 < eps)
    (n N : ℕ)
    (hceilN : Nat.ceil ((alpha * B) / eps) ≤ N)
    (hN : N ≤ n + 1) :
    gap (n + 1) ≤ eps :=
  dualRate_iterationThreshold_of_masterAbstractRate_closedFormCeil_succ
    halpha hgap_res hres_ascent hphi_bound hmono_gap heps n (hceilN.trans hN)

/--
Successor-index final-pipeline closed-form bridge from an intermediate natural bound.
-/
theorem regularizedApproximation_pipeline_of_masterAbstractRate_closedFormCeil_succ_of_natBound
    {phi gap residual : ℕ → ℝ}
    {F0 FgammaStar bias target alpha B : ℝ}
    {Fgamma : ℕ → ℝ}
    (hbias : |F0 - FgammaStar| ≤ bias)
    (halpha : 0 ≤ alpha)
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (hobj_gap : ∀ n : ℕ, |FgammaStar - Fgamma n| ≤ gap n)
    (hmargin : bias < target)
    (n N : ℕ)
    (hceilN : Nat.ceil ((alpha * B) / (target - bias)) ≤ N)
    (hN : N ≤ n + 1) :
    |F0 - Fgamma (n + 1)| ≤ target :=
  regularizedApproximation_pipeline_of_masterAbstractRate_closedFormCeil_succ
    hbias halpha hgap_res hres_ascent hphi_bound hmono_gap hobj_gap hmargin n
    (hceilN.trans hN)

end DualConvergence
end KLProjection
end FlowSinkhorn
