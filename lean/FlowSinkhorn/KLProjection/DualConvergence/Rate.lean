import FlowSinkhorn.KLProjection.DualConvergence.GapResidual
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# Robust `O(1/k)` rate

This module is reserved for the Lean formalization of Theorem `thm:KL-dual-rate` and the
regularization/complexity corollaries from
`papers/kl-projections/sections/sec-dual-convergence.tex`.

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
- `regularizedApproximation_error_le_eps_of_closedFormIterationThreshold`;
- `regularizedApproximation_finalEpsilon_of_stoppingRule`;
- `regularizedApproximation_error_le_eps_of_iterationThreshold`;
- `regularizedApproximation_complexity_of_closedFormIterationThreshold`;
- `regularizedApproximation_targetAccuracy_of_closedFormIterationThreshold`;
- `regularizedApproximation_complexity`.

Design note:
this file should expose a paper-faithful top theorem whose hypotheses are exactly the constants
later discharged in `PrimalDualBounds` and the application modules.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace DualConvergence

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
