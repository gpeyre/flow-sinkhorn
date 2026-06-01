import FlowSinkhorn.KLProjection.DualConvergence.Pinsker
import FlowSinkhorn.KLProjection.DualConvergence.Vocabulary
import Mathlib.Data.Real.Basic
import Mathlib.Order.Monotone.Basic
import Mathlib.Tactic

set_option linter.style.longLine false

/-!
# Per-step ascent

This module is reserved for the Lean formalization of Lemma `lem:per-step-ascent` from
the dual-convergence material in `neurips/paper.tex`.

Intended theorem names:
- `perStepAscent_block1`;
- `perStepAscent_block2`;
- `perStepAscent_twoStep`.

Design note:
this is the first genuinely optimization-specific ingredient in the rate proof, so it should keep
all objective-difference identities local and export only the final residual-ascent inequality to
later modules.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace DualConvergence

open Nat

variable {phi residual : ℕ → ℝ}

/--
Recursive cumulative sum used to avoid extra finite-sum machinery in the first formalization pass.
-/
def cumulative (r : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => cumulative r n + r n

/--
Block-1 paper-facing wrapper for a one-step ascent inequality.
-/
theorem perStepAscent_block1
    {phi1 r1 : ℕ → ℝ}
    (hstep1 : ∀ k : ℕ, r1 k ≤ phi1 (k + 1) - phi1 k)
    (k : ℕ) :
    r1 k ≤ phi1 (k + 1) - phi1 k :=
  hstep1 k

/--
Block-2 paper-facing wrapper for a one-step ascent inequality.
-/
theorem perStepAscent_block2
    {phi2 r2 : ℕ → ℝ}
    (hstep2 : ∀ k : ℕ, r2 k ≤ phi2 (k + 1) - phi2 k)
    (k : ℕ) :
    r2 k ≤ phi2 (k + 1) - phi2 k :=
  hstep2 k

/--
Cumulative ascent control: if each residual is bounded by one-step objective increase, then the
cumulative residual is controlled by the total objective increase.
-/
theorem perStepAscent_twoStep
    (hstep : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k) :
    ∀ n : ℕ, cumulative residual n ≤ phi n - phi 0 := by
  intro n
  induction n with
  | zero =>
      simp [cumulative]
  | succ n ih =>
      calc
        cumulative residual (n + 1)
            = cumulative residual n + residual n := by
              simp [cumulative]
        _ ≤ (phi n - phi 0) + (phi (n + 1) - phi n) := by
              exact add_le_add ih (hstep n)
        _ = phi (n + 1) - phi 0 := by ring

/--
Antitone sequence lower-bounds its own cumulative sum:
`(n+1) * g_n ≤ \sum_{i=0}^n g_i` (encoded with `cumulative`).
-/
theorem mul_le_cumulative_of_antitone
    {g : ℕ → ℝ}
    (hmono : Antitone g) :
    ∀ n : ℕ, ((n + 1 : ℝ) * g n) ≤ cumulative g (n + 1) := by
  intro n
  induction n with
  | zero =>
      simp [cumulative]
  | succ n ih =>
      have hmono_step : g (n + 1) ≤ g n := hmono (Nat.le_succ n)
      have hmul_step : ((↑n + 1 : ℝ) * g (n + 1)) ≤ ((↑n + 1 : ℝ) * g n) := by
        nlinarith [hmono_step]
      have hprod : ((↑n + 1 : ℝ) * g (n + 1)) ≤ cumulative g (n + 1) :=
        hmul_step.trans ih
      have hsplit : (((↑n + 1 : ℝ) + 1) * g (n + 1)) =
          ((↑n + 1 : ℝ) * g (n + 1)) + g (n + 1) := by
        ring
      have hmain : ((↑(n + 1) + 1) * g (n + 1)) ≤ cumulative g (n + 1) + g (n + 1) := by
        have hmain' : (((↑n + 1 : ℝ) + 1) * g (n + 1)) ≤ cumulative g (n + 1) + g (n + 1) := by
          calc
            (((↑n + 1 : ℝ) + 1) * g (n + 1))
                = ((↑n + 1 : ℝ) * g (n + 1)) + g (n + 1) := hsplit
            _ ≤ cumulative g (n + 1) + g (n + 1) := by
                  exact add_le_add hprod (le_rfl)
        simpa [Nat.cast_add, Nat.cast_one, add_assoc, add_left_comm, add_comm] using hmain'
      calc
        ((↑(n + 1) + 1) * g (n + 1))
            ≤ cumulative g (n + 1) + g (n + 1) := hmain
        _ = cumulative g ((n + 1) + 1) := by
              simp [cumulative]

/--
Monotonicity of the recursive cumulative sum with respect to pointwise order.
-/
theorem cumulative_le_of_le
    {a b : ℕ → ℝ}
    (h : ∀ k : ℕ, a k ≤ b k) :
    ∀ n : ℕ, cumulative a n ≤ cumulative b n := by
  intro n
  induction n with
  | zero =>
      simp [cumulative]
  | succ n ih =>
      simp [cumulative, add_le_add ih (h n)]

/--
Uniform upper bound implies linear control of the recursive cumulative sum.

This is a reusable arithmetic lemma for future residual-budget estimates, independent of the
specific ascent argument.
-/
theorem cumulative_le_linear_of_uniform_bound
    {a : ℕ → ℝ} {B : ℝ}
    (h : ∀ k : ℕ, a k ≤ B) :
    ∀ n : ℕ, cumulative a n ≤ (n : ℝ) * B := by
  intro n
  induction n with
  | zero =>
      simp [cumulative]
  | succ n ih =>
      have hstep : cumulative a n + a n ≤ (n : ℝ) * B + B := by
        exact add_le_add ih (h n)
      calc
        cumulative a (n + 1)
            = cumulative a n + a n := by
              simp [cumulative]
        _ ≤ (n : ℝ) * B + B := hstep
        _ = ((n : ℝ) + 1) * B := by
              ring
        _ = ((n + 1 : ℕ) : ℝ) * B := by
              simp [Nat.cast_add, Nat.cast_one]

/--
Compatibility of `cumulative` with left scalar multiplication.
-/
theorem cumulative_mul_left
    (c : ℝ) (a : ℕ → ℝ) :
    ∀ n : ℕ, cumulative (fun k => c * a k) n = c * cumulative a n := by
  intro n
  induction n with
  | zero =>
      simp [cumulative]
  | succ n ih =>
      simp [cumulative, ih, mul_add, add_comm]

/--
Nonnegativity of the recursive cumulative sum when every term is nonneg.
-/
theorem cumulative_nonneg_of_nonneg
    {r : ℕ → ℝ} (hr : ∀ k, 0 ≤ r k) (n : ℕ) :
    0 ≤ cumulative r n := by
  induction n with
  | zero =>
      simp [cumulative]
  | succ n ih =>
      simp [cumulative]
      linarith [hr n]

/--
The recursive cumulative sum is nondecreasing when every term is nonneg.
-/
theorem cumulative_mono_of_nonneg
    {r : ℕ → ℝ} (hr : ∀ k, 0 ≤ r k) :
    ∀ m n : ℕ, m ≤ n → cumulative r m ≤ cumulative r n := by
  intro m n hmn
  induction hmn with
  | refl =>
      exact le_refl _
  | @step n' _ ih =>
      simp [cumulative]
      linarith [hr n', ih]

/--
Additivity of the recursive cumulative sum: `cumulative (r + s) = cumulative r + cumulative s`.
-/
theorem cumulative_add
    (r s : ℕ → ℝ) (n : ℕ) :
    cumulative (fun k => r k + s k) n = cumulative r n + cumulative s n := by
  induction n with
  | zero =>
      simp [cumulative]
  | succ n ih =>
      simp [cumulative, ih]
      ring

/--
Compatibility of `cumulative` with constant scalar multiplication (right scalar form).
-/
theorem cumulative_const_mul
    (c : ℝ) (r : ℕ → ℝ) (n : ℕ) :
    cumulative (fun k => c * r k) n = c * cumulative r n := by
  induction n with
  | zero =>
      simp [cumulative]
  | succ n ih =>
      simp [cumulative, ih]
      ring

/--
Cumulative sum bounded by a telescoping argument.

If each `gap k` is bounded by a one-step objective increase `phi (k+1) - phi k`, and the total
objective growth is bounded by `B`, then the cumulative sum of `gap` is at most `B`.
-/
theorem perStepAscent_cumulative_bounded
    {gap phi : ℕ → ℝ} {B : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hbounded : ∀ n : ℕ, phi (n + 1) ≤ phi 0 + B)
    (hB : 0 ≤ B)
    (n : ℕ) :
    cumulative gap n ≤ B := by
  have hper := perStepAscent_twoStep hstep n
  cases n with
  | zero =>
      change (0 : ℝ) ≤ B
      exact hB
  | succ n =>
      have hphi : phi (n + 1) - phi 0 ≤ B := by linarith [hbounded n]
      linarith

/--
Successor-index cumulative bound from per-step ascent and objective budget.

Unlike `perStepAscent_cumulative_bounded`, this version targets `cumulative gap (n+1)` directly,
so it does not require an explicit nonnegativity assumption on `B`.
-/
theorem perStepAscent_cumulative_succ_bounded
    {gap phi : ℕ → ℝ} {B : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hbounded : ∀ n : ℕ, phi (n + 1) ≤ phi 0 + B)
    (n : ℕ) :
    cumulative gap (n + 1) ≤ B := by
  have hper : cumulative gap (n + 1) ≤ phi (n + 1) - phi 0 :=
    perStepAscent_twoStep (phi := phi) (residual := gap) hstep (n + 1)
  have hphi : phi (n + 1) - phi 0 ≤ B := by
    linarith [hbounded n]
  exact hper.trans hphi

/--
One full sweep ascent from the two block-wise ascent inequalities.

This is the algebraic composition of the paper inequalities (A1) and (A2): if each half-step gains
at least a block-residual term, then the whole sweep gains at least the sum of both terms.
-/
theorem perStepAscent_fullSweep_of_twoBlockIncrements
    {F Fhalf r1 r2 : ℕ → ℝ} {c : ℝ}
    (hA1 : ∀ k : ℕ, c * (r1 k) ^ 2 ≤ Fhalf k - F k)
    (hA2 : ∀ k : ℕ, c * (r2 k) ^ 2 ≤ F (k + 1) - Fhalf k) :
    ∀ k : ℕ, c * ((r1 k) ^ 2 + (r2 k) ^ 2) ≤ F (k + 1) - F k := by
  intro k
  have hsum : c * (r1 k) ^ 2 + c * (r2 k) ^ 2 ≤ (Fhalf k - F k) + (F (k + 1) - Fhalf k) :=
    add_le_add (hA1 k) (hA2 k)
  calc
    c * ((r1 k) ^ 2 + (r2 k) ^ 2)
        = c * (r1 k) ^ 2 + c * (r2 k) ^ 2 := by ring
    _ ≤ (Fhalf k - F k) + (F (k + 1) - Fhalf k) := hsum
    _ = F (k + 1) - F k := by ring

/--
Paper-shaped one-step ascent interface for the rate theorem:
define the residual proxy as the sum of two block residual squares over one full sweep.
-/
theorem perStepAscent_residualProxy_of_twoBlockIncrements
    {F Fhalf r1 r2 : ℕ → ℝ} {c : ℝ}
    (hA1 : ∀ k : ℕ, c * (r1 k) ^ 2 ≤ Fhalf k - F k)
    (hA2 : ∀ k : ℕ, c * (r2 k) ^ 2 ≤ F (k + 1) - Fhalf k) :
    ∀ k : ℕ, c * ((r1 k) ^ 2 + (r2 k) ^ 2) ≤ F (k + 1) - F k :=
  perStepAscent_fullSweep_of_twoBlockIncrements hA1 hA2

/--
Half-step ascent from a KL-gain estimate and a Pinsker residual estimate.

This internalizes the algebraic Pinsker-to-ascent step used in Lemma A.1: once the KL gain of a
block update is at most the objective increase, and Pinsker controls the squared residual by that
KL gain, the desired quadratic ascent inequality follows by transitivity.
-/
theorem halfStepAscent_of_klGain_of_pinsker
    {Fbefore Fafter kl r M : ℝ}
    (hpinsker : r ^ 2 / (2 * M) ≤ kl)
    (hgain : kl ≤ Fafter - Fbefore) :
    r ^ 2 / (2 * M) ≤ Fafter - Fbefore :=
  hpinsker.trans hgain

/--
Sequence form of `halfStepAscent_of_klGain_of_pinsker`.

The hypotheses expose exactly the two analytic inputs that should eventually be derived from
KL projection geometry: a Pinsker residual estimate and a KL/objective-gain identity.
-/
theorem halfStepAscent_seq_of_klGain_of_pinsker
    {Fbefore Fafter kl r : ℕ → ℝ} {M : ℝ}
    (hpinsker : ∀ k : ℕ, (r k) ^ 2 / (2 * M) ≤ kl k)
    (hgain : ∀ k : ℕ, kl k ≤ Fafter k - Fbefore k) :
    ∀ k : ℕ, (r k) ^ 2 / (2 * M) ≤ Fafter k - Fbefore k := by
  intro k
  exact halfStepAscent_of_klGain_of_pinsker (hpinsker k) (hgain k)

/--
Convert an additive KL-gain statement to the difference form used by the base half-step lemmas.
-/
theorem klGain_of_klGain_add
    {Fbefore Fafter kl : ℝ}
    (hgain_add : Fbefore + kl ≤ Fafter) :
    kl ≤ Fafter - Fbefore := by
  linarith

/-- Sequence form of `klGain_of_klGain_add`. -/
theorem klGain_seq_of_klGain_add
    {Fbefore Fafter kl : ℕ → ℝ}
    (hgain_add : ∀ k : ℕ, Fbefore k + kl k ≤ Fafter k) :
    ∀ k : ℕ, kl k ≤ Fafter k - Fbefore k := by
  intro k
  exact klGain_of_klGain_add (hgain_add k)

/--
Half-step ascent from Pinsker and an additive KL-gain statement.

This is the common paper shape `Fbefore + KL ≤ Fafter`, avoiding local `linarith` calls before
using `halfStepAscent_of_klGain_of_pinsker`.
-/
theorem halfStepAscent_of_klGain_add_of_pinsker
    {Fbefore Fafter kl r M : ℝ}
    (hpinsker : r ^ 2 / (2 * M) ≤ kl)
    (hgain_add : Fbefore + kl ≤ Fafter) :
    r ^ 2 / (2 * M) ≤ Fafter - Fbefore :=
  halfStepAscent_of_klGain_of_pinsker hpinsker (klGain_of_klGain_add hgain_add)

/-- Sequence form of `halfStepAscent_of_klGain_add_of_pinsker`. -/
theorem halfStepAscent_seq_of_klGain_add_of_pinsker
    {Fbefore Fafter kl r : ℕ → ℝ} {M : ℝ}
    (hpinsker : ∀ k : ℕ, (r k) ^ 2 / (2 * M) ≤ kl k)
    (hgain_add : ∀ k : ℕ, Fbefore k + kl k ≤ Fafter k) :
    ∀ k : ℕ, (r k) ^ 2 / (2 * M) ≤ Fafter k - Fbefore k := by
  intro k
  exact halfStepAscent_of_klGain_add_of_pinsker (hpinsker k) (hgain_add k)

/--
Convert a half-step ascent estimate in the primal-change norm into the paper's residual form.

In Lemma A.1 the residual satisfies
`||r||₁ ≤ ||A||_{1→1} ||x⁺-x||₁`.  This lemma packages the purely algebraic conversion from
the primal-change estimate to the displayed residual estimate with denominator
`||A||_{1→1}²`.
-/
theorem halfStepAscent_paperConstant_of_primalChangeBound
    {change residual Fbefore Fafter gamma Xmax Anorm : ℝ}
    (hgamma_nonneg : 0 ≤ gamma)
    (hXmax_pos : 0 < Xmax)
    (hAnorm_pos : 0 < Anorm)
    (hresidual_nonneg : 0 ≤ residual)
    (hresidual_bound : residual ≤ Anorm * change)
    (hchange_ascent : (gamma / (2 * Xmax)) * change ^ 2 ≤ Fafter - Fbefore) :
    (gamma / (2 * Xmax)) * (residual ^ 2 / Anorm ^ 2) ≤ Fafter - Fbefore := by
  have hres_sq_le : residual ^ 2 ≤ Anorm ^ 2 * change ^ 2 := by
    nlinarith [sq_nonneg (Anorm * change - residual)]
  have hAnorm_sq_pos : 0 < Anorm ^ 2 := sq_pos_of_pos hAnorm_pos
  have hdiv_le : residual ^ 2 / Anorm ^ 2 ≤ change ^ 2 := by
    rw [div_le_iff₀ hAnorm_sq_pos]
    simpa [pow_two, mul_assoc, mul_left_comm, mul_comm] using hres_sq_le
  have hcoef_nonneg : 0 ≤ gamma / (2 * Xmax) := by
    exact div_nonneg hgamma_nonneg (le_of_lt (mul_pos (by norm_num) hXmax_pos))
  exact (mul_le_mul_of_nonneg_left hdiv_le hcoef_nonneg).trans hchange_ascent

/--
Full-sweep A.1 endpoint from KL-gain and Pinsker premises for both blocks.

Compared with `perStepAscent_residualProxy_of_twoBlockIncrements`, this theorem no longer asks
for the two half-step ascent inequalities directly.  Instead it derives them internally from
blockwise KL-gain controls plus Pinsker-style residual estimates, then composes the two
half-steps into the full sweep.
-/
theorem perStepAscent_residualProxy_of_klGains_pinsker_commonMass
    {F Fhalf kl1 kl2 r1 r2 : ℕ → ℝ} {M : ℝ}
    (hpinsker1 : ∀ k : ℕ, (r1 k) ^ 2 / (2 * M) ≤ kl1 k)
    (hgain1 : ∀ k : ℕ, kl1 k ≤ Fhalf k - F k)
    (hpinsker2 : ∀ k : ℕ, (r2 k) ^ 2 / (2 * M) ≤ kl2 k)
    (hgain2 : ∀ k : ℕ, kl2 k ≤ F (k + 1) - Fhalf k) :
    ∀ k : ℕ, (1 / (2 * M)) * ((r1 k) ^ 2 + (r2 k) ^ 2) ≤ F (k + 1) - F k := by
  have hA1 :
      ∀ k : ℕ, (1 / (2 * M)) * (r1 k) ^ 2 ≤ Fhalf k - F k := by
    intro k
    have h := halfStepAscent_of_klGain_of_pinsker (hpinsker1 k) (hgain1 k)
    simpa [div_eq_mul_inv, one_div, mul_comm, mul_left_comm, mul_assoc] using h
  have hA2 :
      ∀ k : ℕ, (1 / (2 * M)) * (r2 k) ^ 2 ≤ F (k + 1) - Fhalf k := by
    intro k
    have h := halfStepAscent_of_klGain_of_pinsker (hpinsker2 k) (hgain2 k)
    simpa [div_eq_mul_inv, one_div, mul_comm, mul_left_comm, mul_assoc] using h
  exact perStepAscent_fullSweep_of_twoBlockIncrements
    (F := F) (Fhalf := Fhalf) (r1 := r1) (r2 := r2)
    (c := 1 / (2 * M)) hA1 hA2

/--
Full-sweep A.1 endpoint from additive KL-gain and Pinsker premises for both blocks.
-/
theorem perStepAscent_residualProxy_of_klGains_add_pinsker_commonMass
    {F Fhalf kl1 kl2 r1 r2 : ℕ → ℝ} {M : ℝ}
    (hpinsker1 : ∀ k : ℕ, (r1 k) ^ 2 / (2 * M) ≤ kl1 k)
    (hgain1_add : ∀ k : ℕ, F k + kl1 k ≤ Fhalf k)
    (hpinsker2 : ∀ k : ℕ, (r2 k) ^ 2 / (2 * M) ≤ kl2 k)
    (hgain2_add : ∀ k : ℕ, Fhalf k + kl2 k ≤ F (k + 1)) :
    ∀ k : ℕ, (1 / (2 * M)) * ((r1 k) ^ 2 + (r2 k) ^ 2) ≤ F (k + 1) - F k :=
  perStepAscent_residualProxy_of_klGains_pinsker_commonMass
    (F := F) (Fhalf := Fhalf) (kl1 := kl1) (kl2 := kl2) (r1 := r1) (r2 := r2)
    (M := M) hpinsker1
    (klGain_seq_of_klGain_add hgain1_add)
    hpinsker2
    (klGain_seq_of_klGain_add hgain2_add)

/--
Half-step ascent from the concrete finite non-normalized Pinsker conclusion.

This specializes `halfStepAscent_of_klGain_of_pinsker` to the finite objects used in Appendix A:
the KL term is `finiteKL p q` and the residual is the finite `l1Norm` of `p-q`.
-/
theorem halfStepAscent_of_klGain_of_finitePinsker
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter M : ℝ}
    (hpinsker :
      finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M))
    (hgain : finiteKL p q ≤ Fafter - Fbefore) :
    (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) ≤ Fafter - Fbefore :=
  hpinsker.trans hgain

/--
Sequence form of `halfStepAscent_of_klGain_of_finitePinsker`.

This is the direct half-step interface once each block update has been represented by finite
vectors before/after the KL projection.
-/
theorem halfStepAscent_seq_of_klGain_of_finitePinsker
    {n : ℕ} {p q : ℕ → Fin n → ℝ} {Fbefore Fafter : ℕ → ℝ} {M : ℝ}
    (hpinsker :
      ∀ k : ℕ,
        finiteKL (p k) (q k) ≥
          (l1Norm (fun i => p k i - q k i)) ^ 2 / (2 * M))
    (hgain : ∀ k : ℕ, finiteKL (p k) (q k) ≤ Fafter k - Fbefore k) :
    ∀ k : ℕ,
      (l1Norm (fun i => p k i - q k i)) ^ 2 / (2 * M)
        ≤ Fafter k - Fbefore k := by
  intro k
  exact halfStepAscent_of_klGain_of_finitePinsker (hpinsker k) (hgain k)

/--
Half-step ascent from concrete finite Pinsker and an additive KL-gain statement.
-/
theorem halfStepAscent_of_klGain_add_of_finitePinsker
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter M : ℝ}
    (hpinsker :
      finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M))
    (hgain_add : Fbefore + finiteKL p q ≤ Fafter) :
    (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) ≤ Fafter - Fbefore :=
  halfStepAscent_of_klGain_of_finitePinsker hpinsker
    (klGain_of_klGain_add hgain_add)

/-- Sequence form of `halfStepAscent_of_klGain_add_of_finitePinsker`. -/
theorem halfStepAscent_seq_of_klGain_add_of_finitePinsker
    {n : ℕ} {p q : ℕ → Fin n → ℝ} {Fbefore Fafter : ℕ → ℝ} {M : ℝ}
    (hpinsker :
      ∀ k : ℕ,
        finiteKL (p k) (q k) ≥
          (l1Norm (fun i => p k i - q k i)) ^ 2 / (2 * M))
    (hgain_add : ∀ k : ℕ, Fbefore k + finiteKL (p k) (q k) ≤ Fafter k) :
    ∀ k : ℕ,
      (l1Norm (fun i => p k i - q k i)) ^ 2 / (2 * M)
        ≤ Fafter k - Fbefore k := by
  intro k
  exact halfStepAscent_of_klGain_add_of_finitePinsker (hpinsker k) (hgain_add k)

/--
Full-sweep A.1 endpoint from concrete finite Pinsker conclusions for the two blocks.

This removes another abstraction layer from the paper-facing ascent proof path: the residuals are
now explicitly the finite `ℓ¹` residuals appearing in Pinsker, and the proof derives the two
half-step ascent inequalities before composing the sweep.
-/
theorem perStepAscent_residualProxy_of_finitePinsker_klGains_commonMass
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hpinsker1 :
      ∀ k : ℕ,
        finiteKL (p1 k) (q1 k) ≥
          (l1Norm (fun i => p1 k i - q1 k i)) ^ 2 / (2 * M))
    (hgain1 : ∀ k : ℕ, finiteKL (p1 k) (q1 k) ≤ Fhalf k - F k)
    (hpinsker2 :
      ∀ k : ℕ,
        finiteKL (p2 k) (q2 k) ≥
          (l1Norm (fun i => p2 k i - q2 k i)) ^ 2 / (2 * M))
    (hgain2 : ∀ k : ℕ, finiteKL (p2 k) (q2 k) ≤ F (k + 1) - Fhalf k) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k := by
  exact perStepAscent_residualProxy_of_klGains_pinsker_commonMass
    (F := F) (Fhalf := Fhalf)
    (kl1 := fun k => finiteKL (p1 k) (q1 k))
    (kl2 := fun k => finiteKL (p2 k) (q2 k))
    (r1 := fun k => l1Norm (fun i => p1 k i - q1 k i))
    (r2 := fun k => l1Norm (fun i => p2 k i - q2 k i))
    (M := M)
    (by
      intro k
      exact hpinsker1 k)
    hgain1
    (by
      intro k
      exact hpinsker2 k)
    hgain2

/--
Full-sweep A.1 endpoint from concrete finite Pinsker conclusions and additive KL gains.
-/
theorem perStepAscent_residualProxy_of_finitePinsker_klGains_add_commonMass
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hpinsker1 :
      ∀ k : ℕ,
        finiteKL (p1 k) (q1 k) ≥
          (l1Norm (fun i => p1 k i - q1 k i)) ^ 2 / (2 * M))
    (hgain1_add : ∀ k : ℕ, F k + finiteKL (p1 k) (q1 k) ≤ Fhalf k)
    (hpinsker2 :
      ∀ k : ℕ,
        finiteKL (p2 k) (q2 k) ≥
          (l1Norm (fun i => p2 k i - q2 k i)) ^ 2 / (2 * M))
    (hgain2_add : ∀ k : ℕ, Fhalf k + finiteKL (p2 k) (q2 k) ≤ F (k + 1)) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k := by
  exact perStepAscent_residualProxy_of_finitePinsker_klGains_commonMass
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hpinsker1
    (klGain_seq_of_klGain_add hgain1_add)
    hpinsker2
    (klGain_seq_of_klGain_add hgain2_add)

/--
Half-step ascent from the current strongest mass-shell non-normalized Pinsker endpoint.

This bridges A.1 directly to A.3: non-normalized finite Pinsker is no longer supplied as a
single premise, but is derived from the signed variational inequality, bounded-Hoeffding MGF
input, and the common-mass shell assumptions.
-/
theorem halfStepAscent_of_massShellPinsker_klGain
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = M)
    (hvar :
      ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p i / M) *
            finiteSign (fun j => p j / M - q j / M) i)
          - Real.log
              (∑ i, (q i / M) *
                Real.exp (t * finiteSign (fun j => p j / M - q j / M) i))
          ≤ finiteKL (fun i => p i / M) (fun i => q i / M))
    (hhoeffding :
      ∀ s : Fin n → ℝ,
        (∀ i, |s i| ≤ 1) →
        ∀ t : ℝ, 0 ≤ t →
          ∑ i, (q i / M) * Real.exp (t * s i)
            ≤ Real.exp (t * (∑ i, (q i / M) * s i) + t ^ 2 / 2))
    (hgain : finiteKL p q ≤ Fafter - Fbefore) :
    (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) ≤ Fafter - Fbefore := by
  have hpinsker :
      finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) :=
    pinsker_nonnormalized_of_signed_variational_bounded_hoeffding_massShell
      (p := p) (q := q) hMpos hq_nonneg hq_mass hvar hhoeffding
  exact halfStepAscent_of_klGain_of_finitePinsker hpinsker hgain

/--
Sequence form of `halfStepAscent_of_massShellPinsker_klGain`.
-/
theorem halfStepAscent_seq_of_massShellPinsker_klGain
    {n : ℕ} {p q : ℕ → Fin n → ℝ} {Fbefore Fafter : ℕ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ k i, 0 ≤ q k i)
    (hq_mass : ∀ k, ∑ i, q k i = M)
    (hvar :
      ∀ k : ℕ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p k i / M) *
            finiteSign (fun j => p k j / M - q k j / M) i)
          - Real.log
              (∑ i, (q k i / M) *
                Real.exp (t * finiteSign (fun j => p k j / M - q k j / M) i))
          ≤ finiteKL (fun i => p k i / M) (fun i => q k i / M))
    (hhoeffding :
      ∀ k, ∀ s : Fin n → ℝ,
        (∀ i, |s i| ≤ 1) →
        ∀ t : ℝ, 0 ≤ t →
          ∑ i, (q k i / M) * Real.exp (t * s i)
            ≤ Real.exp (t * (∑ i, (q k i / M) * s i) + t ^ 2 / 2))
    (hgain : ∀ k : ℕ, finiteKL (p k) (q k) ≤ Fafter k - Fbefore k) :
    ∀ k : ℕ,
      (l1Norm (fun i => p k i - q k i)) ^ 2 / (2 * M)
        ≤ Fafter k - Fbefore k := by
  intro k
  exact halfStepAscent_of_massShellPinsker_klGain
    (p := p k) (q := q k) (M := M) hMpos
    (hq_nonneg k) (hq_mass k)
    (by
      intro t ht
      exact hvar k t ht)
    (hhoeffding k)
    (hgain k)

/--
Half-step ascent from mass-shell Pinsker ingredients and an additive KL-gain statement.
-/
theorem halfStepAscent_of_massShellPinsker_klGain_add
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = M)
    (hvar :
      ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p i / M) *
            finiteSign (fun j => p j / M - q j / M) i)
          - Real.log
              (∑ i, (q i / M) *
                Real.exp (t * finiteSign (fun j => p j / M - q j / M) i))
          ≤ finiteKL (fun i => p i / M) (fun i => q i / M))
    (hhoeffding :
      ∀ s : Fin n → ℝ,
        (∀ i, |s i| ≤ 1) →
        ∀ t : ℝ, 0 ≤ t →
          ∑ i, (q i / M) * Real.exp (t * s i)
            ≤ Real.exp (t * (∑ i, (q i / M) * s i) + t ^ 2 / 2))
    (hgain_add : Fbefore + finiteKL p q ≤ Fafter) :
    (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) ≤ Fafter - Fbefore :=
  halfStepAscent_of_massShellPinsker_klGain
    (p := p) (q := q) (M := M) hMpos hq_nonneg hq_mass hvar hhoeffding
    (klGain_of_klGain_add hgain_add)

/-- Sequence form of `halfStepAscent_of_massShellPinsker_klGain_add`. -/
theorem halfStepAscent_seq_of_massShellPinsker_klGain_add
    {n : ℕ} {p q : ℕ → Fin n → ℝ} {Fbefore Fafter : ℕ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ k i, 0 ≤ q k i)
    (hq_mass : ∀ k, ∑ i, q k i = M)
    (hvar :
      ∀ k : ℕ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p k i / M) *
            finiteSign (fun j => p k j / M - q k j / M) i)
          - Real.log
              (∑ i, (q k i / M) *
                Real.exp (t * finiteSign (fun j => p k j / M - q k j / M) i))
          ≤ finiteKL (fun i => p k i / M) (fun i => q k i / M))
    (hhoeffding :
      ∀ k, ∀ s : Fin n → ℝ,
        (∀ i, |s i| ≤ 1) →
        ∀ t : ℝ, 0 ≤ t →
          ∑ i, (q k i / M) * Real.exp (t * s i)
            ≤ Real.exp (t * (∑ i, (q k i / M) * s i) + t ^ 2 / 2))
    (hgain_add : ∀ k : ℕ, Fbefore k + finiteKL (p k) (q k) ≤ Fafter k) :
    ∀ k : ℕ,
      (l1Norm (fun i => p k i - q k i)) ^ 2 / (2 * M)
        ≤ Fafter k - Fbefore k := by
  intro k
  exact halfStepAscent_of_massShellPinsker_klGain_add
    (p := p k) (q := q k) (M := M) hMpos
    (hq_nonneg k) (hq_mass k)
    (by
      intro t ht
      exact hvar k t ht)
    (hhoeffding k)
    (hgain_add k)

/--
Full-sweep A.1 endpoint from mass-shell finite Pinsker premises and KL-gain controls.

This is currently the strongest A.1 bridge: the proof derives the two non-normalized Pinsker
inequalities from the A.3 mass-shell endpoint, derives the half-step ascent inequalities, and
then composes the two block updates into a full sweep.
-/
theorem perStepAscent_residualProxy_of_massShellPinsker_klGains_commonMass
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq1_nonneg : ∀ k i, 0 ≤ q1 k i)
    (hq1_mass : ∀ k, ∑ i, q1 k i = M)
    (hvar1 :
      ∀ k : ℕ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p1 k i / M) *
            finiteSign (fun j => p1 k j / M - q1 k j / M) i)
          - Real.log
              (∑ i, (q1 k i / M) *
                Real.exp (t * finiteSign (fun j => p1 k j / M - q1 k j / M) i))
          ≤ finiteKL (fun i => p1 k i / M) (fun i => q1 k i / M))
    (hhoeffding1 :
      ∀ k, ∀ s : Fin n₁ → ℝ,
        (∀ i, |s i| ≤ 1) →
        ∀ t : ℝ, 0 ≤ t →
          ∑ i, (q1 k i / M) * Real.exp (t * s i)
            ≤ Real.exp (t * (∑ i, (q1 k i / M) * s i) + t ^ 2 / 2))
    (hgain1 : ∀ k : ℕ, finiteKL (p1 k) (q1 k) ≤ Fhalf k - F k)
    (hq2_nonneg : ∀ k i, 0 ≤ q2 k i)
    (hq2_mass : ∀ k, ∑ i, q2 k i = M)
    (hvar2 :
      ∀ k : ℕ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p2 k i / M) *
            finiteSign (fun j => p2 k j / M - q2 k j / M) i)
          - Real.log
              (∑ i, (q2 k i / M) *
                Real.exp (t * finiteSign (fun j => p2 k j / M - q2 k j / M) i))
          ≤ finiteKL (fun i => p2 k i / M) (fun i => q2 k i / M))
    (hhoeffding2 :
      ∀ k, ∀ s : Fin n₂ → ℝ,
        (∀ i, |s i| ≤ 1) →
        ∀ t : ℝ, 0 ≤ t →
          ∑ i, (q2 k i / M) * Real.exp (t * s i)
            ≤ Real.exp (t * (∑ i, (q2 k i / M) * s i) + t ^ 2 / 2))
    (hgain2 : ∀ k : ℕ, finiteKL (p2 k) (q2 k) ≤ F (k + 1) - Fhalf k) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k := by
  have hpinsker1 :
      ∀ k : ℕ,
        finiteKL (p1 k) (q1 k) ≥
          (l1Norm (fun i => p1 k i - q1 k i)) ^ 2 / (2 * M) := by
    intro k
    exact pinsker_nonnormalized_of_signed_variational_bounded_hoeffding_massShell
      (p := p1 k) (q := q1 k) hMpos (hq1_nonneg k) (hq1_mass k)
      (by
        intro t ht
        exact hvar1 k t ht)
      (hhoeffding1 k)
  have hpinsker2 :
      ∀ k : ℕ,
        finiteKL (p2 k) (q2 k) ≥
          (l1Norm (fun i => p2 k i - q2 k i)) ^ 2 / (2 * M) := by
    intro k
    exact pinsker_nonnormalized_of_signed_variational_bounded_hoeffding_massShell
      (p := p2 k) (q := q2 k) hMpos (hq2_nonneg k) (hq2_mass k)
      (by
        intro t ht
        exact hvar2 k t ht)
      (hhoeffding2 k)
  exact perStepAscent_residualProxy_of_finitePinsker_klGains_commonMass
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hpinsker1 hgain1 hpinsker2 hgain2

/--
Full-sweep A.1 endpoint from mass-shell Pinsker ingredients and additive KL gains.
-/
theorem perStepAscent_residualProxy_of_massShellPinsker_klGains_commonMass_add
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq1_nonneg : ∀ k i, 0 ≤ q1 k i)
    (hq1_mass : ∀ k, ∑ i, q1 k i = M)
    (hvar1 :
      ∀ k : ℕ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p1 k i / M) *
            finiteSign (fun j => p1 k j / M - q1 k j / M) i)
          - Real.log
              (∑ i, (q1 k i / M) *
                Real.exp (t * finiteSign (fun j => p1 k j / M - q1 k j / M) i))
          ≤ finiteKL (fun i => p1 k i / M) (fun i => q1 k i / M))
    (hhoeffding1 :
      ∀ k, ∀ s : Fin n₁ → ℝ,
        (∀ i, |s i| ≤ 1) →
        ∀ t : ℝ, 0 ≤ t →
          ∑ i, (q1 k i / M) * Real.exp (t * s i)
            ≤ Real.exp (t * (∑ i, (q1 k i / M) * s i) + t ^ 2 / 2))
    (hgain1_add : ∀ k : ℕ, F k + finiteKL (p1 k) (q1 k) ≤ Fhalf k)
    (hq2_nonneg : ∀ k i, 0 ≤ q2 k i)
    (hq2_mass : ∀ k, ∑ i, q2 k i = M)
    (hvar2 :
      ∀ k : ℕ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p2 k i / M) *
            finiteSign (fun j => p2 k j / M - q2 k j / M) i)
          - Real.log
              (∑ i, (q2 k i / M) *
                Real.exp (t * finiteSign (fun j => p2 k j / M - q2 k j / M) i))
          ≤ finiteKL (fun i => p2 k i / M) (fun i => q2 k i / M))
    (hhoeffding2 :
      ∀ k, ∀ s : Fin n₂ → ℝ,
        (∀ i, |s i| ≤ 1) →
        ∀ t : ℝ, 0 ≤ t →
          ∑ i, (q2 k i / M) * Real.exp (t * s i)
            ≤ Real.exp (t * (∑ i, (q2 k i / M) * s i) + t ^ 2 / 2))
    (hgain2_add : ∀ k : ℕ, Fhalf k + finiteKL (p2 k) (q2 k) ≤ F (k + 1)) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k :=
  perStepAscent_residualProxy_of_massShellPinsker_klGains_commonMass
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hMpos hq1_nonneg hq1_mass hvar1 hhoeffding1
    (klGain_seq_of_klGain_add hgain1_add)
    hq2_nonneg hq2_mass hvar2 hhoeffding2
    (klGain_seq_of_klGain_add hgain2_add)

/--
Half-step ascent from the two-point-Hoeffding version of the mass-shell Pinsker endpoint.

This removes the general bounded-test Hoeffding premise from the half-step interface: for the
finite sign test used by Pinsker, Lean decomposes the MGF into positive/negative sign masses and
uses only the scalar two-point Hoeffding inequality.
-/
theorem halfStepAscent_of_twoPointPinsker_klGain
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = M)
    (hvar :
      ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p i / M) *
            finiteSign (fun j => p j / M - q j / M) i)
          - Real.log
              (∑ i, (q i / M) *
                Real.exp (t * finiteSign (fun j => p j / M - q j / M) i))
          ≤ finiteKL (fun i => p i / M) (fun i => q i / M))
    (htwoPoint :
      ∀ a b t : ℝ, 0 ≤ a → 0 ≤ b → a + b = 1 → 0 ≤ t →
        a * Real.exp t + b * Real.exp (-t)
          ≤ Real.exp (t * (a - b) + t ^ 2 / 2))
    (hgain : finiteKL p q ≤ Fafter - Fbefore) :
    (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) ≤ Fafter - Fbefore := by
  have hpinsker :
      finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) :=
    pinsker_nonnormalized_of_signed_variational_twoPoint_hoeffding_massShell
      (p := p) (q := q) hMpos hq_nonneg hq_mass hvar htwoPoint
  exact halfStepAscent_of_klGain_of_finitePinsker hpinsker hgain

/-- Sequence form of `halfStepAscent_of_twoPointPinsker_klGain`. -/
theorem halfStepAscent_seq_of_twoPointPinsker_klGain
    {n : ℕ} {p q : ℕ → Fin n → ℝ} {Fbefore Fafter : ℕ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ k i, 0 ≤ q k i)
    (hq_mass : ∀ k, ∑ i, q k i = M)
    (hvar :
      ∀ k : ℕ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p k i / M) *
            finiteSign (fun j => p k j / M - q k j / M) i)
          - Real.log
              (∑ i, (q k i / M) *
                Real.exp (t * finiteSign (fun j => p k j / M - q k j / M) i))
          ≤ finiteKL (fun i => p k i / M) (fun i => q k i / M))
    (htwoPoint :
      ∀ a b t : ℝ, 0 ≤ a → 0 ≤ b → a + b = 1 → 0 ≤ t →
        a * Real.exp t + b * Real.exp (-t)
          ≤ Real.exp (t * (a - b) + t ^ 2 / 2))
    (hgain : ∀ k : ℕ, finiteKL (p k) (q k) ≤ Fafter k - Fbefore k) :
    ∀ k : ℕ,
      (l1Norm (fun i => p k i - q k i)) ^ 2 / (2 * M)
        ≤ Fafter k - Fbefore k := by
  intro k
  exact halfStepAscent_of_twoPointPinsker_klGain
    (p := p k) (q := q k) (M := M) hMpos
    (hq_nonneg k) (hq_mass k)
    (by
      intro t ht
      exact hvar k t ht)
    htwoPoint
    (hgain k)

/--
Full-sweep A.1 endpoint from signed variational input, scalar two-point Hoeffding, and KL gains.

This is the strongest current A.1 bridge. It uses the strongest A.3 sign-test path, where the
finite MGF is internally reduced to a two-point mixture before applying the scalar analytic
inequality.
-/
theorem perStepAscent_residualProxy_of_twoPointPinsker_klGains_commonMass
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (htwoPoint :
      ∀ a b t : ℝ, 0 ≤ a → 0 ≤ b → a + b = 1 → 0 ≤ t →
        a * Real.exp t + b * Real.exp (-t)
          ≤ Real.exp (t * (a - b) + t ^ 2 / 2))
    (hq1_nonneg : ∀ k i, 0 ≤ q1 k i)
    (hq1_mass : ∀ k, ∑ i, q1 k i = M)
    (hvar1 :
      ∀ k : ℕ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p1 k i / M) *
            finiteSign (fun j => p1 k j / M - q1 k j / M) i)
          - Real.log
              (∑ i, (q1 k i / M) *
                Real.exp (t * finiteSign (fun j => p1 k j / M - q1 k j / M) i))
          ≤ finiteKL (fun i => p1 k i / M) (fun i => q1 k i / M))
    (hgain1 : ∀ k : ℕ, finiteKL (p1 k) (q1 k) ≤ Fhalf k - F k)
    (hq2_nonneg : ∀ k i, 0 ≤ q2 k i)
    (hq2_mass : ∀ k, ∑ i, q2 k i = M)
    (hvar2 :
      ∀ k : ℕ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p2 k i / M) *
            finiteSign (fun j => p2 k j / M - q2 k j / M) i)
          - Real.log
              (∑ i, (q2 k i / M) *
                Real.exp (t * finiteSign (fun j => p2 k j / M - q2 k j / M) i))
          ≤ finiteKL (fun i => p2 k i / M) (fun i => q2 k i / M))
    (hgain2 : ∀ k : ℕ, finiteKL (p2 k) (q2 k) ≤ F (k + 1) - Fhalf k) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k := by
  have hpinsker1 :
      ∀ k : ℕ,
        finiteKL (p1 k) (q1 k) ≥
          (l1Norm (fun i => p1 k i - q1 k i)) ^ 2 / (2 * M) := by
    intro k
    exact pinsker_nonnormalized_of_signed_variational_twoPoint_hoeffding_massShell
      (p := p1 k) (q := q1 k) hMpos (hq1_nonneg k) (hq1_mass k)
      (by
        intro t ht
        exact hvar1 k t ht)
      htwoPoint
  have hpinsker2 :
      ∀ k : ℕ,
        finiteKL (p2 k) (q2 k) ≥
          (l1Norm (fun i => p2 k i - q2 k i)) ^ 2 / (2 * M) := by
    intro k
    exact pinsker_nonnormalized_of_signed_variational_twoPoint_hoeffding_massShell
      (p := p2 k) (q := q2 k) hMpos (hq2_nonneg k) (hq2_mass k)
      (by
        intro t ht
        exact hvar2 k t ht)
      htwoPoint
  exact perStepAscent_residualProxy_of_finitePinsker_klGains_commonMass
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hpinsker1 hgain1 hpinsker2 hgain2

/--
Half-step ascent from the all-test finite variational version of two-point Pinsker.

This is stronger than `halfStepAscent_of_twoPointPinsker_klGain`: the sign-test variational
inequality is derived internally from a finite variational inequality available for every test
function.
-/
theorem halfStepAscent_of_finiteVariationalTwoPointPinsker_klGain
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = M)
    (hvarAll :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p i / M) * s i)
          - Real.log (∑ i, (q i / M) * Real.exp (t * s i))
          ≤ finiteKL (fun i => p i / M) (fun i => q i / M))
    (htwoPoint :
      ∀ a b t : ℝ, 0 ≤ a → 0 ≤ b → a + b = 1 → 0 ≤ t →
        a * Real.exp t + b * Real.exp (-t)
          ≤ Real.exp (t * (a - b) + t ^ 2 / 2))
    (hgain : finiteKL p q ≤ Fafter - Fbefore) :
    (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) ≤ Fafter - Fbefore := by
  have hpinsker :
      finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) :=
    pinsker_nonnormalized_of_finite_variational_twoPoint_hoeffding_massShell
      (p := p) (q := q) hMpos hq_nonneg hq_mass hvarAll htwoPoint
  exact halfStepAscent_of_klGain_of_finitePinsker hpinsker hgain

/-- Sequence form of `halfStepAscent_of_finiteVariationalTwoPointPinsker_klGain`. -/
theorem halfStepAscent_seq_of_finiteVariationalTwoPointPinsker_klGain
    {n : ℕ} {p q : ℕ → Fin n → ℝ} {Fbefore Fafter : ℕ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ k i, 0 ≤ q k i)
    (hq_mass : ∀ k, ∑ i, q k i = M)
    (hvarAll :
      ∀ k : ℕ, ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p k i / M) * s i)
          - Real.log (∑ i, (q k i / M) * Real.exp (t * s i))
          ≤ finiteKL (fun i => p k i / M) (fun i => q k i / M))
    (htwoPoint :
      ∀ a b t : ℝ, 0 ≤ a → 0 ≤ b → a + b = 1 → 0 ≤ t →
        a * Real.exp t + b * Real.exp (-t)
          ≤ Real.exp (t * (a - b) + t ^ 2 / 2))
    (hgain : ∀ k : ℕ, finiteKL (p k) (q k) ≤ Fafter k - Fbefore k) :
    ∀ k : ℕ,
      (l1Norm (fun i => p k i - q k i)) ^ 2 / (2 * M)
        ≤ Fafter k - Fbefore k := by
  intro k
  exact halfStepAscent_of_finiteVariationalTwoPointPinsker_klGain
    (p := p k) (q := q k) (M := M) hMpos
    (hq_nonneg k) (hq_mass k)
    (hvarAll k)
    htwoPoint
    (hgain k)

/--
Full-sweep A.1 endpoint from all-test finite variational input, scalar two-point Hoeffding, and
KL gains.

This is the strongest current A.1 endpoint: both the sign-test variational specialization and the
sign-test MGF decomposition are internalized before the two half-step ascent inequalities are
composed.
-/
theorem perStepAscent_residualProxy_of_finiteVariationalTwoPointPinsker_klGains_commonMass
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (htwoPoint :
      ∀ a b t : ℝ, 0 ≤ a → 0 ≤ b → a + b = 1 → 0 ≤ t →
        a * Real.exp t + b * Real.exp (-t)
          ≤ Real.exp (t * (a - b) + t ^ 2 / 2))
    (hq1_nonneg : ∀ k i, 0 ≤ q1 k i)
    (hq1_mass : ∀ k, ∑ i, q1 k i = M)
    (hvarAll1 :
      ∀ k : ℕ, ∀ s : Fin n₁ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p1 k i / M) * s i)
          - Real.log (∑ i, (q1 k i / M) * Real.exp (t * s i))
          ≤ finiteKL (fun i => p1 k i / M) (fun i => q1 k i / M))
    (hgain1 : ∀ k : ℕ, finiteKL (p1 k) (q1 k) ≤ Fhalf k - F k)
    (hq2_nonneg : ∀ k i, 0 ≤ q2 k i)
    (hq2_mass : ∀ k, ∑ i, q2 k i = M)
    (hvarAll2 :
      ∀ k : ℕ, ∀ s : Fin n₂ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p2 k i / M) * s i)
          - Real.log (∑ i, (q2 k i / M) * Real.exp (t * s i))
          ≤ finiteKL (fun i => p2 k i / M) (fun i => q2 k i / M))
    (hgain2 : ∀ k : ℕ, finiteKL (p2 k) (q2 k) ≤ F (k + 1) - Fhalf k) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k := by
  have hpinsker1 :
      ∀ k : ℕ,
        finiteKL (p1 k) (q1 k) ≥
          (l1Norm (fun i => p1 k i - q1 k i)) ^ 2 / (2 * M) := by
    intro k
    exact pinsker_nonnormalized_of_finite_variational_twoPoint_hoeffding_massShell
      (p := p1 k) (q := q1 k) hMpos (hq1_nonneg k) (hq1_mass k)
      (hvarAll1 k) htwoPoint
  have hpinsker2 :
      ∀ k : ℕ,
        finiteKL (p2 k) (q2 k) ≥
          (l1Norm (fun i => p2 k i - q2 k i)) ^ 2 / (2 * M) := by
    intro k
    exact pinsker_nonnormalized_of_finite_variational_twoPoint_hoeffding_massShell
      (p := p2 k) (q := q2 k) hMpos (hq2_nonneg k) (hq2_mass k)
      (hvarAll2 k) htwoPoint
  exact perStepAscent_residualProxy_of_finitePinsker_klGains_commonMass
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hpinsker1 hgain1 hpinsker2 hgain2

/--
Full-sweep A.1 endpoint from all-test finite variational input, centered Bernoulli Hoeffding,
and KL gains.

Compared with `perStepAscent_residualProxy_of_finiteVariationalTwoPointPinsker_klGains_commonMass`,
this endpoint no longer assumes the two-point sign-variable Hoeffding form directly.  Lean derives
that two-point form from the centered Bernoulli MGF inequality before applying A.3 to both blocks.
-/
theorem perStepAscent_residualProxy_of_finiteVariationalCenteredBernoulliPinsker_klGains_commonMass
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hcentered :
      ∀ a b t : ℝ, 0 ≤ a → 0 ≤ b → a + b = 1 → 0 ≤ t →
        a * Real.exp (2 * b * t) + b * Real.exp (-2 * a * t)
          ≤ Real.exp (t ^ 2 / 2))
    (hq1_nonneg : ∀ k i, 0 ≤ q1 k i)
    (hq1_mass : ∀ k, ∑ i, q1 k i = M)
    (hvarAll1 :
      ∀ k : ℕ, ∀ s : Fin n₁ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p1 k i / M) * s i)
          - Real.log (∑ i, (q1 k i / M) * Real.exp (t * s i))
          ≤ finiteKL (fun i => p1 k i / M) (fun i => q1 k i / M))
    (hgain1 : ∀ k : ℕ, finiteKL (p1 k) (q1 k) ≤ Fhalf k - F k)
    (hq2_nonneg : ∀ k i, 0 ≤ q2 k i)
    (hq2_mass : ∀ k, ∑ i, q2 k i = M)
    (hvarAll2 :
      ∀ k : ℕ, ∀ s : Fin n₂ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p2 k i / M) * s i)
          - Real.log (∑ i, (q2 k i / M) * Real.exp (t * s i))
          ≤ finiteKL (fun i => p2 k i / M) (fun i => q2 k i / M))
    (hgain2 : ∀ k : ℕ, finiteKL (p2 k) (q2 k) ≤ F (k + 1) - Fhalf k) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k := by
  exact perStepAscent_residualProxy_of_finiteVariationalTwoPointPinsker_klGains_commonMass
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hMpos
    (twoPoint_hoeffding_of_centeredBernoulli_hoeffding hcentered)
    hq1_nonneg hq1_mass hvarAll1 hgain1
    hq2_nonneg hq2_mass hvarAll2 hgain2

/--
Full-sweep A.1 endpoint from all-test finite variational input, the strict-interior centered
Bernoulli Hoeffding inequality, and KL gains.

This is the strongest current A.1 bridge: degenerate sign-mass and zero-time cases of the scalar
Bernoulli bound are discharged internally before applying the A.3 Pinsker mass-shell endpoint to
both blocks.
-/
theorem perStepAscent_residualProxy_of_finiteVariationalCenteredBernoulliInteriorPinsker_klGains_commonMass
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hcenteredInterior :
      ∀ a b t : ℝ, 0 < a → 0 < b → a + b = 1 → 0 < t →
        a * Real.exp (2 * b * t) + b * Real.exp (-2 * a * t)
          ≤ Real.exp (t ^ 2 / 2))
    (hq1_nonneg : ∀ k i, 0 ≤ q1 k i)
    (hq1_mass : ∀ k, ∑ i, q1 k i = M)
    (hvarAll1 :
      ∀ k : ℕ, ∀ s : Fin n₁ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p1 k i / M) * s i)
          - Real.log (∑ i, (q1 k i / M) * Real.exp (t * s i))
          ≤ finiteKL (fun i => p1 k i / M) (fun i => q1 k i / M))
    (hgain1 : ∀ k : ℕ, finiteKL (p1 k) (q1 k) ≤ Fhalf k - F k)
    (hq2_nonneg : ∀ k i, 0 ≤ q2 k i)
    (hq2_mass : ∀ k, ∑ i, q2 k i = M)
    (hvarAll2 :
      ∀ k : ℕ, ∀ s : Fin n₂ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p2 k i / M) * s i)
          - Real.log (∑ i, (q2 k i / M) * Real.exp (t * s i))
          ≤ finiteKL (fun i => p2 k i / M) (fun i => q2 k i / M))
    (hgain2 : ∀ k : ℕ, finiteKL (p2 k) (q2 k) ≤ F (k + 1) - Fhalf k) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k := by
  exact perStepAscent_residualProxy_of_finiteVariationalCenteredBernoulliPinsker_klGains_commonMass
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hMpos
    (centeredBernoulli_hoeffding_of_pos_pos_pos hcenteredInterior)
    hq1_nonneg hq1_mass hvarAll1 hgain1
    hq2_nonneg hq2_mass hvarAll2 hgain2

/--
Full-sweep A.1 endpoint from all-test finite variational input, the one-parameter
strict-interior centered Bernoulli Hoeffding inequality, and KL gains.

The scalar analytic input is now the usual one-probability form with mass `a` and complementary
mass `1-a`; Lean reconstructs the two-mass form and discharges degenerate cases internally.
-/
theorem perStepAscent_residualProxy_of_finiteVariationalCenteredBernoulliUnitPinsker_klGains_commonMass
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hcenteredUnit :
      ∀ a t : ℝ, 0 < a → a < 1 → 0 < t →
        a * Real.exp (2 * (1 - a) * t) + (1 - a) * Real.exp (-2 * a * t)
          ≤ Real.exp (t ^ 2 / 2))
    (hq1_nonneg : ∀ k i, 0 ≤ q1 k i)
    (hq1_mass : ∀ k, ∑ i, q1 k i = M)
    (hvarAll1 :
      ∀ k : ℕ, ∀ s : Fin n₁ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p1 k i / M) * s i)
          - Real.log (∑ i, (q1 k i / M) * Real.exp (t * s i))
          ≤ finiteKL (fun i => p1 k i / M) (fun i => q1 k i / M))
    (hgain1 : ∀ k : ℕ, finiteKL (p1 k) (q1 k) ≤ Fhalf k - F k)
    (hq2_nonneg : ∀ k i, 0 ≤ q2 k i)
    (hq2_mass : ∀ k, ∑ i, q2 k i = M)
    (hvarAll2 :
      ∀ k : ℕ, ∀ s : Fin n₂ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p2 k i / M) * s i)
          - Real.log (∑ i, (q2 k i / M) * Real.exp (t * s i))
          ≤ finiteKL (fun i => p2 k i / M) (fun i => q2 k i / M))
    (hgain2 : ∀ k : ℕ, finiteKL (p2 k) (q2 k) ≤ F (k + 1) - Fhalf k) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k := by
  exact perStepAscent_residualProxy_of_finiteVariationalCenteredBernoulliInteriorPinsker_klGains_commonMass
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hMpos
    (centeredBernoulli_interior_hoeffding_of_unitInterval hcenteredUnit)
    hq1_nonneg hq1_mass hvarAll1 hgain1
    hq2_nonneg hq2_mass hvarAll2 hgain2

/--
Half-step ascent from normalized all-test finite variational input with Hoeffding discharged by
mathlib.

This is the probability-coordinate analogue of the paper-facing mass-shell A.1 bridge: the caller
does not provide a Pinsker inequality or any scalar Hoeffding premise.
-/
theorem halfStepAscent_of_finiteVariationalMathlibPinsker_klGain
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = M)
    (hvarAll :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p i / M) * s i)
          - Real.log (∑ i, (q i / M) * Real.exp (t * s i))
          ≤ finiteKL (fun i => p i / M) (fun i => q i / M))
    (hgain : finiteKL p q ≤ Fafter - Fbefore) :
    (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) ≤ Fafter - Fbefore := by
  have hpinsker :
      finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) :=
    pinsker_nonnormalized_of_finite_variational_mathlib_hoeffding_massShell
      (p := p) (q := q) hMpos hq_nonneg hq_mass hvarAll
  exact halfStepAscent_of_klGain_of_finitePinsker hpinsker hgain

/-- Sequence form of `halfStepAscent_of_finiteVariationalMathlibPinsker_klGain`. -/
theorem halfStepAscent_seq_of_finiteVariationalMathlibPinsker_klGain
    {n : ℕ} {p q : ℕ → Fin n → ℝ} {Fbefore Fafter : ℕ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ k i, 0 ≤ q k i)
    (hq_mass : ∀ k, ∑ i, q k i = M)
    (hvarAll :
      ∀ k : ℕ, ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p k i / M) * s i)
          - Real.log (∑ i, (q k i / M) * Real.exp (t * s i))
          ≤ finiteKL (fun i => p k i / M) (fun i => q k i / M))
    (hgain : ∀ k : ℕ, finiteKL (p k) (q k) ≤ Fafter k - Fbefore k) :
    ∀ k : ℕ,
      (l1Norm (fun i => p k i - q k i)) ^ 2 / (2 * M)
        ≤ Fafter k - Fbefore k := by
  intro k
  exact halfStepAscent_of_finiteVariationalMathlibPinsker_klGain
    (p := p k) (q := q k) (M := M) hMpos
    (hq_nonneg k) (hq_mass k) (hvarAll k) (hgain k)

/--
Full-sweep A.1 endpoint from normalized all-test finite variational input and KL gains, with
Hoeffding discharged by mathlib.

Compared with the centered-Bernoulli endpoint above, this theorem removes the last scalar analytic
argument from the normalized finite-variational interface.
-/
theorem perStepAscent_residualProxy_of_finiteVariationalMathlibPinsker_klGains_commonMass
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq1_nonneg : ∀ k i, 0 ≤ q1 k i)
    (hq1_mass : ∀ k, ∑ i, q1 k i = M)
    (hvarAll1 :
      ∀ k : ℕ, ∀ s : Fin n₁ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p1 k i / M) * s i)
          - Real.log (∑ i, (q1 k i / M) * Real.exp (t * s i))
          ≤ finiteKL (fun i => p1 k i / M) (fun i => q1 k i / M))
    (hgain1 : ∀ k : ℕ, finiteKL (p1 k) (q1 k) ≤ Fhalf k - F k)
    (hq2_nonneg : ∀ k i, 0 ≤ q2 k i)
    (hq2_mass : ∀ k, ∑ i, q2 k i = M)
    (hvarAll2 :
      ∀ k : ℕ, ∀ s : Fin n₂ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p2 k i / M) * s i)
          - Real.log (∑ i, (q2 k i / M) * Real.exp (t * s i))
          ≤ finiteKL (fun i => p2 k i / M) (fun i => q2 k i / M))
    (hgain2 : ∀ k : ℕ, finiteKL (p2 k) (q2 k) ≤ F (k + 1) - Fhalf k) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k := by
  have hpinsker1 :
      ∀ k : ℕ,
        finiteKL (p1 k) (q1 k) ≥
          (l1Norm (fun i => p1 k i - q1 k i)) ^ 2 / (2 * M) := by
    intro k
    exact pinsker_nonnormalized_of_finite_variational_mathlib_hoeffding_massShell
      (p := p1 k) (q := q1 k) hMpos (hq1_nonneg k) (hq1_mass k) (hvarAll1 k)
  have hpinsker2 :
      ∀ k : ℕ,
        finiteKL (p2 k) (q2 k) ≥
          (l1Norm (fun i => p2 k i - q2 k i)) ^ 2 / (2 * M) := by
    intro k
    exact pinsker_nonnormalized_of_finite_variational_mathlib_hoeffding_massShell
      (p := p2 k) (q := q2 k) hMpos (hq2_nonneg k) (hq2_mass k) (hvarAll2 k)
  exact perStepAscent_residualProxy_of_finitePinsker_klGains_commonMass
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hpinsker1 hgain1 hpinsker2 hgain2

/--
Half-step ascent for probability-mass finite variational input (`M = 1`).

This convenience theorem keeps the common probability-simplex case out of the mass-shell notation:
the denominator is simplified to `2`, and scalar Hoeffding is still discharged by mathlib.
-/
theorem halfStepAscent_of_probabilityVariationalMathlibPinsker_klGain
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter : ℝ}
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = 1)
    (hvarAll :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p i * s i)
          - Real.log (∑ i, q i * Real.exp (t * s i))
          ≤ finiteKL p q)
    (hgain : finiteKL p q ≤ Fafter - Fbefore) :
    (l1Norm (fun i => p i - q i)) ^ 2 / 2 ≤ Fafter - Fbefore := by
  have hpinsker :
      finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * 1) :=
    pinsker_probabilityMass_of_finite_variational_mathlib_hoeffding
      (p := p) (q := q) hq_nonneg hq_mass hvarAll
  have h :=
    halfStepAscent_of_klGain_of_finitePinsker
      (M := 1) hpinsker hgain
  simpa using h

/-- Sequence form of `halfStepAscent_of_probabilityVariationalMathlibPinsker_klGain`. -/
theorem halfStepAscent_seq_of_probabilityVariationalMathlibPinsker_klGain
    {n : ℕ} {p q : ℕ → Fin n → ℝ} {Fbefore Fafter : ℕ → ℝ}
    (hq_nonneg : ∀ k i, 0 ≤ q k i)
    (hq_mass : ∀ k, ∑ i, q k i = 1)
    (hvarAll :
      ∀ k : ℕ, ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p k i * s i)
          - Real.log (∑ i, q k i * Real.exp (t * s i))
          ≤ finiteKL (p k) (q k))
    (hgain : ∀ k : ℕ, finiteKL (p k) (q k) ≤ Fafter k - Fbefore k) :
    ∀ k : ℕ,
      (l1Norm (fun i => p k i - q k i)) ^ 2 / 2
        ≤ Fafter k - Fbefore k := by
  intro k
  exact halfStepAscent_of_probabilityVariationalMathlibPinsker_klGain
    (p := p k) (q := q k) (hq_nonneg k) (hq_mass k) (hvarAll k) (hgain k)

/--
Full-sweep A.1 endpoint for already-normalized probability-mass block updates.

This is the cleanest probability-coordinate A.1 bridge: both block measures have unit mass, the
finite all-test variational inequalities are stated without `/ M`, and the final coefficient is
`1 / 2`.
-/
theorem perStepAscent_residualProxy_of_probabilityVariationalMathlibPinsker_klGains
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ}
    (hq1_nonneg : ∀ k i, 0 ≤ q1 k i)
    (hq1_mass : ∀ k, ∑ i, q1 k i = 1)
    (hvarAll1 :
      ∀ k : ℕ, ∀ s : Fin n₁ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p1 k i * s i)
          - Real.log (∑ i, q1 k i * Real.exp (t * s i))
          ≤ finiteKL (p1 k) (q1 k))
    (hgain1 : ∀ k : ℕ, finiteKL (p1 k) (q1 k) ≤ Fhalf k - F k)
    (hq2_nonneg : ∀ k i, 0 ≤ q2 k i)
    (hq2_mass : ∀ k, ∑ i, q2 k i = 1)
    (hvarAll2 :
      ∀ k : ℕ, ∀ s : Fin n₂ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p2 k i * s i)
          - Real.log (∑ i, q2 k i * Real.exp (t * s i))
          ≤ finiteKL (p2 k) (q2 k))
    (hgain2 : ∀ k : ℕ, finiteKL (p2 k) (q2 k) ≤ F (k + 1) - Fhalf k) :
    ∀ k : ℕ,
      (1 / 2) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k := by
  have hmass1 : ∀ k, ∑ i, q1 k i = (1 : ℝ) := hq1_mass
  have hmass2 : ∀ k, ∑ i, q2 k i = (1 : ℝ) := hq2_mass
  have h :=
    perStepAscent_residualProxy_of_finiteVariationalMathlibPinsker_klGains_commonMass
      (F := F) (Fhalf := Fhalf)
      (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2) (M := 1)
      (by norm_num) hq1_nonneg hmass1
      (by
        intro k s t ht
        simpa using hvarAll1 k s t ht)
      hgain1 hq2_nonneg hmass2
      (by
        intro k s t ht
        simpa using hvarAll2 k s t ht)
      hgain2
  intro k
  have hk := h k
  simpa using hk

/--
Half-step ascent for probability blocks from the paper-native mass-shell variational shape.

This is the unit-mass specialization of the common-mass A.1 interface: the logarithmic term is
written as `1 * log (sum (q / 1) * exp ...)`, as it appears after specializing the
non-normalized mass-shell variational inequality to probability measures, and Lean normalizes it
to the probability-coordinate endpoint.
-/
theorem halfStepAscent_of_probabilityMassShellVariationalMathlibPinsker_klGain
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter : ℝ}
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = 1)
    (hvarMass :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p i * s i)
          - 1 * Real.log (∑ i, (q i / 1) * Real.exp (t * s i))
          ≤ finiteKL p q)
    (hgain : finiteKL p q ≤ Fafter - Fbefore) :
    (l1Norm (fun i => p i - q i)) ^ 2 / 2 ≤ Fafter - Fbefore := by
  exact halfStepAscent_of_probabilityVariationalMathlibPinsker_klGain
    (p := p) (q := q) hq_nonneg hq_mass
    (by
      intro s t ht
      simpa using hvarMass s t ht)
    hgain

/--
Sequence form of
`halfStepAscent_of_probabilityMassShellVariationalMathlibPinsker_klGain`.
-/
theorem halfStepAscent_seq_of_probabilityMassShellVariationalMathlibPinsker_klGain
    {n : ℕ} {p q : ℕ → Fin n → ℝ} {Fbefore Fafter : ℕ → ℝ}
    (hq_nonneg : ∀ k i, 0 ≤ q k i)
    (hq_mass : ∀ k, ∑ i, q k i = 1)
    (hvarMass :
      ∀ k : ℕ, ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p k i * s i)
          - 1 * Real.log (∑ i, (q k i / 1) * Real.exp (t * s i))
          ≤ finiteKL (p k) (q k))
    (hgain : ∀ k : ℕ, finiteKL (p k) (q k) ≤ Fafter k - Fbefore k) :
    ∀ k : ℕ,
      (l1Norm (fun i => p k i - q k i)) ^ 2 / 2
        ≤ Fafter k - Fbefore k := by
  intro k
  exact halfStepAscent_of_probabilityMassShellVariationalMathlibPinsker_klGain
    (p := p k) (q := q k) (hq_nonneg k) (hq_mass k) (hvarMass k) (hgain k)

/--
Full-sweep A.1 endpoint for unit-mass blocks stated in the paper's mass-shell variational form.

This is useful when a concrete block update has already been proved on the probability simplex
but the variational optimality inequality was produced by the non-normalized KL projection
machinery specialized at mass `1`.
-/
theorem perStepAscent_residualProxy_of_probabilityMassShellVariationalMathlibPinsker_klGains
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ}
    (hq1_nonneg : ∀ k i, 0 ≤ q1 k i)
    (hq1_mass : ∀ k, ∑ i, q1 k i = 1)
    (hvarMass1 :
      ∀ k : ℕ, ∀ s : Fin n₁ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p1 k i * s i)
          - 1 * Real.log (∑ i, (q1 k i / 1) * Real.exp (t * s i))
          ≤ finiteKL (p1 k) (q1 k))
    (hgain1 : ∀ k : ℕ, finiteKL (p1 k) (q1 k) ≤ Fhalf k - F k)
    (hq2_nonneg : ∀ k i, 0 ≤ q2 k i)
    (hq2_mass : ∀ k, ∑ i, q2 k i = 1)
    (hvarMass2 :
      ∀ k : ℕ, ∀ s : Fin n₂ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p2 k i * s i)
          - 1 * Real.log (∑ i, (q2 k i / 1) * Real.exp (t * s i))
          ≤ finiteKL (p2 k) (q2 k))
    (hgain2 : ∀ k : ℕ, finiteKL (p2 k) (q2 k) ≤ F (k + 1) - Fhalf k) :
    ∀ k : ℕ,
      (1 / 2) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k := by
  exact perStepAscent_residualProxy_of_probabilityVariationalMathlibPinsker_klGains
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    hq1_nonneg hq1_mass
    (by
      intro k s t ht
      simpa using hvarMass1 k s t ht)
    hgain1 hq2_nonneg hq2_mass
    (by
      intro k s t ht
      simpa using hvarMass2 k s t ht)
    hgain2

/--
Half-step A.1 endpoint with the KL-gain premise in additive ascent form.

Many projection optimality proofs naturally produce `Fbefore + KL ≤ Fafter`; this theorem
converts that form internally before invoking the probability-coordinate Pinsker route.
-/
theorem halfStepAscent_of_probabilityVariationalMathlibPinsker_klGain_add
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter : ℝ}
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = 1)
    (hvarAll :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p i * s i)
          - Real.log (∑ i, q i * Real.exp (t * s i))
          ≤ finiteKL p q)
    (hgain_add : Fbefore + finiteKL p q ≤ Fafter) :
    (l1Norm (fun i => p i - q i)) ^ 2 / 2 ≤ Fafter - Fbefore := by
  have hgain : finiteKL p q ≤ Fafter - Fbefore := by
    linarith
  exact halfStepAscent_of_probabilityVariationalMathlibPinsker_klGain
    (p := p) (q := q) hq_nonneg hq_mass hvarAll hgain

/-- Sequence form of `halfStepAscent_of_probabilityVariationalMathlibPinsker_klGain_add`. -/
theorem halfStepAscent_seq_of_probabilityVariationalMathlibPinsker_klGain_add
    {n : ℕ} {p q : ℕ → Fin n → ℝ} {Fbefore Fafter : ℕ → ℝ}
    (hq_nonneg : ∀ k i, 0 ≤ q k i)
    (hq_mass : ∀ k, ∑ i, q k i = 1)
    (hvarAll :
      ∀ k : ℕ, ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p k i * s i)
          - Real.log (∑ i, q k i * Real.exp (t * s i))
          ≤ finiteKL (p k) (q k))
    (hgain_add : ∀ k : ℕ, Fbefore k + finiteKL (p k) (q k) ≤ Fafter k) :
    ∀ k : ℕ,
      (l1Norm (fun i => p k i - q k i)) ^ 2 / 2
        ≤ Fafter k - Fbefore k := by
  intro k
  exact halfStepAscent_of_probabilityVariationalMathlibPinsker_klGain_add
    (p := p k) (q := q k) (hq_nonneg k) (hq_mass k) (hvarAll k) (hgain_add k)

/--
Full-sweep probability-coordinate A.1 endpoint with both KL-gain premises additive.
-/
theorem perStepAscent_residualProxy_of_probabilityVariationalMathlibPinsker_klGains_add
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ}
    (hq1_nonneg : ∀ k i, 0 ≤ q1 k i)
    (hq1_mass : ∀ k, ∑ i, q1 k i = 1)
    (hvarAll1 :
      ∀ k : ℕ, ∀ s : Fin n₁ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p1 k i * s i)
          - Real.log (∑ i, q1 k i * Real.exp (t * s i))
          ≤ finiteKL (p1 k) (q1 k))
    (hgain1_add : ∀ k : ℕ, F k + finiteKL (p1 k) (q1 k) ≤ Fhalf k)
    (hq2_nonneg : ∀ k i, 0 ≤ q2 k i)
    (hq2_mass : ∀ k, ∑ i, q2 k i = 1)
    (hvarAll2 :
      ∀ k : ℕ, ∀ s : Fin n₂ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p2 k i * s i)
          - Real.log (∑ i, q2 k i * Real.exp (t * s i))
          ≤ finiteKL (p2 k) (q2 k))
    (hgain2_add : ∀ k : ℕ, Fhalf k + finiteKL (p2 k) (q2 k) ≤ F (k + 1)) :
    ∀ k : ℕ,
      (1 / 2) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k := by
  exact perStepAscent_residualProxy_of_probabilityVariationalMathlibPinsker_klGains
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    hq1_nonneg hq1_mass hvarAll1
    (by
      intro k
      linarith [hgain1_add k])
    hq2_nonneg hq2_mass hvarAll2
    (by
      intro k
      linarith [hgain2_add k])

/--
Half-step A.1 endpoint in paper-native unit-mass variational form and additive gain form.
-/
theorem halfStepAscent_of_probabilityMassShellVariationalMathlibPinsker_klGain_add
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter : ℝ}
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = 1)
    (hvarMass :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p i * s i)
          - 1 * Real.log (∑ i, (q i / 1) * Real.exp (t * s i))
          ≤ finiteKL p q)
    (hgain_add : Fbefore + finiteKL p q ≤ Fafter) :
    (l1Norm (fun i => p i - q i)) ^ 2 / 2 ≤ Fafter - Fbefore := by
  have hgain : finiteKL p q ≤ Fafter - Fbefore := by
    linarith
  exact halfStepAscent_of_probabilityMassShellVariationalMathlibPinsker_klGain
    (p := p) (q := q) hq_nonneg hq_mass hvarMass hgain

/-- Sequence form of `halfStepAscent_of_probabilityMassShellVariationalMathlibPinsker_klGain_add`. -/
theorem halfStepAscent_seq_of_probabilityMassShellVariationalMathlibPinsker_klGain_add
    {n : ℕ} {p q : ℕ → Fin n → ℝ} {Fbefore Fafter : ℕ → ℝ}
    (hq_nonneg : ∀ k i, 0 ≤ q k i)
    (hq_mass : ∀ k, ∑ i, q k i = 1)
    (hvarMass :
      ∀ k : ℕ, ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p k i * s i)
          - 1 * Real.log (∑ i, (q k i / 1) * Real.exp (t * s i))
          ≤ finiteKL (p k) (q k))
    (hgain_add : ∀ k : ℕ, Fbefore k + finiteKL (p k) (q k) ≤ Fafter k) :
    ∀ k : ℕ,
      (l1Norm (fun i => p k i - q k i)) ^ 2 / 2
        ≤ Fafter k - Fbefore k := by
  intro k
  exact halfStepAscent_of_probabilityMassShellVariationalMathlibPinsker_klGain_add
    (p := p k) (q := q k) (hq_nonneg k) (hq_mass k) (hvarMass k) (hgain_add k)

/--
Full-sweep paper-native unit-mass A.1 endpoint with both KL-gain premises additive.
-/
theorem perStepAscent_residualProxy_of_probabilityMassShellVariationalMathlibPinsker_klGains_add
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ}
    (hq1_nonneg : ∀ k i, 0 ≤ q1 k i)
    (hq1_mass : ∀ k, ∑ i, q1 k i = 1)
    (hvarMass1 :
      ∀ k : ℕ, ∀ s : Fin n₁ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p1 k i * s i)
          - 1 * Real.log (∑ i, (q1 k i / 1) * Real.exp (t * s i))
          ≤ finiteKL (p1 k) (q1 k))
    (hgain1_add : ∀ k : ℕ, F k + finiteKL (p1 k) (q1 k) ≤ Fhalf k)
    (hq2_nonneg : ∀ k i, 0 ≤ q2 k i)
    (hq2_mass : ∀ k, ∑ i, q2 k i = 1)
    (hvarMass2 :
      ∀ k : ℕ, ∀ s : Fin n₂ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p2 k i * s i)
          - 1 * Real.log (∑ i, (q2 k i / 1) * Real.exp (t * s i))
          ≤ finiteKL (p2 k) (q2 k))
    (hgain2_add : ∀ k : ℕ, Fhalf k + finiteKL (p2 k) (q2 k) ≤ F (k + 1)) :
    ∀ k : ℕ,
      (1 / 2) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k := by
  exact perStepAscent_residualProxy_of_probabilityMassShellVariationalMathlibPinsker_klGains
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    hq1_nonneg hq1_mass hvarMass1
    (by
      intro k
      linarith [hgain1_add k])
    hq2_nonneg hq2_mass hvarMass2
    (by
      intro k
      linarith [hgain2_add k])

/--
Full-sweep common-mass A.1 endpoint with additive KL-gain premises.

This is the paper-facing mass-shell route with mathlib-Hoeffding Pinsker, but callers may state
both block optimality gains as `F + KL ≤ Fafter`.
-/
theorem perStepAscent_residualProxy_of_massShellVariationalMathlibPinsker_klGains_commonMass_add
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq1_nonneg : ∀ k i, 0 ≤ q1 k i)
    (hq1_mass : ∀ k, ∑ i, q1 k i = M)
    (hvarMass1 :
      ∀ k : ℕ, ∀ s : Fin n₁ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p1 k i * s i)
          - M * Real.log (∑ i, (q1 k i / M) * Real.exp (t * s i))
          ≤ finiteKL (p1 k) (q1 k))
    (hgain1_add : ∀ k : ℕ, F k + finiteKL (p1 k) (q1 k) ≤ Fhalf k)
    (hq2_nonneg : ∀ k i, 0 ≤ q2 k i)
    (hq2_mass : ∀ k, ∑ i, q2 k i = M)
    (hvarMass2 :
      ∀ k : ℕ, ∀ s : Fin n₂ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p2 k i * s i)
          - M * Real.log (∑ i, (q2 k i / M) * Real.exp (t * s i))
          ≤ finiteKL (p2 k) (q2 k))
    (hgain2_add : ∀ k : ℕ, Fhalf k + finiteKL (p2 k) (q2 k) ≤ F (k + 1)) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k := by
  exact perStepAscent_residualProxy_of_finiteVariationalMathlibPinsker_klGains_commonMass
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hMpos hq1_nonneg hq1_mass
    (fun k =>
      finite_variational_normalized_of_massShell_variational
        (p := p1 k) (q := q1 k) hMpos (hvarMass1 k))
    (by
      intro k
      linarith [hgain1_add k])
    hq2_nonneg hq2_mass
    (fun k =>
      finite_variational_normalized_of_massShell_variational
        (p := p2 k) (q := q2 k) hMpos (hvarMass2 k))
    (by
      intro k
      linarith [hgain2_add k])

/--
Full-sweep A.1 endpoint from paper-native common-mass variational input, the one-parameter
strict-interior centered Bernoulli Hoeffding inequality, and KL gains.

This version no longer requires callers to pre-normalize the finite variational inequality by
the common mass `M`; Lean performs that normalization internally before applying the A.3
mass-shell Pinsker endpoint to the two blocks.
-/
theorem perStepAscent_residualProxy_of_massShellVariationalCenteredBernoulliUnitPinsker_klGains_commonMass
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hcenteredUnit :
      ∀ a t : ℝ, 0 < a → a < 1 → 0 < t →
        a * Real.exp (2 * (1 - a) * t) + (1 - a) * Real.exp (-2 * a * t)
          ≤ Real.exp (t ^ 2 / 2))
    (hq1_nonneg : ∀ k i, 0 ≤ q1 k i)
    (hq1_mass : ∀ k, ∑ i, q1 k i = M)
    (hvarMass1 :
      ∀ k : ℕ, ∀ s : Fin n₁ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p1 k i * s i)
          - M * Real.log (∑ i, (q1 k i / M) * Real.exp (t * s i))
          ≤ finiteKL (p1 k) (q1 k))
    (hgain1 : ∀ k : ℕ, finiteKL (p1 k) (q1 k) ≤ Fhalf k - F k)
    (hq2_nonneg : ∀ k i, 0 ≤ q2 k i)
    (hq2_mass : ∀ k, ∑ i, q2 k i = M)
    (hvarMass2 :
      ∀ k : ℕ, ∀ s : Fin n₂ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p2 k i * s i)
          - M * Real.log (∑ i, (q2 k i / M) * Real.exp (t * s i))
          ≤ finiteKL (p2 k) (q2 k))
    (hgain2 : ∀ k : ℕ, finiteKL (p2 k) (q2 k) ≤ F (k + 1) - Fhalf k) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k := by
  exact perStepAscent_residualProxy_of_finiteVariationalCenteredBernoulliUnitPinsker_klGains_commonMass
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hMpos hcenteredUnit
    hq1_nonneg hq1_mass
    (fun k =>
      finite_variational_normalized_of_massShell_variational
        (p := p1 k) (q := q1 k) hMpos (hvarMass1 k))
    hgain1
    hq2_nonneg hq2_mass
    (fun k =>
      finite_variational_normalized_of_massShell_variational
        (p := p2 k) (q := q2 k) hMpos (hvarMass2 k))
    hgain2

/--
Full-sweep A.1 endpoint with scalar Hoeffding discharged by mathlib.

Compared with
`perStepAscent_residualProxy_of_massShellVariationalCenteredBernoulliUnitPinsker_klGains_commonMass`,
this theorem no longer exposes the one-parameter Bernoulli Hoeffding inequality as an input.
-/
theorem perStepAscent_residualProxy_of_massShellVariationalMathlibPinsker_klGains_commonMass
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq1_nonneg : ∀ k i, 0 ≤ q1 k i)
    (hq1_mass : ∀ k, ∑ i, q1 k i = M)
    (hvarMass1 :
      ∀ k : ℕ, ∀ s : Fin n₁ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p1 k i * s i)
          - M * Real.log (∑ i, (q1 k i / M) * Real.exp (t * s i))
          ≤ finiteKL (p1 k) (q1 k))
    (hgain1 : ∀ k : ℕ, finiteKL (p1 k) (q1 k) ≤ Fhalf k - F k)
    (hq2_nonneg : ∀ k i, 0 ≤ q2 k i)
    (hq2_mass : ∀ k, ∑ i, q2 k i = M)
    (hvarMass2 :
      ∀ k : ℕ, ∀ s : Fin n₂ → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p2 k i * s i)
          - M * Real.log (∑ i, (q2 k i / M) * Real.exp (t * s i))
          ≤ finiteKL (p2 k) (q2 k))
    (hgain2 : ∀ k : ℕ, finiteKL (p2 k) (q2 k) ≤ F (k + 1) - Fhalf k) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k := by
  exact perStepAscent_residualProxy_of_finiteVariationalMathlibPinsker_klGains_commonMass
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hMpos hq1_nonneg hq1_mass
    (fun k =>
      finite_variational_normalized_of_massShell_variational
        (p := p1 k) (q := q1 k) hMpos (hvarMass1 k))
    hgain1 hq2_nonneg hq2_mass
    (fun k =>
      finite_variational_normalized_of_massShell_variational
        (p := p2 k) (q := q2 k) hMpos (hvarMass2 k))
    hgain2

/--
One-block finite mass-shell KL-projection optimality package for Lemma A.1.

This is intentionally still an inequality-shaped predicate rather than a minimizer structure:
concrete block-update modules can prove it directly from their KL projection optimality
conditions.  It packages the exact data consumed by the A.1 bridge: the updated block lies on the
common mass shell, the all-test finite KL variational inequality holds in paper-native
non-normalized form, and the block update gives additive objective ascent by the finite KL gain.
-/
def FiniteMassShellKLProjectionOptimality
    {n : ℕ} (p q : Fin n → ℝ) (M Fbefore Fafter : ℝ) : Prop :=
  (∀ i, 0 ≤ q i) ∧
  (∑ i, q i = M) ∧
  (∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
    t * (∑ i, p i * s i)
      - M * Real.log (∑ i, (q i / M) * Real.exp (t * s i))
      ≤ finiteKL p q) ∧
  Fbefore + finiteKL p q ≤ Fafter

/--
Primitive variational part of one finite mass-shell block update.

This separates the actual KL-projection optimality inequality from the objective-ascent
bookkeeping.  It is closer to the paper statement: the updated block is feasible for the common
mass shell, and satisfies the all-test variational inequality in non-normalized coordinates.
-/
def FiniteMassShellVariationalOptimality
    {n : ℕ} (p q : Fin n → ℝ) (M : ℝ) : Prop :=
  (∀ i, 0 ≤ q i) ∧
  (∑ i, q i = M) ∧
  ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
    t * (∑ i, p i * s i)
      - M * Real.log (∑ i, (q i / M) * Real.exp (t * s i))
      ≤ finiteKL p q

/--
Primitive objective-gain part of one finite block update.

This records the block-update ascent statement separately from the variational optimality of the
projection.  For concrete algorithms, this is the place where the dual-objective identity for the
block update should land.
-/
def FiniteKLGainFromBlockUpdate
    {n : ℕ} (p q : Fin n → ℝ) (Fbefore Fafter : ℝ) : Prop :=
  Fbefore + finiteKL p q ≤ Fafter

/--
Build the primitive finite mass-shell variational optimality predicate from its three concrete
facts.
-/
theorem finiteMassShellVariationalOptimality_of_primitives
    {n : ℕ} {p q : Fin n → ℝ} {M : ℝ}
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = M)
    (hvar :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p i * s i)
          - M * Real.log (∑ i, (q i / M) * Real.exp (t * s i))
          ≤ finiteKL p q) :
    FiniteMassShellVariationalOptimality p q M :=
  ⟨hq_nonneg, hq_mass, hvar⟩

/--
Extract the nonnegativity part of primitive finite mass-shell variational optimality.
-/
theorem finiteMassShellVariationalOptimality_nonneg
    {n : ℕ} {p q : Fin n → ℝ} {M : ℝ}
    (hvar : FiniteMassShellVariationalOptimality p q M) :
    ∀ i, 0 ≤ q i :=
  hvar.1

/--
Extract the common-mass part of primitive finite mass-shell variational optimality.
-/
theorem finiteMassShellVariationalOptimality_mass
    {n : ℕ} {p q : Fin n → ℝ} {M : ℝ}
    (hvar : FiniteMassShellVariationalOptimality p q M) :
    ∑ i, q i = M :=
  hvar.2.1

/--
Extract the all-test variational inequality from primitive finite mass-shell variational
optimality.
-/
theorem finiteMassShellVariationalOptimality_variational
    {n : ℕ} {p q : Fin n → ℝ} {M : ℝ}
    (hvar : FiniteMassShellVariationalOptimality p q M) :
    ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
      t * (∑ i, p i * s i)
        - M * Real.log (∑ i, (q i / M) * Real.exp (t * s i))
        ≤ finiteKL p q :=
  hvar.2.2

/--
Build mass-shell variational optimality from the concrete finite-measure KL variational theorem.

This is the current strongest non-wrapper bridge into the A.1 primitive: if the source block is a
nonnegative finite vector of mass `M` and the updated/reference block is strictly positive with the
same mass, Lean derives the all-test variational inequality by constructing the normalized finite
probability measures and computing mathlib's `klDiv` back to `finiteKL`.

Future concrete block-update proofs should use this theorem when they can prove finite feasibility
and strict positivity of the updated block, instead of postulating
`FiniteMassShellVariationalOptimality` directly.
-/
theorem finiteMassShellVariationalOptimality_of_finiteProbabilityMeasure_klDiv_computed
    {n : ℕ} {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = M)
    (hq_pos : ∀ i, 0 < q i)
    (hq_mass : ∑ i, q i = M) :
    FiniteMassShellVariationalOptimality p q M :=
  finiteMassShellVariationalOptimality_of_primitives
    (fun i => (hq_pos i).le)
    hq_mass
    (finite_variational_massShell_of_finiteProbabilityMeasure_klDiv_computed
      (p := p) (q := q) hMpos hp_nonneg hp_mass hq_pos hq_mass)

/--
Support-aware mass-shell variational optimality from the concrete finite-measure KL theorem.

This boundary variant replaces strict positivity of the updated/reference block by
nonnegativity plus support domination `q_i = 0 → p_i = 0`.  It is the right interface for
block updates that may produce zeros but remain absolutely continuous with respect to the
reference block.
-/
theorem finiteMassShellVariationalOptimality_of_finiteProbabilityMeasure_klDiv_computed_of_support
    {n : ℕ} {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = M)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = M)
    (hsupp : ∀ i, q i = 0 → p i = 0) :
    FiniteMassShellVariationalOptimality p q M :=
  finiteMassShellVariationalOptimality_of_primitives
    hq_nonneg
    hq_mass
    (finite_variational_massShell_of_finiteProbabilityMeasure_klDiv_computed_of_support
      (p := p) (q := q) hMpos hp_nonneg hp_mass hq_nonneg hq_mass hsupp)

/--
Extract the additive KL-gain statement from primitive block-update gain.
-/
theorem finiteKLGainFromBlockUpdate_gain_add
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter : ℝ}
    (hgain : FiniteKLGainFromBlockUpdate p q Fbefore Fafter) :
    Fbefore + finiteKL p q ≤ Fafter :=
  hgain

/--
Algorithm-facing finite mass-shell block-update certificate.

There is currently no concrete KL-minimizer/argmin definition in `Sweep.lean` or the setup layer.
This structure is therefore deliberately minimal and operational: a future block-update proof must
show that the pre-update block `p` is a nonnegative vector of mass `M`, the updated block `q` is
strictly positive with the same mass, and the concrete dual-objective calculation gives the KL
gain `Fbefore + KL(p‖q) ≤ Fafter`.

The variational optimality part is *not* a field: it is derived below from the finite-measure KL
variational theorem.  Thus this certificate records exactly the algorithm-facing facts that remain
after the measure-theoretic variational step has been internalized.
-/
structure FiniteMassShellBlockUpdateCertificate
    {n : ℕ} (p q : Fin n → ℝ) (M Fbefore Fafter : ℝ) : Prop where
  source_nonneg : ∀ i, 0 ≤ p i
  source_mass : ∑ i, p i = M
  update_pos : ∀ i, 0 < q i
  update_mass : ∑ i, q i = M
  gain : FiniteKLGainFromBlockUpdate p q Fbefore Fafter

/--
A finite mass-shell block-update certificate derives the variational primitive internally.
-/
theorem finiteMassShellVariationalOptimality_of_blockUpdateCertificate
    {n : ℕ} {p q : Fin n → ℝ} {M Fbefore Fafter : ℝ}
    (hMpos : 0 < M)
    (hcert : FiniteMassShellBlockUpdateCertificate p q M Fbefore Fafter) :
    FiniteMassShellVariationalOptimality p q M :=
  finiteMassShellVariationalOptimality_of_finiteProbabilityMeasure_klDiv_computed
    (p := p) (q := q) hMpos
    hcert.source_nonneg hcert.source_mass hcert.update_pos hcert.update_mass

/--
Extract the primitive additive KL gain from a finite mass-shell block-update certificate.
-/
theorem finiteKLGainFromBlockUpdate_of_blockUpdateCertificate
    {n : ℕ} {p q : Fin n → ℝ} {M Fbefore Fafter : ℝ}
    (hcert : FiniteMassShellBlockUpdateCertificate p q M Fbefore Fafter) :
    FiniteKLGainFromBlockUpdate p q Fbefore Fafter :=
  hcert.gain

/--
Promote an algorithm-facing finite block-update certificate to the existing KL-projection
optimality package.
-/
theorem finiteMassShellKLProjectionOptimality_of_blockUpdateCertificate
    {n : ℕ} {p q : Fin n → ℝ} {M Fbefore Fafter : ℝ}
    (hMpos : 0 < M)
    (hcert : FiniteMassShellBlockUpdateCertificate p q M Fbefore Fafter) :
    FiniteMassShellKLProjectionOptimality p q M Fbefore Fafter :=
  ⟨fun i => (hcert.update_pos i).le,
    hcert.update_mass,
    finiteMassShellVariationalOptimality_variational
      (finiteMassShellVariationalOptimality_of_blockUpdateCertificate hMpos hcert),
    finiteKLGainFromBlockUpdate_gain_add hcert.gain⟩

/--
Support-aware finite mass-shell block-update certificate.

This is the boundary analogue of `FiniteMassShellBlockUpdateCertificate`.  It does not require
strict positivity of `q`; instead it records the exact finite absolute-continuity condition
needed by the computed KL variational theorem.
-/
structure FiniteMassShellSupportBlockUpdateCertificate
    {n : ℕ} (p q : Fin n → ℝ) (M Fbefore Fafter : ℝ) : Prop where
  source_nonneg : ∀ i, 0 ≤ p i
  source_mass : ∑ i, p i = M
  update_nonneg : ∀ i, 0 ≤ q i
  update_mass : ∑ i, q i = M
  support : ∀ i, q i = 0 → p i = 0
  gain : FiniteKLGainFromBlockUpdate p q Fbefore Fafter

/--
A support-aware finite block-update certificate derives the variational primitive internally.
-/
theorem finiteMassShellVariationalOptimality_of_supportBlockUpdateCertificate
    {n : ℕ} {p q : Fin n → ℝ} {M Fbefore Fafter : ℝ}
    (hMpos : 0 < M)
    (hcert : FiniteMassShellSupportBlockUpdateCertificate p q M Fbefore Fafter) :
    FiniteMassShellVariationalOptimality p q M :=
  finiteMassShellVariationalOptimality_of_finiteProbabilityMeasure_klDiv_computed_of_support
    (p := p) (q := q) hMpos
    hcert.source_nonneg hcert.source_mass
    hcert.update_nonneg hcert.update_mass hcert.support

/--
Extract the primitive additive KL gain from a support-aware block-update certificate.
-/
theorem finiteKLGainFromBlockUpdate_of_supportBlockUpdateCertificate
    {n : ℕ} {p q : Fin n → ℝ} {M Fbefore Fafter : ℝ}
    (hcert : FiniteMassShellSupportBlockUpdateCertificate p q M Fbefore Fafter) :
    FiniteKLGainFromBlockUpdate p q Fbefore Fafter :=
  hcert.gain

/--
Promote a support-aware finite block-update certificate to the KL-projection optimality package.
-/
theorem finiteMassShellKLProjectionOptimality_of_supportBlockUpdateCertificate
    {n : ℕ} {p q : Fin n → ℝ} {M Fbefore Fafter : ℝ}
    (hMpos : 0 < M)
    (hcert : FiniteMassShellSupportBlockUpdateCertificate p q M Fbefore Fafter) :
    FiniteMassShellKLProjectionOptimality p q M Fbefore Fafter :=
  ⟨hcert.update_nonneg,
    hcert.update_mass,
    finiteMassShellVariationalOptimality_variational
      (finiteMassShellVariationalOptimality_of_supportBlockUpdateCertificate hMpos hcert),
    finiteKLGainFromBlockUpdate_gain_add hcert.gain⟩

/--
Assemble the current finite mass-shell KL-projection optimality package from the separated
paper-shaped primitives.
-/
theorem finiteMassShellKLProjectionOptimality_of_variationalOptimality_blockUpdateGain
    {n : ℕ} {p q : Fin n → ℝ} {M Fbefore Fafter : ℝ}
    (hvar : FiniteMassShellVariationalOptimality p q M)
    (hgain : FiniteKLGainFromBlockUpdate p q Fbefore Fafter) :
    FiniteMassShellKLProjectionOptimality p q M Fbefore Fafter :=
  ⟨finiteMassShellVariationalOptimality_nonneg hvar,
    finiteMassShellVariationalOptimality_mass hvar,
    finiteMassShellVariationalOptimality_variational hvar,
    finiteKLGainFromBlockUpdate_gain_add hgain⟩

/--
Assemble the current finite mass-shell KL-projection optimality package directly from concrete
primitive facts.
-/
theorem finiteMassShellKLProjectionOptimality_of_primitives
    {n : ℕ} {p q : Fin n → ℝ} {M Fbefore Fafter : ℝ}
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = M)
    (hvar :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p i * s i)
          - M * Real.log (∑ i, (q i / M) * Real.exp (t * s i))
          ≤ finiteKL p q)
    (hgain_add : Fbefore + finiteKL p q ≤ Fafter) :
    FiniteMassShellKLProjectionOptimality p q M Fbefore Fafter :=
  finiteMassShellKLProjectionOptimality_of_variationalOptimality_blockUpdateGain
    (finiteMassShellVariationalOptimality_of_primitives hq_nonneg hq_mass hvar)
    hgain_add

/--
The packaged optimality predicate is exactly the conjunction of primitive variational optimality
and primitive block-update KL gain.
-/
theorem finiteMassShellKLProjectionOptimality_iff_variationalOptimality_and_blockUpdateGain
    {n : ℕ} {p q : Fin n → ℝ} {M Fbefore Fafter : ℝ} :
    FiniteMassShellKLProjectionOptimality p q M Fbefore Fafter ↔
      FiniteMassShellVariationalOptimality p q M ∧
        FiniteKLGainFromBlockUpdate p q Fbefore Fafter := by
  constructor
  · intro hopt
    exact
      ⟨finiteMassShellVariationalOptimality_of_primitives
          hopt.1
          hopt.2.1
          hopt.2.2.1,
        hopt.2.2.2⟩
  · intro h
    exact finiteMassShellKLProjectionOptimality_of_variationalOptimality_blockUpdateGain h.1 h.2

/--
Extract the nonnegativity part of a finite mass-shell KL-projection optimality package.
-/
theorem finiteMassShellKLProjectionOptimality_nonneg
    {n : ℕ} {p q : Fin n → ℝ} {M Fbefore Fafter : ℝ}
    (hopt : FiniteMassShellKLProjectionOptimality p q M Fbefore Fafter) :
    ∀ i, 0 ≤ q i :=
  hopt.1

/--
Extract the common-mass part of a finite mass-shell KL-projection optimality package.
-/
theorem finiteMassShellKLProjectionOptimality_mass
    {n : ℕ} {p q : Fin n → ℝ} {M Fbefore Fafter : ℝ}
    (hopt : FiniteMassShellKLProjectionOptimality p q M Fbefore Fafter) :
    ∑ i, q i = M :=
  hopt.2.1

/--
Extract the paper-native all-test variational inequality from a finite mass-shell
KL-projection optimality package.
-/
theorem finiteMassShellKLProjectionOptimality_variational
    {n : ℕ} {p q : Fin n → ℝ} {M Fbefore Fafter : ℝ}
    (hopt : FiniteMassShellKLProjectionOptimality p q M Fbefore Fafter) :
    ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
      t * (∑ i, p i * s i)
        - M * Real.log (∑ i, (q i / M) * Real.exp (t * s i))
        ≤ finiteKL p q :=
  hopt.2.2.1

/--
Extract the additive KL-gain/objective-ascent inequality from a finite mass-shell
KL-projection optimality package.
-/
theorem finiteMassShellKLProjectionOptimality_gain_add
    {n : ℕ} {p q : Fin n → ℝ} {M Fbefore Fafter : ℝ}
    (hopt : FiniteMassShellKLProjectionOptimality p q M Fbefore Fafter) :
    Fbefore + finiteKL p q ≤ Fafter :=
  hopt.2.2.2

/--
Single-block A.1 bridge from finite KL-projection optimality-shaped hypotheses.

The proof step mirrors the paper: projection optimality supplies the common-mass variational
inequality and additive KL gain; Lean normalizes the variational inequality, invokes the
mathlib-Hoeffding Pinsker endpoint, and converts the additive gain to half-step ascent.
-/
theorem halfStepAscent_of_finiteMassShellKLProjectionOptimality_commonMass
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter M : ℝ}
    (hMpos : 0 < M)
    (hopt : FiniteMassShellKLProjectionOptimality p q M Fbefore Fafter) :
    (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) ≤ Fafter - Fbefore := by
  exact halfStepAscent_of_finiteVariationalMathlibPinsker_klGain
    (p := p) (q := q) (M := M) hMpos
    (finiteMassShellKLProjectionOptimality_nonneg hopt)
    (finiteMassShellKLProjectionOptimality_mass hopt)
    (by
      have hvarMass := finiteMassShellKLProjectionOptimality_variational hopt
      exact finite_variational_normalized_of_massShell_variational
        (p := p) (q := q) hMpos hvarMass)
    (klGain_of_klGain_add (finiteMassShellKLProjectionOptimality_gain_add hopt))

/--
Sequence form of `halfStepAscent_of_finiteMassShellKLProjectionOptimality_commonMass`.
-/
theorem halfStepAscent_seq_of_finiteMassShellKLProjectionOptimality_commonMass
    {n : ℕ} {p q : ℕ → Fin n → ℝ} {Fbefore Fafter : ℕ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hopt :
      ∀ k : ℕ,
        FiniteMassShellKLProjectionOptimality (p k) (q k) M (Fbefore k) (Fafter k)) :
    ∀ k : ℕ,
      (l1Norm (fun i => p k i - q k i)) ^ 2 / (2 * M)
        ≤ Fafter k - Fbefore k := by
  intro k
  exact halfStepAscent_of_finiteMassShellKLProjectionOptimality_commonMass
    (p := p k) (q := q k) (M := M) hMpos (hopt k)

/--
Full-sweep A.1 bridge from two block-update KL-projection optimality packages.

This is the concrete bridge targeted by Lemma A.1 internalization: each block update provides the
paper-native common-mass variational inequality and additive KL-gain ascent in one optimality
package, and the theorem feeds those projections into the existing common-mass A.1 endpoint.
-/
theorem perStepAscent_residualProxy_of_finiteMassShellKLProjectionOptimality_commonMass
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hopt1 :
      ∀ k : ℕ,
        FiniteMassShellKLProjectionOptimality (p1 k) (q1 k) M (F k) (Fhalf k))
    (hopt2 :
      ∀ k : ℕ,
        FiniteMassShellKLProjectionOptimality (p2 k) (q2 k) M (Fhalf k) (F (k + 1))) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k := by
  exact perStepAscent_residualProxy_of_massShellVariationalMathlibPinsker_klGains_commonMass_add
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hMpos
    (fun k => finiteMassShellKLProjectionOptimality_nonneg (hopt1 k))
    (fun k => finiteMassShellKLProjectionOptimality_mass (hopt1 k))
    (fun k => finiteMassShellKLProjectionOptimality_variational (hopt1 k))
    (fun k => finiteMassShellKLProjectionOptimality_gain_add (hopt1 k))
    (fun k => finiteMassShellKLProjectionOptimality_nonneg (hopt2 k))
    (fun k => finiteMassShellKLProjectionOptimality_mass (hopt2 k))
    (fun k => finiteMassShellKLProjectionOptimality_variational (hopt2 k))
    (fun k => finiteMassShellKLProjectionOptimality_gain_add (hopt2 k))

/--
Single-block A.1 bridge from separated primitive variational optimality and block-update gain.

Compared with `halfStepAscent_of_finiteMassShellKLProjectionOptimality_commonMass`, callers no
longer need to build the bundled `FiniteMassShellKLProjectionOptimality` package first.
-/
theorem halfStepAscent_of_finiteMassShellVariationalOptimality_blockUpdateGain_commonMass
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter M : ℝ}
    (hMpos : 0 < M)
    (hvar : FiniteMassShellVariationalOptimality p q M)
    (hgain : FiniteKLGainFromBlockUpdate p q Fbefore Fafter) :
    (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) ≤ Fafter - Fbefore := by
  exact halfStepAscent_of_finiteVariationalMathlibPinsker_klGain
    (p := p) (q := q) (M := M) hMpos
    (finiteMassShellVariationalOptimality_nonneg hvar)
    (finiteMassShellVariationalOptimality_mass hvar)
    (by
      exact finite_variational_normalized_of_massShell_variational
        (p := p) (q := q) hMpos
        (finiteMassShellVariationalOptimality_variational hvar))
    (klGain_of_klGain_add (finiteKLGainFromBlockUpdate_gain_add hgain))

/--
Sequence form of
`halfStepAscent_of_finiteMassShellVariationalOptimality_blockUpdateGain_commonMass`.
-/
theorem halfStepAscent_seq_of_finiteMassShellVariationalOptimality_blockUpdateGain_commonMass
    {n : ℕ} {p q : ℕ → Fin n → ℝ} {Fbefore Fafter : ℕ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hvar : ∀ k : ℕ, FiniteMassShellVariationalOptimality (p k) (q k) M)
    (hgain :
      ∀ k : ℕ, FiniteKLGainFromBlockUpdate (p k) (q k) (Fbefore k) (Fafter k)) :
    ∀ k : ℕ,
      (l1Norm (fun i => p k i - q k i)) ^ 2 / (2 * M)
        ≤ Fafter k - Fbefore k := by
  intro k
  exact halfStepAscent_of_finiteMassShellVariationalOptimality_blockUpdateGain_commonMass
    (p := p k) (q := q k) (M := M) hMpos (hvar k) (hgain k)

/--
Full-sweep A.1 bridge from separated primitive variational optimality and block-update gains.

This is the stronger paper-facing bridge for concrete block updates: each block supplies its
mass-shell variational optimality and its additive KL gain separately, and the theorem derives the
same residual-ascent inequality as the older packaged bridge.
-/
theorem perStepAscent_residualProxy_of_finiteMassShellVariationalOptimality_blockUpdateGains_commonMass
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hvar1 : ∀ k : ℕ, FiniteMassShellVariationalOptimality (p1 k) (q1 k) M)
    (hgain1 :
      ∀ k : ℕ, FiniteKLGainFromBlockUpdate (p1 k) (q1 k) (F k) (Fhalf k))
    (hvar2 : ∀ k : ℕ, FiniteMassShellVariationalOptimality (p2 k) (q2 k) M)
    (hgain2 :
      ∀ k : ℕ, FiniteKLGainFromBlockUpdate (p2 k) (q2 k) (Fhalf k) (F (k + 1))) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k := by
  exact perStepAscent_residualProxy_of_massShellVariationalMathlibPinsker_klGains_commonMass_add
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hMpos
    (fun k => finiteMassShellVariationalOptimality_nonneg (hvar1 k))
    (fun k => finiteMassShellVariationalOptimality_mass (hvar1 k))
    (fun k => finiteMassShellVariationalOptimality_variational (hvar1 k))
    (fun k => finiteKLGainFromBlockUpdate_gain_add (hgain1 k))
    (fun k => finiteMassShellVariationalOptimality_nonneg (hvar2 k))
    (fun k => finiteMassShellVariationalOptimality_mass (hvar2 k))
    (fun k => finiteMassShellVariationalOptimality_variational (hvar2 k))
    (fun k => finiteKLGainFromBlockUpdate_gain_add (hgain2 k))

/--
Difference-form characterization of primitive block-update KL gain.

This is the exact algebraic shape used by half-step ascent lemmas, while
`FiniteKLGainFromBlockUpdate` keeps the projection-update statement in the additive form that
usually comes out of objective identities.
-/
theorem finiteKLGainFromBlockUpdate_iff_kl_le_objectiveDiff
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter : ℝ} :
    FiniteKLGainFromBlockUpdate p q Fbefore Fafter ↔
      finiteKL p q ≤ Fafter - Fbefore := by
  constructor
  · intro hgain
    linarith [finiteKLGainFromBlockUpdate_gain_add hgain]
  · intro hgain
    change Fbefore + finiteKL p q ≤ Fafter
    linarith

/--
Build primitive block-update KL gain from the objective-difference form.
-/
theorem finiteKLGainFromBlockUpdate_of_kl_le_objectiveDiff
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter : ℝ}
    (hgain : finiteKL p q ≤ Fafter - Fbefore) :
    FiniteKLGainFromBlockUpdate p q Fbefore Fafter :=
  (finiteKLGainFromBlockUpdate_iff_kl_le_objectiveDiff).2 hgain

/--
Exact objective-difference form for a finite block update.

Concrete block-update calculations often produce an identity before it is weakened to an
inequality.  Keeping this separate lets downstream code preserve the sharper computation.
-/
def FiniteKLExactGainFromBlockUpdate
    {n : ℕ} (p q : Fin n → ℝ) (Fbefore Fafter : ℝ) : Prop :=
  Fafter - Fbefore = finiteKL p q

/--
Exact difference gain implies the primitive additive KL-gain predicate.
-/
theorem finiteKLGainFromBlockUpdate_of_exactGain
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter : ℝ}
    (hexact : FiniteKLExactGainFromBlockUpdate p q Fbefore Fafter) :
    FiniteKLGainFromBlockUpdate p q Fbefore Fafter :=
  finiteKLGainFromBlockUpdate_of_kl_le_objectiveDiff (le_of_eq hexact.symm)

/--
Exact additive gain implies the primitive additive KL-gain predicate.
-/
theorem finiteKLGainFromBlockUpdate_of_exactAddGain
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter : ℝ}
    (hexact : FiniteKLExactAddGainFromBlockUpdate p q Fbefore Fafter) :
    FiniteKLGainFromBlockUpdate p q Fbefore Fafter :=
  le_of_eq hexact.symm

/--
Exact additive and exact difference block-update identities are equivalent.
-/
theorem finiteKLExactAddGainFromBlockUpdate_iff_exactGain
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter : ℝ} :
    FiniteKLExactAddGainFromBlockUpdate p q Fbefore Fafter ↔
      FiniteKLExactGainFromBlockUpdate p q Fbefore Fafter := by
  constructor
  · intro h
    dsimp [FiniteKLExactAddGainFromBlockUpdate] at h
    dsimp [FiniteKLExactGainFromBlockUpdate]
    linarith
  · intro h
    dsimp [FiniteKLExactGainFromBlockUpdate] at h
    dsimp [FiniteKLExactAddGainFromBlockUpdate]
    linarith

/--
Exact-gain variant of the algorithm-facing mass-shell block-update certificate.

Concrete block-update algebra often proves the sharper identity
`Fafter = Fbefore + KL(p‖q)`.  This structure keeps that exact statement available and converts to
`FiniteMassShellBlockUpdateCertificate` only when an A.1 theorem needs the weaker monotone gain.
-/
structure FiniteMassShellExactBlockUpdateCertificate
    {n : ℕ} (p q : Fin n → ℝ) (M Fbefore Fafter : ℝ) : Prop where
  source_nonneg : ∀ i, 0 ≤ p i
  source_mass : ∑ i, p i = M
  update_pos : ∀ i, 0 < q i
  update_mass : ∑ i, q i = M
  exact_gain : FiniteKLExactAddGainFromBlockUpdate p q Fbefore Fafter

/--
Forget exact objective-gain equality to the inequality-shaped block-update certificate.
-/
theorem finiteMassShellBlockUpdateCertificate_of_exactBlockUpdateCertificate
    {n : ℕ} {p q : Fin n → ℝ} {M Fbefore Fafter : ℝ}
    (hcert : FiniteMassShellExactBlockUpdateCertificate p q M Fbefore Fafter) :
    FiniteMassShellBlockUpdateCertificate p q M Fbefore Fafter :=
  { source_nonneg := hcert.source_nonneg
    source_mass := hcert.source_mass
    update_pos := hcert.update_pos
    update_mass := hcert.update_mass
    gain := finiteKLGainFromBlockUpdate_of_exactAddGain hcert.exact_gain }

/--
Forget exact objective-gain equality to the inequality-shaped support-aware certificate.
-/
theorem finiteMassShellSupportBlockUpdateCertificate_of_exactSupportBlockUpdateCertificate
    {n : ℕ} {p q : Fin n → ℝ} {M Fbefore Fafter : ℝ}
    (hcert : FiniteMassShellExactSupportBlockUpdateCertificate p q M Fbefore Fafter) :
    FiniteMassShellSupportBlockUpdateCertificate p q M Fbefore Fafter :=
  { source_nonneg := hcert.source_nonneg
    source_mass := hcert.source_mass
    update_nonneg := hcert.update_nonneg
    update_mass := hcert.update_mass
    support := hcert.support
    gain := finiteKLGainFromBlockUpdate_of_exactAddGain hcert.exact_gain }

/--
Primitive variational optimality for one finite probability-simplex block update.

This is the normalized analogue of `FiniteMassShellVariationalOptimality`: the updated block has
unit mass, and the all-test variational inequality is stated directly in probability
coordinates.
-/
def FiniteProbabilityVariationalOptimality
    {n : ℕ} (p q : Fin n → ℝ) : Prop :=
  (∀ i, 0 ≤ q i) ∧
  (∑ i, q i = 1) ∧
  ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
    t * (∑ i, p i * s i)
      - Real.log (∑ i, q i * Real.exp (t * s i))
      ≤ finiteKL p q

/--
Build probability variational optimality from its concrete facts.
-/
theorem finiteProbabilityVariationalOptimality_of_primitives
    {n : ℕ} {p q : Fin n → ℝ}
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = 1)
    (hvar :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p i * s i)
          - Real.log (∑ i, q i * Real.exp (t * s i))
          ≤ finiteKL p q) :
    FiniteProbabilityVariationalOptimality p q :=
  ⟨hq_nonneg, hq_mass, hvar⟩

/--
Build probability-simplex variational optimality from concrete finite probability measures.

This is the unit-mass companion to
`finiteMassShellVariationalOptimality_of_finiteProbabilityMeasure_klDiv_computed`: the source
probability vector is nonnegative, the updated/reference vector is strictly positive, and the
finite all-test KL variational inequality is derived by computing mathlib's finite `klDiv`.
-/
theorem finiteProbabilityVariationalOptimality_of_finiteProbabilityMeasure_klDiv_computed
    {n : ℕ} {p q : Fin n → ℝ}
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = 1)
    (hq_pos : ∀ i, 0 < q i)
    (hq_mass : ∑ i, q i = 1) :
    FiniteProbabilityVariationalOptimality p q :=
  finiteProbabilityVariationalOptimality_of_primitives
    (fun i => (hq_pos i).le)
    hq_mass
    (finite_variational_of_finiteProbabilityMeasure_klDiv_computed
      (p := p) (q := q) hp_nonneg hp_mass hq_pos hq_mass)

/-- Extract nonnegativity from probability variational optimality. -/
theorem finiteProbabilityVariationalOptimality_nonneg
    {n : ℕ} {p q : Fin n → ℝ}
    (hvar : FiniteProbabilityVariationalOptimality p q) :
    ∀ i, 0 ≤ q i :=
  hvar.1

/-- Extract unit mass from probability variational optimality. -/
theorem finiteProbabilityVariationalOptimality_mass
    {n : ℕ} {p q : Fin n → ℝ}
    (hvar : FiniteProbabilityVariationalOptimality p q) :
    ∑ i, q i = 1 :=
  hvar.2.1

/-- Extract the all-test inequality from probability variational optimality. -/
theorem finiteProbabilityVariationalOptimality_variational
    {n : ℕ} {p q : Fin n → ℝ}
    (hvar : FiniteProbabilityVariationalOptimality p q) :
    ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
      t * (∑ i, p i * s i)
        - Real.log (∑ i, q i * Real.exp (t * s i))
        ≤ finiteKL p q :=
  hvar.2.2

/--
View probability variational optimality as mass-shell variational optimality at `M = 1`.
-/
theorem finiteMassShellVariationalOptimality_of_probabilityVariationalOptimality
    {n : ℕ} {p q : Fin n → ℝ}
    (hvar : FiniteProbabilityVariationalOptimality p q) :
    FiniteMassShellVariationalOptimality p q 1 :=
  finiteMassShellVariationalOptimality_of_primitives
    (finiteProbabilityVariationalOptimality_nonneg hvar)
    (finiteProbabilityVariationalOptimality_mass hvar)
    (by
      intro s t ht
      simpa using finiteProbabilityVariationalOptimality_variational hvar s t ht)

/--
Recover probability variational optimality from the unit-mass mass-shell predicate.
-/
theorem finiteProbabilityVariationalOptimality_of_massShellVariationalOptimality_one
    {n : ℕ} {p q : Fin n → ℝ}
    (hvar : FiniteMassShellVariationalOptimality p q 1) :
    FiniteProbabilityVariationalOptimality p q :=
  finiteProbabilityVariationalOptimality_of_primitives
    (finiteMassShellVariationalOptimality_nonneg hvar)
    (finiteMassShellVariationalOptimality_mass hvar)
    (by
      intro s t ht
      simpa using finiteMassShellVariationalOptimality_variational hvar s t ht)

/--
Single-block A.1 bridge from normalized probability variational optimality and a block-update
KL-gain primitive.
-/
theorem halfStepAscent_of_finiteProbabilityVariationalOptimality_blockUpdateGain
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter : ℝ}
    (hvar : FiniteProbabilityVariationalOptimality p q)
    (hgain : FiniteKLGainFromBlockUpdate p q Fbefore Fafter) :
    (l1Norm (fun i => p i - q i)) ^ 2 / 2 ≤ Fafter - Fbefore := by
  exact halfStepAscent_of_probabilityVariationalMathlibPinsker_klGain
    (p := p) (q := q)
    (finiteProbabilityVariationalOptimality_nonneg hvar)
    (finiteProbabilityVariationalOptimality_mass hvar)
    (finiteProbabilityVariationalOptimality_variational hvar)
    ((finiteKLGainFromBlockUpdate_iff_kl_le_objectiveDiff).1 hgain)

/--
Sequence form of
`halfStepAscent_of_finiteProbabilityVariationalOptimality_blockUpdateGain`.
-/
theorem halfStepAscent_seq_of_finiteProbabilityVariationalOptimality_blockUpdateGain
    {n : ℕ} {p q : ℕ → Fin n → ℝ} {Fbefore Fafter : ℕ → ℝ}
    (hvar : ∀ k : ℕ, FiniteProbabilityVariationalOptimality (p k) (q k))
    (hgain :
      ∀ k : ℕ, FiniteKLGainFromBlockUpdate (p k) (q k) (Fbefore k) (Fafter k)) :
    ∀ k : ℕ,
      (l1Norm (fun i => p k i - q k i)) ^ 2 / 2
        ≤ Fafter k - Fbefore k := by
  intro k
  exact halfStepAscent_of_finiteProbabilityVariationalOptimality_blockUpdateGain
    (p := p k) (q := q k) (hvar k) (hgain k)

/--
Full-sweep A.1 bridge from normalized probability variational optimality and primitive
block-update gains.
-/
theorem perStepAscent_residualProxy_of_finiteProbabilityVariationalOptimality_blockUpdateGains
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ}
    (hvar1 : ∀ k : ℕ, FiniteProbabilityVariationalOptimality (p1 k) (q1 k))
    (hgain1 :
      ∀ k : ℕ, FiniteKLGainFromBlockUpdate (p1 k) (q1 k) (F k) (Fhalf k))
    (hvar2 : ∀ k : ℕ, FiniteProbabilityVariationalOptimality (p2 k) (q2 k))
    (hgain2 :
      ∀ k : ℕ, FiniteKLGainFromBlockUpdate (p2 k) (q2 k) (Fhalf k) (F (k + 1))) :
    ∀ k : ℕ,
      (1 / 2) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k := by
  exact perStepAscent_residualProxy_of_probabilityVariationalMathlibPinsker_klGains_add
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (fun k => finiteProbabilityVariationalOptimality_nonneg (hvar1 k))
    (fun k => finiteProbabilityVariationalOptimality_mass (hvar1 k))
    (fun k => finiteProbabilityVariationalOptimality_variational (hvar1 k))
    (fun k => finiteKLGainFromBlockUpdate_gain_add (hgain1 k))
    (fun k => finiteProbabilityVariationalOptimality_nonneg (hvar2 k))
    (fun k => finiteProbabilityVariationalOptimality_mass (hvar2 k))
    (fun k => finiteProbabilityVariationalOptimality_variational (hvar2 k))
    (fun k => finiteKLGainFromBlockUpdate_gain_add (hgain2 k))

/--
Full-sweep probability bridge when both block objective gains are exact additive identities.
-/
theorem perStepAscent_residualProxy_of_finiteProbabilityVariationalOptimality_exactBlockUpdateGains
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ}
    (hvar1 : ∀ k : ℕ, FiniteProbabilityVariationalOptimality (p1 k) (q1 k))
    (hexact1 :
      ∀ k : ℕ, FiniteKLExactAddGainFromBlockUpdate (p1 k) (q1 k) (F k) (Fhalf k))
    (hvar2 : ∀ k : ℕ, FiniteProbabilityVariationalOptimality (p2 k) (q2 k))
    (hexact2 :
      ∀ k : ℕ, FiniteKLExactAddGainFromBlockUpdate (p2 k) (q2 k) (Fhalf k) (F (k + 1))) :
    ∀ k : ℕ,
      (1 / 2) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k :=
  perStepAscent_residualProxy_of_finiteProbabilityVariationalOptimality_blockUpdateGains
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    hvar1
    (fun k => finiteKLGainFromBlockUpdate_of_exactAddGain (hexact1 k))
    hvar2
    (fun k => finiteKLGainFromBlockUpdate_of_exactAddGain (hexact2 k))

/--
Full-sweep mass-shell bridge when both block objective gains are exact additive identities.
-/
theorem perStepAscent_residualProxy_of_finiteMassShellVariationalOptimality_exactBlockUpdateGains_commonMass
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hvar1 : ∀ k : ℕ, FiniteMassShellVariationalOptimality (p1 k) (q1 k) M)
    (hexact1 :
      ∀ k : ℕ, FiniteKLExactAddGainFromBlockUpdate (p1 k) (q1 k) (F k) (Fhalf k))
    (hvar2 : ∀ k : ℕ, FiniteMassShellVariationalOptimality (p2 k) (q2 k) M)
    (hexact2 :
      ∀ k : ℕ, FiniteKLExactAddGainFromBlockUpdate (p2 k) (q2 k) (Fhalf k) (F (k + 1))) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k :=
  perStepAscent_residualProxy_of_finiteMassShellVariationalOptimality_blockUpdateGains_commonMass
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hMpos hvar1
    (fun k => finiteKLGainFromBlockUpdate_of_exactAddGain (hexact1 k))
    hvar2
    (fun k => finiteKLGainFromBlockUpdate_of_exactAddGain (hexact2 k))

/--
Single-block A.1 bridge using the computed positive-reference finite-measure A.3 endpoint.

Here the all-test variational premise has disappeared: strict positivity of the reference block
lets Lean compute the finite measure KL identity and invoke the concrete finite-measure Pinsker
endpoint directly.
-/
theorem halfStepAscent_of_finiteProbabilityMeasureComputedPinsker_klGain_commonMass
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter M : ℝ}
    (hMpos : 0 < M)
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = M)
    (hq_pos : ∀ i, 0 < q i)
    (hq_mass : ∑ i, q i = M)
    (hgain : finiteKL p q ≤ Fafter - Fbefore) :
    (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) ≤ Fafter - Fbefore := by
  have hpinsker :
      finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) :=
    pinsker_nonnormalized_of_finiteProbabilityMeasure_klDiv_computed_mathlib_hoeffding_massShell
      (p := p) (q := q) hMpos hp_nonneg hp_mass hq_pos hq_mass
  exact halfStepAscent_of_klGain_of_finitePinsker hpinsker hgain

/--
Additive-gain form of
`halfStepAscent_of_finiteProbabilityMeasureComputedPinsker_klGain_commonMass`.
-/
theorem halfStepAscent_of_finiteProbabilityMeasureComputedPinsker_klGain_add_commonMass
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter M : ℝ}
    (hMpos : 0 < M)
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = M)
    (hq_pos : ∀ i, 0 < q i)
    (hq_mass : ∑ i, q i = M)
    (hgain_add : Fbefore + finiteKL p q ≤ Fafter) :
    (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) ≤ Fafter - Fbefore :=
  halfStepAscent_of_finiteProbabilityMeasureComputedPinsker_klGain_commonMass
    (p := p) (q := q) (M := M) hMpos hp_nonneg hp_mass hq_pos hq_mass
    (klGain_of_klGain_add hgain_add)

/--
Single-block A.1 bridge using the support-aware computed finite-measure A.3 endpoint.

This is the boundary analogue of
`halfStepAscent_of_finiteProbabilityMeasureComputedPinsker_klGain_commonMass`: the all-test
variational premise is not assumed, and zero coordinates in `q` are allowed when `p` is also zero
there.
-/
theorem halfStepAscent_of_finiteProbabilityMeasureComputedPinsker_klGain_commonMass_of_support
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter M : ℝ}
    (hMpos : 0 < M)
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = M)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = M)
    (hsupp : ∀ i, q i = 0 → p i = 0)
    (hgain : finiteKL p q ≤ Fafter - Fbefore) :
    (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) ≤ Fafter - Fbefore := by
  have hpinsker :
      finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) :=
    pinsker_nonnormalized_of_finiteProbabilityMeasure_klDiv_computed_support
      (p := p) (q := q) hMpos hp_nonneg hp_mass hq_nonneg hq_mass hsupp
  exact halfStepAscent_of_klGain_of_finitePinsker hpinsker hgain

/--
Additive-gain form of the support-aware computed-Pinsker half-step bridge.
-/
theorem halfStepAscent_of_finiteProbabilityMeasureComputedPinsker_klGain_add_commonMass_of_support
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter M : ℝ}
    (hMpos : 0 < M)
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = M)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = M)
    (hsupp : ∀ i, q i = 0 → p i = 0)
    (hgain_add : Fbefore + finiteKL p q ≤ Fafter) :
    (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) ≤ Fafter - Fbefore :=
  halfStepAscent_of_finiteProbabilityMeasureComputedPinsker_klGain_commonMass_of_support
    (p := p) (q := q) (M := M) hMpos hp_nonneg hp_mass hq_nonneg hq_mass hsupp
    (klGain_of_klGain_add hgain_add)

/--
Sequence form of the positive-reference computed-Pinsker half-step bridge.
-/
theorem halfStepAscent_seq_of_finiteProbabilityMeasureComputedPinsker_klGain_commonMass
    {n : ℕ} {p q : ℕ → Fin n → ℝ} {Fbefore Fafter : ℕ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hp_nonneg : ∀ k i, 0 ≤ p k i)
    (hp_mass : ∀ k, ∑ i, p k i = M)
    (hq_pos : ∀ k i, 0 < q k i)
    (hq_mass : ∀ k, ∑ i, q k i = M)
    (hgain : ∀ k : ℕ, finiteKL (p k) (q k) ≤ Fafter k - Fbefore k) :
    ∀ k : ℕ,
      (l1Norm (fun i => p k i - q k i)) ^ 2 / (2 * M)
        ≤ Fafter k - Fbefore k := by
  intro k
  exact halfStepAscent_of_finiteProbabilityMeasureComputedPinsker_klGain_commonMass
    (p := p k) (q := q k) (M := M) hMpos
    (hp_nonneg k) (hp_mass k) (hq_pos k) (hq_mass k) (hgain k)

/--
Sequence form of the support-aware computed-Pinsker half-step bridge.
-/
theorem halfStepAscent_seq_of_finiteProbabilityMeasureComputedPinsker_klGain_commonMass_of_support
    {n : ℕ} {p q : ℕ → Fin n → ℝ} {Fbefore Fafter : ℕ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hp_nonneg : ∀ k i, 0 ≤ p k i)
    (hp_mass : ∀ k, ∑ i, p k i = M)
    (hq_nonneg : ∀ k i, 0 ≤ q k i)
    (hq_mass : ∀ k, ∑ i, q k i = M)
    (hsupp : ∀ k i, q k i = 0 → p k i = 0)
    (hgain : ∀ k : ℕ, finiteKL (p k) (q k) ≤ Fafter k - Fbefore k) :
    ∀ k : ℕ,
      (l1Norm (fun i => p k i - q k i)) ^ 2 / (2 * M)
        ≤ Fafter k - Fbefore k := by
  intro k
  exact halfStepAscent_of_finiteProbabilityMeasureComputedPinsker_klGain_commonMass_of_support
    (p := p k) (q := q k) (M := M) hMpos
    (hp_nonneg k) (hp_mass k) (hq_nonneg k) (hq_mass k) (hsupp k) (hgain k)

/--
Full-sweep A.1 bridge from positive-reference finite blocks and computed finite-measure Pinsker.
-/
theorem perStepAscent_residualProxy_of_finiteProbabilityMeasureComputedPinsker_klGains_commonMass
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hp1_nonneg : ∀ k i, 0 ≤ p1 k i)
    (hp1_mass : ∀ k, ∑ i, p1 k i = M)
    (hq1_pos : ∀ k i, 0 < q1 k i)
    (hq1_mass : ∀ k, ∑ i, q1 k i = M)
    (hgain1 : ∀ k : ℕ, finiteKL (p1 k) (q1 k) ≤ Fhalf k - F k)
    (hp2_nonneg : ∀ k i, 0 ≤ p2 k i)
    (hp2_mass : ∀ k, ∑ i, p2 k i = M)
    (hq2_pos : ∀ k i, 0 < q2 k i)
    (hq2_mass : ∀ k, ∑ i, q2 k i = M)
    (hgain2 : ∀ k : ℕ, finiteKL (p2 k) (q2 k) ≤ F (k + 1) - Fhalf k) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k := by
  have hpinsker1 :
      ∀ k : ℕ,
        finiteKL (p1 k) (q1 k) ≥
          (l1Norm (fun i => p1 k i - q1 k i)) ^ 2 / (2 * M) := by
    intro k
    exact pinsker_nonnormalized_of_finiteProbabilityMeasure_klDiv_computed_mathlib_hoeffding_massShell
      (p := p1 k) (q := q1 k) hMpos
      (hp1_nonneg k) (hp1_mass k) (hq1_pos k) (hq1_mass k)
  have hpinsker2 :
      ∀ k : ℕ,
        finiteKL (p2 k) (q2 k) ≥
          (l1Norm (fun i => p2 k i - q2 k i)) ^ 2 / (2 * M) := by
    intro k
    exact pinsker_nonnormalized_of_finiteProbabilityMeasure_klDiv_computed_mathlib_hoeffding_massShell
      (p := p2 k) (q := q2 k) hMpos
      (hp2_nonneg k) (hp2_mass k) (hq2_pos k) (hq2_mass k)
  exact perStepAscent_residualProxy_of_finitePinsker_klGains_commonMass
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hpinsker1 hgain1 hpinsker2 hgain2

/--
Additive-gain form of the positive-reference computed-Pinsker full-sweep bridge.
-/
theorem perStepAscent_residualProxy_of_finiteProbabilityMeasureComputedPinsker_klGains_add_commonMass
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hp1_nonneg : ∀ k i, 0 ≤ p1 k i)
    (hp1_mass : ∀ k, ∑ i, p1 k i = M)
    (hq1_pos : ∀ k i, 0 < q1 k i)
    (hq1_mass : ∀ k, ∑ i, q1 k i = M)
    (hgain1_add : ∀ k : ℕ, F k + finiteKL (p1 k) (q1 k) ≤ Fhalf k)
    (hp2_nonneg : ∀ k i, 0 ≤ p2 k i)
    (hp2_mass : ∀ k, ∑ i, p2 k i = M)
    (hq2_pos : ∀ k i, 0 < q2 k i)
    (hq2_mass : ∀ k, ∑ i, q2 k i = M)
    (hgain2_add : ∀ k : ℕ, Fhalf k + finiteKL (p2 k) (q2 k) ≤ F (k + 1)) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k :=
  perStepAscent_residualProxy_of_finiteProbabilityMeasureComputedPinsker_klGains_commonMass
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hMpos hp1_nonneg hp1_mass hq1_pos hq1_mass
    (klGain_seq_of_klGain_add hgain1_add)
    hp2_nonneg hp2_mass hq2_pos hq2_mass
    (klGain_seq_of_klGain_add hgain2_add)

/--
Full-sweep A.1 bridge from support-aware finite blocks and computed finite-measure Pinsker.

This is the direct boundary counterpart of
`perStepAscent_residualProxy_of_finiteProbabilityMeasureComputedPinsker_klGains_commonMass`.
It derives Pinsker from concrete finite-measure KL plus support domination, not from a supplied
variational inequality.
-/
theorem
    perStepAscent_residualProxy_of_finiteProbabilityMeasureComputedPinsker_klGains_commonMass_of_support
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hp1_nonneg : ∀ k i, 0 ≤ p1 k i)
    (hp1_mass : ∀ k, ∑ i, p1 k i = M)
    (hq1_nonneg : ∀ k i, 0 ≤ q1 k i)
    (hq1_mass : ∀ k, ∑ i, q1 k i = M)
    (hsupp1 : ∀ k i, q1 k i = 0 → p1 k i = 0)
    (hgain1 : ∀ k : ℕ, finiteKL (p1 k) (q1 k) ≤ Fhalf k - F k)
    (hp2_nonneg : ∀ k i, 0 ≤ p2 k i)
    (hp2_mass : ∀ k, ∑ i, p2 k i = M)
    (hq2_nonneg : ∀ k i, 0 ≤ q2 k i)
    (hq2_mass : ∀ k, ∑ i, q2 k i = M)
    (hsupp2 : ∀ k i, q2 k i = 0 → p2 k i = 0)
    (hgain2 : ∀ k : ℕ, finiteKL (p2 k) (q2 k) ≤ F (k + 1) - Fhalf k) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k := by
  have hpinsker1 :
      ∀ k : ℕ,
        finiteKL (p1 k) (q1 k) ≥
          (l1Norm (fun i => p1 k i - q1 k i)) ^ 2 / (2 * M) := by
    intro k
    exact pinsker_nonnormalized_of_finiteProbabilityMeasure_klDiv_computed_support
      (p := p1 k) (q := q1 k) hMpos
      (hp1_nonneg k) (hp1_mass k) (hq1_nonneg k) (hq1_mass k) (hsupp1 k)
  have hpinsker2 :
      ∀ k : ℕ,
        finiteKL (p2 k) (q2 k) ≥
          (l1Norm (fun i => p2 k i - q2 k i)) ^ 2 / (2 * M) := by
    intro k
    exact pinsker_nonnormalized_of_finiteProbabilityMeasure_klDiv_computed_support
      (p := p2 k) (q := q2 k) hMpos
      (hp2_nonneg k) (hp2_mass k) (hq2_nonneg k) (hq2_mass k) (hsupp2 k)
  exact perStepAscent_residualProxy_of_finitePinsker_klGains_commonMass
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hpinsker1 hgain1 hpinsker2 hgain2

/--
Additive-gain form of the support-aware computed-Pinsker full-sweep bridge.
-/
theorem
    perStepAscent_residualProxy_of_finiteProbabilityMeasureComputedPinsker_klGains_add_commonMass_of_support
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hp1_nonneg : ∀ k i, 0 ≤ p1 k i)
    (hp1_mass : ∀ k, ∑ i, p1 k i = M)
    (hq1_nonneg : ∀ k i, 0 ≤ q1 k i)
    (hq1_mass : ∀ k, ∑ i, q1 k i = M)
    (hsupp1 : ∀ k i, q1 k i = 0 → p1 k i = 0)
    (hgain1_add : ∀ k : ℕ, F k + finiteKL (p1 k) (q1 k) ≤ Fhalf k)
    (hp2_nonneg : ∀ k i, 0 ≤ p2 k i)
    (hp2_mass : ∀ k, ∑ i, p2 k i = M)
    (hq2_nonneg : ∀ k i, 0 ≤ q2 k i)
    (hq2_mass : ∀ k, ∑ i, q2 k i = M)
    (hsupp2 : ∀ k i, q2 k i = 0 → p2 k i = 0)
    (hgain2_add : ∀ k : ℕ, Fhalf k + finiteKL (p2 k) (q2 k) ≤ F (k + 1)) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k :=
  perStepAscent_residualProxy_of_finiteProbabilityMeasureComputedPinsker_klGains_commonMass_of_support
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hMpos hp1_nonneg hp1_mass hq1_nonneg hq1_mass hsupp1
    (klGain_seq_of_klGain_add hgain1_add)
    hp2_nonneg hp2_mass hq2_nonneg hq2_mass hsupp2
    (klGain_seq_of_klGain_add hgain2_add)

/--
One-sweep, non-sequence A.1 bridge from separated mass-shell primitives.

This is close to algorithm update notation: one first block update from `Fbefore` to `Fhalf`,
then one second block update from `Fhalf` to `Fafter`.
-/
theorem twoBlockAscent_of_finiteMassShellVariationalOptimality_blockUpdateGains_commonMass
    {n₁ n₂ : ℕ}
    {p1 q1 : Fin n₁ → ℝ} {p2 q2 : Fin n₂ → ℝ} {Fbefore Fhalf Fafter M : ℝ}
    (hMpos : 0 < M)
    (hvar1 : FiniteMassShellVariationalOptimality p1 q1 M)
    (hgain1 : FiniteKLGainFromBlockUpdate p1 q1 Fbefore Fhalf)
    (hvar2 : FiniteMassShellVariationalOptimality p2 q2 M)
    (hgain2 : FiniteKLGainFromBlockUpdate p2 q2 Fhalf Fafter) :
    (1 / (2 * M)) *
        ((l1Norm (fun i => p1 i - q1 i)) ^ 2
          + (l1Norm (fun i => p2 i - q2 i)) ^ 2)
      ≤ Fafter - Fbefore := by
  have hA1 :
      (l1Norm (fun i => p1 i - q1 i)) ^ 2 / (2 * M) ≤ Fhalf - Fbefore :=
    halfStepAscent_of_finiteMassShellVariationalOptimality_blockUpdateGain_commonMass
      (p := p1) (q := q1) (M := M) hMpos hvar1 hgain1
  have hA2 :
      (l1Norm (fun i => p2 i - q2 i)) ^ 2 / (2 * M) ≤ Fafter - Fhalf :=
    halfStepAscent_of_finiteMassShellVariationalOptimality_blockUpdateGain_commonMass
      (p := p2) (q := q2) (M := M) hMpos hvar2 hgain2
  have hsum :
      (l1Norm (fun i => p1 i - q1 i)) ^ 2 / (2 * M)
        + (l1Norm (fun i => p2 i - q2 i)) ^ 2 / (2 * M)
        ≤ (Fhalf - Fbefore) + (Fafter - Fhalf) :=
    add_le_add hA1 hA2
  calc
    (1 / (2 * M)) *
        ((l1Norm (fun i => p1 i - q1 i)) ^ 2
          + (l1Norm (fun i => p2 i - q2 i)) ^ 2)
        =
          (l1Norm (fun i => p1 i - q1 i)) ^ 2 / (2 * M)
            + (l1Norm (fun i => p2 i - q2 i)) ^ 2 / (2 * M) := by
              ring
    _ ≤ (Fhalf - Fbefore) + (Fafter - Fhalf) := hsum
    _ = Fafter - Fbefore := by ring

/--
Single-block A.1 bridge from an algorithm-facing finite block-update certificate.

The all-test variational inequality is not assumed here: it is derived from finite measure KL
variationality using the feasibility/strict-positivity fields in the certificate.
-/
theorem halfStepAscent_of_finiteMassShellBlockUpdateCertificate_commonMass
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter M : ℝ}
    (hMpos : 0 < M)
    (hcert : FiniteMassShellBlockUpdateCertificate p q M Fbefore Fafter) :
    (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) ≤ Fafter - Fbefore :=
  halfStepAscent_of_finiteProbabilityMeasureComputedPinsker_klGain_add_commonMass
    (p := p) (q := q) (M := M) hMpos
    hcert.source_nonneg hcert.source_mass hcert.update_pos hcert.update_mass
    (finiteKLGainFromBlockUpdate_gain_add
      (finiteKLGainFromBlockUpdate_of_blockUpdateCertificate hcert))

/--
Sequence form of `halfStepAscent_of_finiteMassShellBlockUpdateCertificate_commonMass`.
-/
theorem halfStepAscent_seq_of_finiteMassShellBlockUpdateCertificate_commonMass
    {n : ℕ} {p q : ℕ → Fin n → ℝ} {Fbefore Fafter : ℕ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hcert :
      ∀ k : ℕ, FiniteMassShellBlockUpdateCertificate (p k) (q k) M
        (Fbefore k) (Fafter k)) :
    ∀ k : ℕ,
      (l1Norm (fun i => p k i - q k i)) ^ 2 / (2 * M)
        ≤ Fafter k - Fbefore k := by
  intro k
  exact halfStepAscent_of_finiteMassShellBlockUpdateCertificate_commonMass
    (p := p k) (q := q k) (M := M) hMpos (hcert k)

/--
Full-sweep A.1 bridge from algorithm-facing finite block-update certificates.

This is the interface future concrete update modules should target when no standalone minimizer
API exists yet: prove finite feasibility/strict positivity and objective KL gain for each block,
then this theorem supplies both variational optimality and the residual-ascent inequality.
-/
theorem perStepAscent_residualProxy_of_finiteMassShellBlockUpdateCertificates_commonMass
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hcert1 :
      ∀ k : ℕ,
        FiniteMassShellBlockUpdateCertificate (p1 k) (q1 k) M (F k) (Fhalf k))
    (hcert2 :
      ∀ k : ℕ,
        FiniteMassShellBlockUpdateCertificate (p2 k) (q2 k) M (Fhalf k) (F (k + 1))) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k :=
  perStepAscent_residualProxy_of_finiteProbabilityMeasureComputedPinsker_klGains_add_commonMass
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hMpos
    (fun k => (hcert1 k).source_nonneg)
    (fun k => (hcert1 k).source_mass)
    (fun k => (hcert1 k).update_pos)
    (fun k => (hcert1 k).update_mass)
    (fun k =>
      finiteKLGainFromBlockUpdate_gain_add
        (finiteKLGainFromBlockUpdate_of_blockUpdateCertificate (hcert1 k)))
    (fun k => (hcert2 k).source_nonneg)
    (fun k => (hcert2 k).source_mass)
    (fun k => (hcert2 k).update_pos)
    (fun k => (hcert2 k).update_mass)
    (fun k =>
      finiteKLGainFromBlockUpdate_gain_add
        (finiteKLGainFromBlockUpdate_of_blockUpdateCertificate (hcert2 k)))

/--
One-sweep, non-sequence A.1 bridge from algorithm-facing finite block-update certificates.
-/
theorem twoBlockAscent_of_finiteMassShellBlockUpdateCertificates_commonMass
    {n₁ n₂ : ℕ}
    {p1 q1 : Fin n₁ → ℝ} {p2 q2 : Fin n₂ → ℝ} {Fbefore Fhalf Fafter M : ℝ}
    (hMpos : 0 < M)
    (hcert1 : FiniteMassShellBlockUpdateCertificate p1 q1 M Fbefore Fhalf)
    (hcert2 : FiniteMassShellBlockUpdateCertificate p2 q2 M Fhalf Fafter) :
    (1 / (2 * M)) *
        ((l1Norm (fun i => p1 i - q1 i)) ^ 2
          + (l1Norm (fun i => p2 i - q2 i)) ^ 2)
      ≤ Fafter - Fbefore :=
  twoBlockAscent_of_finiteMassShellVariationalOptimality_blockUpdateGains_commonMass
    (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2) (M := M) hMpos
    (finiteMassShellVariationalOptimality_of_blockUpdateCertificate hMpos hcert1)
    (finiteKLGainFromBlockUpdate_of_blockUpdateCertificate hcert1)
    (finiteMassShellVariationalOptimality_of_blockUpdateCertificate hMpos hcert2)
    (finiteKLGainFromBlockUpdate_of_blockUpdateCertificate hcert2)

/--
Single-block A.1 bridge from a support-aware finite block-update certificate.

This is the boundary version of
`halfStepAscent_of_finiteMassShellBlockUpdateCertificate_commonMass`: strict positivity of the
updated block is replaced by nonnegativity plus support domination.
-/
theorem halfStepAscent_of_finiteMassShellSupportBlockUpdateCertificate_commonMass
    {n : ℕ} {p q : Fin n → ℝ} {Fbefore Fafter M : ℝ}
    (hMpos : 0 < M)
    (hcert : FiniteMassShellSupportBlockUpdateCertificate p q M Fbefore Fafter) :
    (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) ≤ Fafter - Fbefore :=
  halfStepAscent_of_finiteProbabilityMeasureComputedPinsker_klGain_add_commonMass_of_support
    (p := p) (q := q) (M := M) hMpos
    hcert.source_nonneg hcert.source_mass hcert.update_nonneg hcert.update_mass hcert.support
    (finiteKLGainFromBlockUpdate_gain_add
      (finiteKLGainFromBlockUpdate_of_supportBlockUpdateCertificate hcert))

/--
Sequence form of the support-aware single-block A.1 bridge.
-/
theorem halfStepAscent_seq_of_finiteMassShellSupportBlockUpdateCertificate_commonMass
    {n : ℕ} {p q : ℕ → Fin n → ℝ} {Fbefore Fafter : ℕ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hcert :
      ∀ k : ℕ, FiniteMassShellSupportBlockUpdateCertificate (p k) (q k) M
        (Fbefore k) (Fafter k)) :
    ∀ k : ℕ,
      (l1Norm (fun i => p k i - q k i)) ^ 2 / (2 * M)
        ≤ Fafter k - Fbefore k := by
  intro k
  exact halfStepAscent_of_finiteMassShellSupportBlockUpdateCertificate_commonMass
    (p := p k) (q := q k) (M := M) hMpos (hcert k)

/--
Full-sweep A.1 bridge from support-aware finite block-update certificates.

Future concrete update modules can target this theorem when block updates may have boundary
coordinates: the all-test variational inequalities are derived from finite KL plus support
domination, not stored in the certificate.
-/
theorem perStepAscent_residualProxy_of_finiteMassShellSupportBlockUpdateCertificates_commonMass
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hcert1 :
      ∀ k : ℕ,
        FiniteMassShellSupportBlockUpdateCertificate (p1 k) (q1 k) M (F k) (Fhalf k))
    (hcert2 :
      ∀ k : ℕ,
        FiniteMassShellSupportBlockUpdateCertificate (p2 k) (q2 k) M
          (Fhalf k) (F (k + 1))) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k :=
  perStepAscent_residualProxy_of_finiteProbabilityMeasureComputedPinsker_klGains_add_commonMass_of_support
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hMpos
    (fun k => (hcert1 k).source_nonneg)
    (fun k => (hcert1 k).source_mass)
    (fun k => (hcert1 k).update_nonneg)
    (fun k => (hcert1 k).update_mass)
    (fun k => (hcert1 k).support)
    (fun k =>
      finiteKLGainFromBlockUpdate_gain_add
        (finiteKLGainFromBlockUpdate_of_supportBlockUpdateCertificate (hcert1 k)))
    (fun k => (hcert2 k).source_nonneg)
    (fun k => (hcert2 k).source_mass)
    (fun k => (hcert2 k).update_nonneg)
    (fun k => (hcert2 k).update_mass)
    (fun k => (hcert2 k).support)
    (fun k =>
      finiteKLGainFromBlockUpdate_gain_add
        (finiteKLGainFromBlockUpdate_of_supportBlockUpdateCertificate (hcert2 k)))

/--
One-sweep A.1 bridge from support-aware finite block-update certificates.
-/
theorem twoBlockAscent_of_finiteMassShellSupportBlockUpdateCertificates_commonMass
    {n₁ n₂ : ℕ}
    {p1 q1 : Fin n₁ → ℝ} {p2 q2 : Fin n₂ → ℝ} {Fbefore Fhalf Fafter M : ℝ}
    (hMpos : 0 < M)
    (hcert1 : FiniteMassShellSupportBlockUpdateCertificate p1 q1 M Fbefore Fhalf)
    (hcert2 : FiniteMassShellSupportBlockUpdateCertificate p2 q2 M Fhalf Fafter) :
    (1 / (2 * M)) *
        ((l1Norm (fun i => p1 i - q1 i)) ^ 2
          + (l1Norm (fun i => p2 i - q2 i)) ^ 2)
      ≤ Fafter - Fbefore :=
  twoBlockAscent_of_finiteMassShellVariationalOptimality_blockUpdateGains_commonMass
    (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2) (M := M) hMpos
    (finiteMassShellVariationalOptimality_of_supportBlockUpdateCertificate hMpos hcert1)
    (finiteKLGainFromBlockUpdate_of_supportBlockUpdateCertificate hcert1)
    (finiteMassShellVariationalOptimality_of_supportBlockUpdateCertificate hMpos hcert2)
    (finiteKLGainFromBlockUpdate_of_supportBlockUpdateCertificate hcert2)

/--
Full-sweep bridge from exact-gain support-aware finite block-update certificates.
-/
theorem perStepAscent_residualProxy_of_finiteMassShellExactSupportBlockUpdateCertificates_commonMass
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hcert1 :
      ∀ k : ℕ,
        FiniteMassShellExactSupportBlockUpdateCertificate (p1 k) (q1 k) M
          (F k) (Fhalf k))
    (hcert2 :
      ∀ k : ℕ,
        FiniteMassShellExactSupportBlockUpdateCertificate (p2 k) (q2 k) M
          (Fhalf k) (F (k + 1))) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k :=
  perStepAscent_residualProxy_of_finiteMassShellSupportBlockUpdateCertificates_commonMass
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hMpos
    (fun k => finiteMassShellSupportBlockUpdateCertificate_of_exactSupportBlockUpdateCertificate
      (hcert1 k))
    (fun k => finiteMassShellSupportBlockUpdateCertificate_of_exactSupportBlockUpdateCertificate
      (hcert2 k))

/--
Paper-shaped two-half-step form of Lemma A.1 from exact support-aware finite block certificates.

The conclusion states the two displayed inequalities separately, with the paper constant
`γ/(2 Xmax ||A||_{1→1}²)`.  The certified KL-projection half-step ascent is supplied by the
existing exact support block-update certificates; the remaining explicit inputs are precisely the
mass-to-`Xmax` scale relation and the residual Lipschitz estimates
`||r_i||₁ ≤ ||A||_{1→1} ||x_i⁺-x_i||₁`.
-/
theorem perStepAscent_twoHalfSteps_paperConstants_of_exactSupportBlockUpdateCertificates_commonMass
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ}
    {r1 r2 : ℕ → ℝ} {gamma Xmax Anorm M : ℝ}
    (hgamma_nonneg : 0 ≤ gamma)
    (hXmax_pos : 0 < Xmax)
    (hAnorm_pos : 0 < Anorm)
    (hMpos : 0 < M)
    (hscale : gamma * M ≤ Xmax)
    (hcert1 :
      ∀ k : ℕ,
        FiniteMassShellExactSupportBlockUpdateCertificate (p1 k) (q1 k) M
          (F k) (Fhalf k))
    (hcert2 :
      ∀ k : ℕ,
        FiniteMassShellExactSupportBlockUpdateCertificate (p2 k) (q2 k) M
          (Fhalf k) (F (k + 1)))
    (hr1_nonneg : ∀ k : ℕ, 0 ≤ r1 k)
    (hr1_bound :
      ∀ k : ℕ, r1 k ≤ Anorm * l1Norm (fun i => p1 k i - q1 k i))
    (hr2_nonneg : ∀ k : ℕ, 0 ≤ r2 k)
    (hr2_bound :
      ∀ k : ℕ, r2 k ≤ Anorm * l1Norm (fun i => p2 k i - q2 k i)) :
    ∀ k : ℕ,
      (gamma / (2 * Xmax)) * ((r1 k) ^ 2 / Anorm ^ 2) ≤ Fhalf k - F k ∧
      (gamma / (2 * Xmax)) * ((r2 k) ^ 2 / Anorm ^ 2) ≤ F (k + 1) - Fhalf k := by
  have hcoef_le : gamma / (2 * Xmax) ≤ 1 / (2 * M) := by
    rw [div_le_div_iff₀ (mul_pos (by norm_num) hXmax_pos) (mul_pos (by norm_num) hMpos)]
    nlinarith
  intro k
  have hraw1 :
      (l1Norm (fun i => p1 k i - q1 k i)) ^ 2 / (2 * M) ≤ Fhalf k - F k := by
    exact halfStepAscent_of_finiteMassShellSupportBlockUpdateCertificate_commonMass
      (p := p1 k) (q := q1 k) (M := M) hMpos
      (finiteMassShellSupportBlockUpdateCertificate_of_exactSupportBlockUpdateCertificate
        (hcert1 k))
  have hchange1 :
      (gamma / (2 * Xmax)) * (l1Norm (fun i => p1 k i - q1 k i)) ^ 2
        ≤ Fhalf k - F k := by
    have hscaled :
        (gamma / (2 * Xmax)) * (l1Norm (fun i => p1 k i - q1 k i)) ^ 2
          ≤ (1 / (2 * M)) * (l1Norm (fun i => p1 k i - q1 k i)) ^ 2 := by
      exact mul_le_mul_of_nonneg_right hcoef_le (sq_nonneg _)
    have hraw1' :
        (1 / (2 * M)) * (l1Norm (fun i => p1 k i - q1 k i)) ^ 2
          ≤ Fhalf k - F k := by
      simpa [div_eq_mul_inv, one_div, mul_comm, mul_left_comm, mul_assoc] using hraw1
    exact hscaled.trans hraw1'
  have hA1 :
      (gamma / (2 * Xmax)) * ((r1 k) ^ 2 / Anorm ^ 2) ≤ Fhalf k - F k :=
    halfStepAscent_paperConstant_of_primalChangeBound
      (change := l1Norm (fun i => p1 k i - q1 k i))
      (residual := r1 k) (Fbefore := F k) (Fafter := Fhalf k)
      (gamma := gamma) (Xmax := Xmax) (Anorm := Anorm)
      hgamma_nonneg hXmax_pos hAnorm_pos (hr1_nonneg k) (hr1_bound k) hchange1
  have hraw2 :
      (l1Norm (fun i => p2 k i - q2 k i)) ^ 2 / (2 * M) ≤ F (k + 1) - Fhalf k := by
    exact halfStepAscent_of_finiteMassShellSupportBlockUpdateCertificate_commonMass
      (p := p2 k) (q := q2 k) (M := M) hMpos
      (finiteMassShellSupportBlockUpdateCertificate_of_exactSupportBlockUpdateCertificate
        (hcert2 k))
  have hchange2 :
      (gamma / (2 * Xmax)) * (l1Norm (fun i => p2 k i - q2 k i)) ^ 2
        ≤ F (k + 1) - Fhalf k := by
    have hscaled :
        (gamma / (2 * Xmax)) * (l1Norm (fun i => p2 k i - q2 k i)) ^ 2
          ≤ (1 / (2 * M)) * (l1Norm (fun i => p2 k i - q2 k i)) ^ 2 := by
      exact mul_le_mul_of_nonneg_right hcoef_le (sq_nonneg _)
    have hraw2' :
        (1 / (2 * M)) * (l1Norm (fun i => p2 k i - q2 k i)) ^ 2
          ≤ F (k + 1) - Fhalf k := by
      simpa [div_eq_mul_inv, one_div, mul_comm, mul_left_comm, mul_assoc] using hraw2
    exact hscaled.trans hraw2'
  have hA2 :
      (gamma / (2 * Xmax)) * ((r2 k) ^ 2 / Anorm ^ 2) ≤ F (k + 1) - Fhalf k :=
    halfStepAscent_paperConstant_of_primalChangeBound
      (change := l1Norm (fun i => p2 k i - q2 k i))
      (residual := r2 k) (Fbefore := Fhalf k) (Fafter := F (k + 1))
      (gamma := gamma) (Xmax := Xmax) (Anorm := Anorm)
      hgamma_nonneg hXmax_pos hAnorm_pos (hr2_nonneg k) (hr2_bound k) hchange2
  exact ⟨hA1, hA2⟩

/--
Single half-step ascent from a gamma-scaled exact support-aware finite block certificate.

This is the literal scaling used by the paper's dual objective:
`Fafter = Fbefore + gamma * KL(p‖q)`.
-/
theorem halfStepAscent_of_finiteMassShellGammaExactSupportBlockUpdateCertificate_commonMass
    {n : ℕ} {p q : Fin n → ℝ} {M gamma Fbefore Fafter : ℝ}
    (hgamma_nonneg : 0 ≤ gamma)
    (hMpos : 0 < M)
    (hcert : FiniteMassShellGammaExactSupportBlockUpdateCertificate
      p q M gamma Fbefore Fafter) :
    (gamma / (2 * M)) * (l1Norm (fun i => p i - q i)) ^ 2 ≤ Fafter - Fbefore := by
  let L := l1Norm (fun i => p i - q i)
  have hpinsker :
      finiteKL p q ≥ L ^ 2 / (2 * M) :=
    pinsker_nonnormalized_of_finiteProbabilityMeasure_klDiv_computed_support
      (p := p) (q := q) hMpos
      hcert.source_nonneg hcert.source_mass hcert.update_nonneg hcert.update_mass
      hcert.support
  have hmul : gamma * (L ^ 2 / (2 * M)) ≤ gamma * finiteKL p q :=
    mul_le_mul_of_nonneg_left hpinsker hgamma_nonneg
  have hgain : Fafter - Fbefore = gamma * finiteKL p q := by
    have hexact := hcert.exact_gain
    dsimp [FiniteKLGammaExactAddGainFromBlockUpdate] at hexact
    linarith
  calc
    (gamma / (2 * M)) * L ^ 2 = gamma * (L ^ 2 / (2 * M)) := by ring
    _ ≤ gamma * finiteKL p q := hmul
    _ = Fafter - Fbefore := hgain.symm

/--
Paper-shaped two-half-step form of Lemma A.1 with the literal gamma-scaled dual gain.

Compared with
`perStepAscent_twoHalfSteps_paperConstants_of_exactSupportBlockUpdateCertificates_commonMass`,
this endpoint uses block certificates whose exact gain is
`Fafter = Fbefore + gamma * KL(p‖q)`, matching the displayed dual objective increment in the
appendix proof.  The mass side condition is therefore the direct paper condition `M ≤ Xmax`.
-/
theorem perStepAscent_twoHalfSteps_paperConstants_of_gammaExactSupportBlockUpdateCertificates_commonMass
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ}
    {r1 r2 : ℕ → ℝ} {gamma Xmax Anorm M : ℝ}
    (hgamma_nonneg : 0 ≤ gamma)
    (hXmax_pos : 0 < Xmax)
    (hAnorm_pos : 0 < Anorm)
    (hMpos : 0 < M)
    (hM_le_Xmax : M ≤ Xmax)
    (hcert1 :
      ∀ k : ℕ,
        FiniteMassShellGammaExactSupportBlockUpdateCertificate (p1 k) (q1 k) M gamma
          (F k) (Fhalf k))
    (hcert2 :
      ∀ k : ℕ,
        FiniteMassShellGammaExactSupportBlockUpdateCertificate (p2 k) (q2 k) M gamma
          (Fhalf k) (F (k + 1)))
    (hr1_nonneg : ∀ k : ℕ, 0 ≤ r1 k)
    (hr1_bound :
      ∀ k : ℕ, r1 k ≤ Anorm * l1Norm (fun i => p1 k i - q1 k i))
    (hr2_nonneg : ∀ k : ℕ, 0 ≤ r2 k)
    (hr2_bound :
      ∀ k : ℕ, r2 k ≤ Anorm * l1Norm (fun i => p2 k i - q2 k i)) :
    ∀ k : ℕ,
      (gamma / (2 * Xmax)) * ((r1 k) ^ 2 / Anorm ^ 2) ≤ Fhalf k - F k ∧
      (gamma / (2 * Xmax)) * ((r2 k) ^ 2 / Anorm ^ 2) ≤ F (k + 1) - Fhalf k := by
  have hcoef_le : gamma / (2 * Xmax) ≤ gamma / (2 * M) := by
    rw [div_le_div_iff₀ (mul_pos (by norm_num) hXmax_pos) (mul_pos (by norm_num) hMpos)]
    nlinarith
  intro k
  have hraw1 :
      (gamma / (2 * M)) * (l1Norm (fun i => p1 k i - q1 k i)) ^ 2
        ≤ Fhalf k - F k :=
    halfStepAscent_of_finiteMassShellGammaExactSupportBlockUpdateCertificate_commonMass
      (p := p1 k) (q := q1 k) (M := M) (gamma := gamma)
      hgamma_nonneg hMpos (hcert1 k)
  have hchange1 :
      (gamma / (2 * Xmax)) * (l1Norm (fun i => p1 k i - q1 k i)) ^ 2
        ≤ Fhalf k - F k := by
    exact (mul_le_mul_of_nonneg_right hcoef_le (sq_nonneg _)).trans hraw1
  have hA1 :
      (gamma / (2 * Xmax)) * ((r1 k) ^ 2 / Anorm ^ 2) ≤ Fhalf k - F k :=
    halfStepAscent_paperConstant_of_primalChangeBound
      (change := l1Norm (fun i => p1 k i - q1 k i))
      (residual := r1 k) (Fbefore := F k) (Fafter := Fhalf k)
      (gamma := gamma) (Xmax := Xmax) (Anorm := Anorm)
      hgamma_nonneg hXmax_pos hAnorm_pos (hr1_nonneg k) (hr1_bound k) hchange1
  have hraw2 :
      (gamma / (2 * M)) * (l1Norm (fun i => p2 k i - q2 k i)) ^ 2
        ≤ F (k + 1) - Fhalf k :=
    halfStepAscent_of_finiteMassShellGammaExactSupportBlockUpdateCertificate_commonMass
      (p := p2 k) (q := q2 k) (M := M) (gamma := gamma)
      hgamma_nonneg hMpos (hcert2 k)
  have hchange2 :
      (gamma / (2 * Xmax)) * (l1Norm (fun i => p2 k i - q2 k i)) ^ 2
        ≤ F (k + 1) - Fhalf k := by
    exact (mul_le_mul_of_nonneg_right hcoef_le (sq_nonneg _)).trans hraw2
  have hA2 :
      (gamma / (2 * Xmax)) * ((r2 k) ^ 2 / Anorm ^ 2) ≤ F (k + 1) - Fhalf k :=
    halfStepAscent_paperConstant_of_primalChangeBound
      (change := l1Norm (fun i => p2 k i - q2 k i))
      (residual := r2 k) (Fbefore := Fhalf k) (Fafter := F (k + 1))
      (gamma := gamma) (Xmax := Xmax) (Anorm := Anorm)
      hgamma_nonneg hXmax_pos hAnorm_pos (hr2_nonneg k) (hr2_bound k) hchange2
  exact ⟨hA1, hA2⟩

/--
One-sweep bridge from exact-gain support-aware finite block-update certificates.
-/
theorem twoBlockAscent_of_finiteMassShellExactSupportBlockUpdateCertificates_commonMass
    {n₁ n₂ : ℕ}
    {p1 q1 : Fin n₁ → ℝ} {p2 q2 : Fin n₂ → ℝ} {Fbefore Fhalf Fafter M : ℝ}
    (hMpos : 0 < M)
    (hcert1 : FiniteMassShellExactSupportBlockUpdateCertificate p1 q1 M Fbefore Fhalf)
    (hcert2 : FiniteMassShellExactSupportBlockUpdateCertificate p2 q2 M Fhalf Fafter) :
    (1 / (2 * M)) *
        ((l1Norm (fun i => p1 i - q1 i)) ^ 2
          + (l1Norm (fun i => p2 i - q2 i)) ^ 2)
      ≤ Fafter - Fbefore :=
  twoBlockAscent_of_finiteMassShellSupportBlockUpdateCertificates_commonMass
    (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2) (M := M) hMpos
    (finiteMassShellSupportBlockUpdateCertificate_of_exactSupportBlockUpdateCertificate hcert1)
    (finiteMassShellSupportBlockUpdateCertificate_of_exactSupportBlockUpdateCertificate hcert2)

/--
Full-sweep bridge from exact objective-gain finite block-update certificates.
-/
theorem perStepAscent_residualProxy_of_finiteMassShellExactBlockUpdateCertificates_commonMass
    {n₁ n₂ : ℕ}
    {F Fhalf : ℕ → ℝ}
    {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hcert1 :
      ∀ k : ℕ,
        FiniteMassShellExactBlockUpdateCertificate (p1 k) (q1 k) M (F k) (Fhalf k))
    (hcert2 :
      ∀ k : ℕ,
        FiniteMassShellExactBlockUpdateCertificate (p2 k) (q2 k) M (Fhalf k) (F (k + 1))) :
    ∀ k : ℕ,
      (1 / (2 * M)) *
          ((l1Norm (fun i => p1 k i - q1 k i)) ^ 2
            + (l1Norm (fun i => p2 k i - q2 k i)) ^ 2)
        ≤ F (k + 1) - F k :=
  perStepAscent_residualProxy_of_finiteMassShellBlockUpdateCertificates_commonMass
    (F := F) (Fhalf := Fhalf) (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2)
    (M := M) hMpos
    (fun k => finiteMassShellBlockUpdateCertificate_of_exactBlockUpdateCertificate (hcert1 k))
    (fun k => finiteMassShellBlockUpdateCertificate_of_exactBlockUpdateCertificate (hcert2 k))

/--
One-sweep bridge from exact objective-gain finite block-update certificates.
-/
theorem twoBlockAscent_of_finiteMassShellExactBlockUpdateCertificates_commonMass
    {n₁ n₂ : ℕ}
    {p1 q1 : Fin n₁ → ℝ} {p2 q2 : Fin n₂ → ℝ} {Fbefore Fhalf Fafter M : ℝ}
    (hMpos : 0 < M)
    (hcert1 : FiniteMassShellExactBlockUpdateCertificate p1 q1 M Fbefore Fhalf)
    (hcert2 : FiniteMassShellExactBlockUpdateCertificate p2 q2 M Fhalf Fafter) :
    (1 / (2 * M)) *
        ((l1Norm (fun i => p1 i - q1 i)) ^ 2
          + (l1Norm (fun i => p2 i - q2 i)) ^ 2)
      ≤ Fafter - Fbefore :=
  twoBlockAscent_of_finiteMassShellBlockUpdateCertificates_commonMass
    (p1 := p1) (q1 := q1) (p2 := p2) (q2 := q2) (M := M) hMpos
    (finiteMassShellBlockUpdateCertificate_of_exactBlockUpdateCertificate hcert1)
    (finiteMassShellBlockUpdateCertificate_of_exactBlockUpdateCertificate hcert2)

/--
Successor-index cumulative bound from per-step ascent and a direct delta-budget.

This variant takes the objective-growth assumption in the common downstream form
`phi (n+1) - phi 0 ≤ B`, avoiding repeated `linarith` reshaping.
-/
theorem perStepAscent_cumulative_succ_bounded_of_deltaBound
    {gap phi : ℕ → ℝ} {B : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (n : ℕ) :
    cumulative gap (n + 1) ≤ B := by
  have hper : cumulative gap (n + 1) ≤ phi (n + 1) - phi 0 :=
    perStepAscent_twoStep (phi := phi) (residual := gap) hstep (n + 1)
  exact hper.trans (hphi_bound n)

/--
Normalization by `n+1`: any successor-index cumulative bound yields an averaged bound.
-/
theorem cumulative_div_succ_le_div_succ
    {a : ℕ → ℝ} {B : ℝ}
    (n : ℕ)
    (hcum : cumulative a (n + 1) ≤ B) :
    cumulative a (n + 1) / (n + 1 : ℝ) ≤ B / (n + 1 : ℝ) := by
  have hpos : 0 < (n + 1 : ℝ) := by
    exact_mod_cast Nat.succ_pos n
  have hmul :
      cumulative a (n + 1) * (n + 1 : ℝ)⁻¹ ≤ B * (n + 1 : ℝ)⁻¹ :=
    mul_le_mul_of_nonneg_right hcum (inv_nonneg.mpr (le_of_lt hpos))
  simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using hmul

/--
Rate-ready arithmetic consequence: antitone term control from a cumulative budget.

If `g` is antitone and `cumulative g (n+1) ≤ B`, then `g n ≤ B/(n+1)`.
-/
theorem le_div_of_antitone_of_cumulative_succ_bound
    {g : ℕ → ℝ} {B : ℝ}
    (hmono : Antitone g)
    {n : ℕ}
    (hcum : cumulative g (n + 1) ≤ B) :
    g n ≤ B / (n + 1 : ℝ) := by
  have hmul : ((n + 1 : ℝ) * g n) ≤ B :=
    (mul_le_cumulative_of_antitone hmono n).trans hcum
  have hpos : 0 < (n + 1 : ℝ) := by
    exact_mod_cast Nat.succ_pos n
  have hmul' : g n * (n + 1 : ℝ) ≤ B := by
    simpa [mul_comm, mul_left_comm, mul_assoc] using hmul
  exact (le_div_iff₀ hpos).2 hmul'

/--
Composed per-step-ascent rate lemma in `O(1/(n+1))` form.

This packages the standard chain:
per-step ascent + bounded objective growth + antitone gap
`⇒ gap n ≤ B/(n+1)`.
-/
theorem perStepAscent_rateReady_of_antitone
    {gap phi : ℕ → ℝ} {B : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hbounded : ∀ n : ℕ, phi (n + 1) ≤ phi 0 + B)
    (hmono : Antitone gap)
    (n : ℕ) :
    gap n ≤ B / (n + 1 : ℝ) := by
  have hcum : cumulative gap (n + 1) ≤ B :=
    perStepAscent_cumulative_succ_bounded hstep hbounded n
  exact le_div_of_antitone_of_cumulative_succ_bound hmono hcum

/--
Successor-index cumulative bound derived from the base cumulative theorem.

This is a compatibility bridge for call sites that already carry `0 ≤ B` and
prefer to reuse `perStepAscent_cumulative_bounded`.
-/
theorem perStepAscent_cumulative_succ_bounded_of_nonneg
    {gap phi : ℕ → ℝ} {B : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hbounded : ∀ n : ℕ, phi (n + 1) ≤ phi 0 + B)
    (hB : 0 ≤ B)
    (n : ℕ) :
    cumulative gap (n + 1) ≤ B := by
  simpa using
    perStepAscent_cumulative_bounded
      (gap := gap) (phi := phi) (B := B) hstep hbounded hB (n + 1)

/--
Normalized successor-index cumulative ascent bound.

This exports the average form directly from per-step ascent and objective budget.
-/
theorem perStepAscent_normalizedCumulative_succ_bounded
    {gap phi : ℕ → ℝ} {B : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hbounded : ∀ n : ℕ, phi (n + 1) ≤ phi 0 + B)
    (n : ℕ) :
    cumulative gap (n + 1) / (n + 1 : ℝ) ≤ B / (n + 1 : ℝ) :=
  cumulative_div_succ_le_div_succ n
    (perStepAscent_cumulative_succ_bounded hstep hbounded n)

/--
Normalized successor-index cumulative ascent bound from a direct delta-budget.

This is the average-form companion of
`perStepAscent_cumulative_succ_bounded_of_deltaBound`.
-/
theorem perStepAscent_normalizedCumulative_succ_bounded_of_deltaBound
    {gap phi : ℕ → ℝ} {B : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (n : ℕ) :
    cumulative gap (n + 1) / (n + 1 : ℝ) ≤ B / (n + 1 : ℝ) :=
  cumulative_div_succ_le_div_succ n
    (perStepAscent_cumulative_succ_bounded_of_deltaBound hstep hphi_bound n)

/--
Nonnegativity-lifted normalized cumulative bound.

This converts the normalized successor bound into a direct budget bound
`cumulative gap (n+1)/(n+1) ≤ B` when `B ≥ 0`.
-/
theorem perStepAscent_normalizedCumulative_succ_le_budget_of_nonneg
    {gap phi : ℕ → ℝ} {B : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hbounded : ∀ n : ℕ, phi (n + 1) ≤ phi 0 + B)
    (hB : 0 ≤ B)
    (n : ℕ) :
    cumulative gap (n + 1) / (n + 1 : ℝ) ≤ B := by
  have hnorm :
      cumulative gap (n + 1) / (n + 1 : ℝ) ≤ B / (n + 1 : ℝ) :=
    perStepAscent_normalizedCumulative_succ_bounded hstep hbounded n
  have hdiv : B / (n + 1 : ℝ) ≤ B := by
    apply div_le_self hB
    exact le_add_of_nonneg_left (Nat.cast_nonneg n)
  exact hnorm.trans hdiv

/--
Composed per-step-ascent rate lemma from a direct delta-budget.

This is the `O(1/(n+1))` endpoint with assumptions in the form
`phi (n+1) - phi 0 ≤ B`.
-/
theorem perStepAscent_rateReady_of_antitone_of_deltaBound
    {gap phi : ℕ → ℝ} {B : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono : Antitone gap)
    (n : ℕ) :
    gap n ≤ B / (n + 1 : ℝ) := by
  have hcum : cumulative gap (n + 1) ≤ B :=
    perStepAscent_cumulative_succ_bounded_of_deltaBound hstep hphi_bound n
  exact le_div_of_antitone_of_cumulative_succ_bound hmono hcum

/--
Successor-index `O(1/k)` rate form.

Convenience wrapper of `perStepAscent_rateReady_of_antitone` at index `n+1`.
-/
theorem perStepAscent_rateReady_of_antitone_succ
    {gap phi : ℕ → ℝ} {B : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hbounded : ∀ n : ℕ, phi (n + 1) ≤ phi 0 + B)
    (hmono : Antitone gap)
    (n : ℕ) :
    gap (n + 1) ≤ B / (((n + 1) + 1 : ℕ) : ℝ) := by
  simpa using
    perStepAscent_rateReady_of_antitone hstep hbounded hmono (n + 1)

/--
Successor-index `O(1/k)` rate form from a direct delta-budget.

This is the `(n+1)` wrapper of
`perStepAscent_rateReady_of_antitone_of_deltaBound`.
-/
theorem perStepAscent_rateReady_of_antitone_succ_of_deltaBound
    {gap phi : ℕ → ℝ} {B : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono : Antitone gap)
    (n : ℕ) :
    gap (n + 1) ≤ B / (((n + 1) + 1 : ℕ) : ℝ) := by
  simpa using
    perStepAscent_rateReady_of_antitone_of_deltaBound hstep hphi_bound hmono (n + 1)

/--
`ε`-accuracy from the per-step-ascent rate via a ratio-form threshold.

If `(B / eps) ≤ n+1` with `eps > 0`, then the `O(1/(n+1))` bound yields `gap n ≤ eps`.
-/
theorem perStepAscent_rateReady_of_antitone_le_eps_of_ratioBound
    {gap phi : ℕ → ℝ} {B eps : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hbounded : ∀ n : ℕ, phi (n + 1) ≤ phi 0 + B)
    (hmono : Antitone gap)
    (heps : 0 < eps)
    (n : ℕ)
    (hratio : B / eps ≤ (n + 1 : ℝ)) :
    gap n ≤ eps := by
  have hrate : gap n ≤ B / (n + 1 : ℝ) :=
    perStepAscent_rateReady_of_antitone hstep hbounded hmono n
  have hmul' : B ≤ (n + 1 : ℝ) * eps :=
    (div_le_iff₀ heps).1 hratio
  have hmul : B ≤ eps * (n + 1 : ℝ) := by
    simpa [mul_comm, mul_left_comm, mul_assoc] using hmul'
  have hpos : 0 < (n + 1 : ℝ) := by
    exact_mod_cast Nat.succ_pos n
  have hdiv : B / (n + 1 : ℝ) ≤ eps :=
    (div_le_iff₀ hpos).2 hmul
  exact hrate.trans hdiv

/--
`ε`-accuracy from the per-step-ascent rate via a closed-form ceil threshold.
-/
theorem perStepAscent_rateReady_of_antitone_le_eps_of_natCeil_threshold
    {gap phi : ℕ → ℝ} {B eps : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hbounded : ∀ n : ℕ, phi (n + 1) ≤ phi 0 + B)
    (hmono : Antitone gap)
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil (B / eps) ≤ n + 1) :
    gap n ≤ eps := by
  have hratio : B / eps ≤ (n + 1 : ℝ) := by
    simpa [Nat.cast_add, Nat.cast_one] using
      (Nat.le_of_ceil_le hn : B / eps ≤ (n + 1 : ℕ))
  exact perStepAscent_rateReady_of_antitone_le_eps_of_ratioBound
    hstep hbounded hmono heps n hratio

/--
Successor-index `ε`-accuracy from the ratio-form threshold.

This is the `(n+1)` wrapper of
`perStepAscent_rateReady_of_antitone_le_eps_of_ratioBound`.
-/
theorem perStepAscent_rateReady_of_antitone_succ_le_eps_of_ratioBound
    {gap phi : ℕ → ℝ} {B eps : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hbounded : ∀ n : ℕ, phi (n + 1) ≤ phi 0 + B)
    (hmono : Antitone gap)
    (heps : 0 < eps)
    (n : ℕ)
    (hratio : B / eps ≤ (((n + 1 : ℕ) : ℝ) + 1)) :
    gap (n + 1) ≤ eps := by
  exact perStepAscent_rateReady_of_antitone_le_eps_of_ratioBound
    hstep hbounded hmono heps (n + 1) hratio

/--
Successor-index `ε`-accuracy from the closed-form ceil threshold.

This is the `(n+1)` wrapper of
`perStepAscent_rateReady_of_antitone_le_eps_of_natCeil_threshold`.
-/
theorem perStepAscent_rateReady_of_antitone_succ_le_eps_of_natCeil_threshold
    {gap phi : ℕ → ℝ} {B eps : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hbounded : ∀ n : ℕ, phi (n + 1) ≤ phi 0 + B)
    (hmono : Antitone gap)
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil (B / eps) ≤ (n + 1) + 1) :
    gap (n + 1) ≤ eps :=
  perStepAscent_rateReady_of_antitone_le_eps_of_natCeil_threshold
    hstep hbounded hmono heps (n + 1) hn

/--
Index-threshold convenience form of
`perStepAscent_rateReady_of_antitone_le_eps_of_ratioBound`.
-/
theorem perStepAscent_rateReady_of_antitone_le_eps_of_ratioBound_of_le_index
    {gap phi : ℕ → ℝ} {B eps : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hbounded : ∀ n : ℕ, phi (n + 1) ≤ phi 0 + B)
    (hmono : Antitone gap)
    (heps : 0 < eps)
    {k n : ℕ} (hk : k ≤ n)
    (hratio : B / eps ≤ (k + 1 : ℝ)) :
    gap n ≤ eps := by
  have hidx : (k + 1 : ℝ) ≤ (n + 1 : ℝ) := by
    exact_mod_cast Nat.succ_le_succ hk
  have hratio' : B / eps ≤ (n + 1 : ℝ) := hratio.trans hidx
  exact perStepAscent_rateReady_of_antitone_le_eps_of_ratioBound
    hstep hbounded hmono heps n hratio'

/--
Successor-index + index-threshold convenience form of
`perStepAscent_rateReady_of_antitone_le_eps_of_ratioBound`.
-/
theorem perStepAscent_rateReady_of_antitone_succ_le_eps_of_ratioBound_of_le_index
    {gap phi : ℕ → ℝ} {B eps : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hbounded : ∀ n : ℕ, phi (n + 1) ≤ phi 0 + B)
    (hmono : Antitone gap)
    (heps : 0 < eps)
    {k n : ℕ} (hk : k ≤ n)
    (hratio : B / eps ≤ (((k + 1 : ℕ) : ℝ) + 1)) :
    gap (n + 1) ≤ eps := by
  have hk' : k + 1 ≤ n + 1 := Nat.succ_le_succ hk
  exact perStepAscent_rateReady_of_antitone_le_eps_of_ratioBound_of_le_index
    hstep hbounded hmono heps (k := k + 1) (n := n + 1) hk' hratio

/--
Index-threshold convenience form of
`perStepAscent_rateReady_of_antitone_le_eps_of_natCeil_threshold`.
-/
theorem perStepAscent_rateReady_of_antitone_le_eps_of_natCeil_threshold_of_le_index
    {gap phi : ℕ → ℝ} {B eps : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hbounded : ∀ n : ℕ, phi (n + 1) ≤ phi 0 + B)
    (hmono : Antitone gap)
    (heps : 0 < eps)
    {k n : ℕ} (hk : k ≤ n)
    (hn : Nat.ceil (B / eps) ≤ k + 1) :
    gap n ≤ eps := by
  have hn' : Nat.ceil (B / eps) ≤ n + 1 :=
    hn.trans (Nat.succ_le_succ hk)
  exact perStepAscent_rateReady_of_antitone_le_eps_of_natCeil_threshold
    hstep hbounded hmono heps n hn'

/--
Successor-index + index-threshold convenience form of
`perStepAscent_rateReady_of_antitone_le_eps_of_natCeil_threshold`.
-/
theorem perStepAscent_rateReady_of_antitone_succ_le_eps_of_natCeil_threshold_of_le_index
    {gap phi : ℕ → ℝ} {B eps : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hbounded : ∀ n : ℕ, phi (n + 1) ≤ phi 0 + B)
    (hmono : Antitone gap)
    (heps : 0 < eps)
    {k n : ℕ} (hk : k ≤ n)
    (hn : Nat.ceil (B / eps) ≤ (k + 1) + 1) :
    gap (n + 1) ≤ eps := by
  have hk' : k + 1 ≤ n + 1 := Nat.succ_le_succ hk
  exact perStepAscent_rateReady_of_antitone_le_eps_of_natCeil_threshold_of_le_index
    hstep hbounded hmono heps (k := k + 1) (n := n + 1) hk' hn

/--
`ε`-accuracy from the delta-budget per-step-ascent rate via a ratio-form threshold.
-/
theorem perStepAscent_rateReady_of_antitone_of_deltaBound_le_eps_of_ratioBound
    {gap phi : ℕ → ℝ} {B eps : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono : Antitone gap)
    (heps : 0 < eps)
    (n : ℕ)
    (hratio : B / eps ≤ (n + 1 : ℝ)) :
    gap n ≤ eps := by
  have hrate : gap n ≤ B / (n + 1 : ℝ) :=
    perStepAscent_rateReady_of_antitone_of_deltaBound hstep hphi_bound hmono n
  have hmul' : B ≤ (n + 1 : ℝ) * eps :=
    (div_le_iff₀ heps).1 hratio
  have hmul : B ≤ eps * (n + 1 : ℝ) := by
    simpa [mul_comm, mul_left_comm, mul_assoc] using hmul'
  have hpos : 0 < (n + 1 : ℝ) := by
    exact_mod_cast Nat.succ_pos n
  have hdiv : B / (n + 1 : ℝ) ≤ eps := (div_le_iff₀ hpos).2 hmul
  exact hrate.trans hdiv

/--
Successor-index `ε`-accuracy from the delta-budget ratio-form threshold.

This is the `(n+1)` wrapper of
`perStepAscent_rateReady_of_antitone_of_deltaBound_le_eps_of_ratioBound`.
-/
theorem perStepAscent_rateReady_of_antitone_succ_of_deltaBound_le_eps_of_ratioBound
    {gap phi : ℕ → ℝ} {B eps : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono : Antitone gap)
    (heps : 0 < eps)
    (n : ℕ)
    (hratio : B / eps ≤ (((n + 1 : ℕ) : ℝ) + 1)) :
    gap (n + 1) ≤ eps := by
  exact perStepAscent_rateReady_of_antitone_of_deltaBound_le_eps_of_ratioBound
    hstep hphi_bound hmono heps (n + 1) hratio

/--
Index-threshold convenience form of
`perStepAscent_rateReady_of_antitone_of_deltaBound_le_eps_of_ratioBound`.
-/
theorem perStepAscent_rateReady_of_antitone_of_deltaBound_le_eps_of_ratioBound_of_le_index
    {gap phi : ℕ → ℝ} {B eps : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono : Antitone gap)
    (heps : 0 < eps)
    {k n : ℕ} (hk : k ≤ n)
    (hratio : B / eps ≤ (k + 1 : ℝ)) :
    gap n ≤ eps := by
  have hidx : (k + 1 : ℝ) ≤ (n + 1 : ℝ) := by
    exact_mod_cast Nat.succ_le_succ hk
  have hratio' : B / eps ≤ (n + 1 : ℝ) := hratio.trans hidx
  exact perStepAscent_rateReady_of_antitone_of_deltaBound_le_eps_of_ratioBound
    hstep hphi_bound hmono heps n hratio'

/--
Successor-index + index-threshold convenience form of
`perStepAscent_rateReady_of_antitone_of_deltaBound_le_eps_of_ratioBound`.
-/
theorem
    perStepAscent_rateReady_of_antitone_succ_of_deltaBound_le_eps_of_ratioBound_of_le_index
    {gap phi : ℕ → ℝ} {B eps : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono : Antitone gap)
    (heps : 0 < eps)
    {k n : ℕ} (hk : k ≤ n)
    (hratio : B / eps ≤ (((k + 1 : ℕ) : ℝ) + 1)) :
    gap (n + 1) ≤ eps := by
  have hk' : k + 1 ≤ n + 1 := Nat.succ_le_succ hk
  exact perStepAscent_rateReady_of_antitone_of_deltaBound_le_eps_of_ratioBound_of_le_index
    hstep hphi_bound hmono heps (k := k + 1) (n := n + 1) hk' hratio

/--
Natural-bound companion of
`perStepAscent_rateReady_of_antitone_of_deltaBound_le_eps_of_ratioBound_of_le_index`.
-/
theorem perStepAscent_rateReady_of_antitone_of_deltaBound_le_eps_of_ratioBound_of_natBound
    {gap phi : ℕ → ℝ} {B eps : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono : Antitone gap)
    (heps : 0 < eps)
    (n N : ℕ)
    (hratioN : B / eps ≤ (N + 1 : ℝ))
    (hNn : N ≤ n) :
    gap n ≤ eps :=
  perStepAscent_rateReady_of_antitone_of_deltaBound_le_eps_of_ratioBound_of_le_index
    hstep hphi_bound hmono heps (k := N) (n := n) hNn hratioN

/--
Successor-index natural-bound companion of
`perStepAscent_rateReady_of_antitone_succ_of_deltaBound_le_eps_of_ratioBound_of_le_index`.
-/
theorem perStepAscent_rateReady_of_antitone_succ_of_deltaBound_le_eps_of_ratioBound_of_natBound
    {gap phi : ℕ → ℝ} {B eps : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono : Antitone gap)
    (heps : 0 < eps)
    (n N : ℕ)
    (hratioN : B / eps ≤ (((N + 1 : ℕ) : ℝ) + 1))
    (hNn : N ≤ n) :
    gap (n + 1) ≤ eps :=
  perStepAscent_rateReady_of_antitone_succ_of_deltaBound_le_eps_of_ratioBound_of_le_index
    hstep hphi_bound hmono heps (k := N) (n := n) hNn hratioN

/--
`ε`-accuracy from the delta-budget per-step-ascent rate via a closed-form ceil threshold.
-/
theorem perStepAscent_rateReady_of_antitone_of_deltaBound_le_eps_of_natCeil_threshold
    {gap phi : ℕ → ℝ} {B eps : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono : Antitone gap)
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil (B / eps) ≤ n + 1) :
    gap n ≤ eps := by
  have hratio : B / eps ≤ (n + 1 : ℝ) := by
    simpa [Nat.cast_add, Nat.cast_one] using
      (Nat.le_of_ceil_le hn : B / eps ≤ (n + 1 : ℕ))
  have hrate : gap n ≤ B / (n + 1 : ℝ) :=
    perStepAscent_rateReady_of_antitone_of_deltaBound hstep hphi_bound hmono n
  have hmul' : B ≤ (n + 1 : ℝ) * eps :=
    (div_le_iff₀ heps).1 hratio
  have hmul : B ≤ eps * (n + 1 : ℝ) := by
    simpa [mul_comm, mul_left_comm, mul_assoc] using hmul'
  have hpos : 0 < (n + 1 : ℝ) := by
    exact_mod_cast Nat.succ_pos n
  have hdiv : B / (n + 1 : ℝ) ≤ eps := (div_le_iff₀ hpos).2 hmul
  exact hrate.trans hdiv

/--
Index-threshold convenience form of
`perStepAscent_rateReady_of_antitone_of_deltaBound_le_eps_of_natCeil_threshold`.
-/
theorem
    perStepAscent_rateReady_of_antitone_of_deltaBound_le_eps_of_natCeil_threshold_of_le_index
    {gap phi : ℕ → ℝ} {B eps : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono : Antitone gap)
    (heps : 0 < eps)
    {k n : ℕ} (hk : k ≤ n)
    (hn : Nat.ceil (B / eps) ≤ k + 1) :
    gap n ≤ eps := by
  have hn' : Nat.ceil (B / eps) ≤ n + 1 :=
    hn.trans (Nat.succ_le_succ hk)
  exact perStepAscent_rateReady_of_antitone_of_deltaBound_le_eps_of_natCeil_threshold
    hstep hphi_bound hmono heps n hn'

/--
Successor-index + index-threshold convenience form of
`perStepAscent_rateReady_of_antitone_of_deltaBound_le_eps_of_natCeil_threshold`.
-/
theorem
    perStepAscent_rateReady_of_antitone_succ_of_deltaBound_le_eps_of_natCeil_threshold_of_le_index
    {gap phi : ℕ → ℝ} {B eps : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono : Antitone gap)
    (heps : 0 < eps)
    {k n : ℕ} (hk : k ≤ n)
    (hn : Nat.ceil (B / eps) ≤ (k + 1) + 1) :
    gap (n + 1) ≤ eps := by
  have hk' : k + 1 ≤ n + 1 := Nat.succ_le_succ hk
  exact
    perStepAscent_rateReady_of_antitone_of_deltaBound_le_eps_of_natCeil_threshold_of_le_index
      hstep hphi_bound hmono heps (k := k + 1) (n := n + 1) hk' hn

/--
Successor-index `ε`-accuracy from the delta-budget ceil-threshold bound.
-/
theorem perStepAscent_rateReady_of_antitone_succ_of_deltaBound_le_eps_of_natCeil_threshold
    {gap phi : ℕ → ℝ} {B eps : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono : Antitone gap)
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil (B / eps) ≤ (n + 1) + 1) :
    gap (n + 1) ≤ eps :=
  perStepAscent_rateReady_of_antitone_of_deltaBound_le_eps_of_natCeil_threshold
    hstep hphi_bound hmono heps (n + 1) hn

/--
Zero-index specialization of the rate-ready bound.
-/
theorem perStepAscent_rateReady_of_antitone_zero
    {gap phi : ℕ → ℝ} {B : ℝ}
    (hstep : ∀ k : ℕ, gap k ≤ phi (k + 1) - phi k)
    (hbounded : ∀ n : ℕ, phi (n + 1) ≤ phi 0 + B)
    (hmono : Antitone gap) :
    gap 0 ≤ B := by
  simpa using perStepAscent_rateReady_of_antitone hstep hbounded hmono 0

end DualConvergence
end KLProjection
end FlowSinkhorn
