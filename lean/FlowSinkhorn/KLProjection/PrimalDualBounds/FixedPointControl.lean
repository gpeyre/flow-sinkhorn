import FlowSinkhorn.KLProjection.PrimalDualBounds.Vocabulary
import FlowSinkhorn.KLProjection.PrimalDualBounds.PrimalFromDual
import FlowSinkhorn.KLProjection.UniformBound
import Mathlib.Analysis.Seminorm
import Mathlib.Tactic

/-!
# Fixed-point control via Hγ and κ

This module defines the orbit-budget formula `hGammaKappaBudget κ cost γ Hγ = κ·(cost + γ·Hγ)`
and proves the generic uniform iterate bound:

  `p (Psi^[k] u0) ≤ p u0 + 2 · hGammaKappaBudget κ cost γ Hγ`

under the assumption that `Psi` is `SeminormNonexpansive p` and `p uStar ≤ hGammaKappaBudget`.

## Note on reuse

This module is completely independent of the specific seminorm, the algorithm, and the
application (OT, graph W1, etc.). Applications only need to provide:
- A `SeminormNonexpansive` proof (via `Topical.lean` → `Blueprint.lean` bridge);
- A fixed-point estimate `p uStar ≤ κ·(cost + γ·Hγ)`.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace PrimalDualBounds

open Function

variable {𝕜 E : Type*}
variable [NormedField 𝕜] [AddCommGroup E] [Module 𝕜 E]

/--
Paper-facing name for the optimizer-specific fixed-point estimate from `H_γ` and `κ`.

At this stage it is a packaging theorem: the application modules are expected to provide the
actual estimate and can then hand it to the generic iterate bound theorem below.
-/
theorem fixedPointBound_of_HGamma_kappa
    (p : Seminorm 𝕜 E) {uStar : E}
    {kappa cost gamma hGamma : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma) :
    p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma := by
  calc
    p uStar ≤ p uStar := le_rfl
    _ ≤ hGammaKappaBudget kappa cost gamma hGamma := hbound

/--
Uniform iterate bound in the paper shape:
non-expansiveness + fixed-point control by `κ (cost + γ H_γ)` imply a uniform orbit estimate.
-/
theorem uniformIterateBound_of_nonexpansive_of_budget
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {B : ℝ}
    (hbound : p uStar ≤ B)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ p u0 + 2 * B := by
  have horbit :
      p ((Psi^[k]) u0) ≤ p u0 + 2 * B :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
      p Psi hPsi hfix hbound k
  exact horbit

/--
Uniform iterate bound in the paper shape:
non-expansiveness + fixed-point control by `κ (cost + γ H_γ)` imply a uniform orbit estimate.
-/
theorem uniformIterateBound_of_nonexpansive_of_HGamma_kappa
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
Equivalent expanded form with `κ (cost + γ H_γ)` written directly.
-/
theorem uniformIterateBound_of_nonexpansive_of_HGamma_kappa_expanded
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma : ℝ}
    (hbound : p uStar ≤ kappa * (cost + gamma * hGamma))
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ p u0 + 2 * (kappa * (cost + gamma * hGamma)) := by
  simpa [hGammaKappaBudget] using
    uniformIterateBound_of_nonexpansive_of_HGamma_kappa
      p Psi hPsi hfix hbound k

/--
Monotonicity of `hGammaKappaBudget` with respect to `H_γ` under nonnegative `κ` and `γ`.
-/
theorem hGammaKappaBudget_le_of_hGamma_le
    {kappa cost gamma hGamma hGammaUpper : ℝ}
    (hkappa : 0 ≤ kappa)
    (hgamma : 0 ≤ gamma)
    (hH : hGamma ≤ hGammaUpper) :
    hGammaKappaBudget kappa cost gamma hGamma ≤
      hGammaKappaBudget kappa cost gamma hGammaUpper := by
  dsimp [hGammaKappaBudget]
  have hmul : gamma * hGamma ≤ gamma * hGammaUpper :=
    mul_le_mul_of_nonneg_left hH hgamma
  linarith [mul_le_mul_of_nonneg_left hmul hkappa]

/--
Monotonicity of `hGammaKappaBudget` with respect to `κ` under a nonnegative inner term.
-/
theorem hGammaKappaBudget_le_of_kappa_le
    {kappa kappaUpper cost gamma hGamma : ℝ}
    (hinnerNonneg : 0 ≤ cost + gamma * hGamma)
    (hkappa : kappa ≤ kappaUpper) :
    hGammaKappaBudget kappa cost gamma hGamma ≤
      hGammaKappaBudget kappaUpper cost gamma hGamma := by
  dsimp [hGammaKappaBudget]
  exact mul_le_mul_of_nonneg_right hkappa hinnerNonneg

/--
Joint monotonicity bridge: increasing both `κ` and `H_γ` increases `hGammaKappaBudget`
under the natural nonnegativity side conditions.
-/
theorem hGammaKappaBudget_le_of_kappa_le_and_hGamma_le
    {kappa kappaUpper cost gamma hGamma hGammaUpper : ℝ}
    (hkappaNonneg : 0 ≤ kappa)
    (hgamma : 0 ≤ gamma)
    (hkappa : kappa ≤ kappaUpper)
    (hH : hGamma ≤ hGammaUpper)
    (hinnerUpperNonneg : 0 ≤ cost + gamma * hGammaUpper) :
    hGammaKappaBudget kappa cost gamma hGamma ≤
      hGammaKappaBudget kappaUpper cost gamma hGammaUpper := by
  have hHmono :
      hGammaKappaBudget kappa cost gamma hGamma ≤
        hGammaKappaBudget kappa cost gamma hGammaUpper :=
    hGammaKappaBudget_le_of_hGamma_le hkappaNonneg hgamma hH
  have hkmono :
      hGammaKappaBudget kappa cost gamma hGammaUpper ≤
        hGammaKappaBudget kappaUpper cost gamma hGammaUpper :=
    hGammaKappaBudget_le_of_kappa_le hinnerUpperNonneg hkappa
  exact hHmono.trans hkmono

/--
Monotonicity of `hGammaKappaBudget` through an upper bound on its inner term.

If `cost + γ H_γ ≤ C` and `κ ≥ 0`, then `κ (cost + γ H_γ) ≤ κ C`.
-/
theorem hGammaKappaBudget_le_of_inner_le
    {kappa cost gamma hGamma C : ℝ}
    (hkappa : 0 ≤ kappa)
    (hinner : cost + gamma * hGamma ≤ C) :
    hGammaKappaBudget kappa cost gamma hGamma ≤ kappa * C := by
  dsimp [hGammaKappaBudget]
  exact mul_le_mul_of_nonneg_left hinner hkappa

/--
Transfer a fixed-point bound through any upper bound on `hGammaKappaBudget`.
-/
theorem fixedPointBound_of_HGamma_kappa_of_budget_le
    (p : Seminorm 𝕜 E) {uStar : E}
    {kappa cost gamma hGamma B : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hbudget : hGammaKappaBudget kappa cost gamma hGamma ≤ B) :
    p uStar ≤ B :=
  hbound.trans hbudget

/--
Transfer the fixed-point bound through an upper bound on the inner term `cost + γ H_γ`.
-/
theorem fixedPointBound_of_HGamma_kappa_of_inner_le
    (p : Seminorm 𝕜 E) {uStar : E}
    {kappa cost gamma hGamma C : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hkappa : 0 ≤ kappa)
    (hinner : cost + gamma * hGamma ≤ C) :
    p uStar ≤ kappa * C :=
  fixedPointBound_of_HGamma_kappa_of_budget_le
    p hbound (hGammaKappaBudget_le_of_inner_le hkappa hinner)

/--
Transfer the fixed-point bound through an upper bound on `H_γ`.
-/
theorem fixedPointBound_of_HGamma_kappa_of_hGamma_le
    (p : Seminorm 𝕜 E) {uStar : E}
    {kappa cost gamma hGamma hGammaUpper : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hkappa : 0 ≤ kappa)
    (hgamma : 0 ≤ gamma)
    (hH : hGamma ≤ hGammaUpper) :
    p uStar ≤ hGammaKappaBudget kappa cost gamma hGammaUpper :=
  fixedPointBound_of_HGamma_kappa_of_budget_le
    p hbound (hGammaKappaBudget_le_of_hGamma_le hkappa hgamma hH)

/--
Transfer the uniform iterate bound through any upper bound on `hGammaKappaBudget`.
-/
theorem uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_budget_le
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma B : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hbudget : hGammaKappaBudget kappa cost gamma hGamma ≤ B)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ p u0 + 2 * B := by
  have hiter :
      p ((Psi^[k]) u0) ≤ p u0 + 2 * hGammaKappaBudget kappa cost gamma hGamma :=
    uniformIterateBound_of_nonexpansive_of_HGamma_kappa
      p Psi hPsi hfix hbound k
  linarith

/--
Transfer the uniform iterate bound through an upper bound on `cost + γ H_γ`.

This is the practical bridge used when applications first prove
`cost + γ H_γ ≤ C` and keep `κ` as an explicit prefactor.
-/
theorem uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_inner_le
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma C : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hkappa : 0 ≤ kappa)
    (hinner : cost + gamma * hGamma ≤ C)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ p u0 + 2 * (kappa * C) :=
  uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_budget_le
    p Psi hPsi hfix hbound
    (hGammaKappaBudget_le_of_inner_le hkappa hinner) k

/--
Transfer the uniform iterate bound through an upper bound on `H_γ`.

This is the orbit-level companion to
`fixedPointBound_of_HGamma_kappa_of_hGamma_le`.
-/
theorem uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_hGamma_le
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma hGammaUpper : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hkappa : 0 ≤ kappa)
    (hgamma : 0 ≤ gamma)
    (hH : hGamma ≤ hGammaUpper)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ p u0 + 2 * hGammaKappaBudget kappa cost gamma hGammaUpper :=
  uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_budget_le
    p Psi hPsi hfix hbound
    (hGammaKappaBudget_le_of_hGamma_le hkappa hgamma hH) k

/--
Zero-seed specialization of the `H_γ`-monotone orbit bridge.
-/
theorem uniformIterateBound_from_zero_of_nonexpansive_of_HGamma_kappa_of_hGamma_le
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma hGammaUpper : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hkappa : 0 ≤ kappa)
    (hgamma : 0 ≤ gamma)
    (hH : hGamma ≤ hGammaUpper)
    (hzero : p u0 = 0)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ 2 * hGammaKappaBudget kappa cost gamma hGammaUpper := by
  have hiter :
      p ((Psi^[k]) u0) ≤ p u0 + 2 * hGammaKappaBudget kappa cost gamma hGammaUpper :=
    uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_hGamma_le
      p Psi hPsi hfix hbound hkappa hgamma hH k
  simpa [hzero] using hiter

/--
Zero-seed specialization of the inner-bound orbit bridge.
-/
theorem uniformIterateBound_from_zero_of_nonexpansive_of_HGamma_kappa_of_inner_le
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma C : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hkappa : 0 ≤ kappa)
    (hinner : cost + gamma * hGamma ≤ C)
    (hzero : p u0 = 0)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ 2 * (kappa * C) := by
  have hiter :
      p ((Psi^[k]) u0) ≤ p u0 + 2 * (kappa * C) :=
    uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_inner_le
      p Psi hPsi hfix hbound hkappa hinner k
  simpa [hzero] using hiter

/--
Zero-base specialization: if `cost + γ H_γ = 0`, then the orbit is bounded by `p u₀`.
-/
theorem uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_zero_base
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hzeroBase : cost + gamma * hGamma = 0)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ p u0 := by
  have hiter :
      p ((Psi^[k]) u0) ≤ p u0 + 2 * (kappa * (cost + gamma * hGamma)) :=
    uniformIterateBound_of_nonexpansive_of_HGamma_kappa_expanded
      p Psi hPsi hfix (by simpa [hGammaKappaBudget] using hbound) k
  simpa [hzeroBase] using hiter

/--
Zero-seed + zero-base consequence: the orbit seminorm is bounded by `0`.
-/
theorem uniformIterateBound_from_zero_of_nonexpansive_of_HGamma_kappa_of_zero_base
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hzeroBase : cost + gamma * hGamma = 0)
    (hzero : p u0 = 0)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ 0 := by
  have hiter :
      p ((Psi^[k]) u0) ≤ p u0 :=
    uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_zero_base
      p Psi hPsi hfix hbound hzeroBase k
  simpa [hzero] using hiter

/--
Zero-seed + zero-base consequence in equality form: the orbit seminorm stays exactly `0`.
-/
theorem uniformIterateEq_zero_of_nonexpansive_of_HGamma_kappa_of_zero_base
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hzeroBase : cost + gamma * hGamma = 0)
    (hzero : p u0 = 0)
    (k : ℕ) :
    p ((Psi^[k]) u0) = 0 := by
  have hle : p ((Psi^[k]) u0) ≤ 0 :=
    uniformIterateBound_from_zero_of_nonexpansive_of_HGamma_kappa_of_zero_base
      p Psi hPsi hfix hbound hzeroBase hzero k
  exact le_antisymm hle (apply_nonneg p ((Psi^[k]) u0))

/--
If the certified budget is nonpositive, each iterate is bounded by the initial seminorm.

This weakens the `zero_base` hypothesis to a direct budget sign condition.
-/
theorem uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_budget_nonpos
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hbudgetNonpos : hGammaKappaBudget kappa cost gamma hGamma ≤ 0)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ p u0 := by
  have hiter :
      p ((Psi^[k]) u0) ≤ p u0 + 2 * (0 : ℝ) :=
    uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_budget_le
      p Psi hPsi hfix hbound hbudgetNonpos k
  simpa using hiter

/--
Zero-seed consequence of a nonpositive budget: every iterate has seminorm at most `0`.
-/
theorem uniformIterateBound_from_zero_of_nonexpansive_of_HGamma_kappa_of_budget_nonpos
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hbudgetNonpos : hGammaKappaBudget kappa cost gamma hGamma ≤ 0)
    (hzero : p u0 = 0)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ 0 := by
  have hiter :
      p ((Psi^[k]) u0) ≤ p u0 :=
    uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_budget_nonpos
      p Psi hPsi hfix hbound hbudgetNonpos k
  simpa [hzero] using hiter

/--
Zero-seed + nonpositive-budget consequence in equality form.
-/
theorem uniformIterateEq_zero_of_nonexpansive_of_HGamma_kappa_of_budget_nonpos
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hbudgetNonpos : hGammaKappaBudget kappa cost gamma hGamma ≤ 0)
    (hzero : p u0 = 0)
    (k : ℕ) :
    p ((Psi^[k]) u0) = 0 := by
  have hle : p ((Psi^[k]) u0) ≤ 0 :=
    uniformIterateBound_from_zero_of_nonexpansive_of_HGamma_kappa_of_budget_nonpos
      p Psi hPsi hfix hbound hbudgetNonpos hzero k
  exact le_antisymm hle (apply_nonneg p ((Psi^[k]) u0))

/--
Successor-index form of the nonpositive-budget iterate control.
-/
theorem uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_budget_nonpos_succ
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hbudgetNonpos : hGammaKappaBudget kappa cost gamma hGamma ≤ 0)
    (k : ℕ) :
    p ((Psi^[k + 1]) u0) ≤ p u0 := by
  simpa [Nat.succ_eq_add_one] using
    uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_budget_nonpos
      p Psi hPsi hfix hbound hbudgetNonpos (k + 1)

/--
Successor-index zero-seed form of the nonpositive-budget iterate control.
-/
theorem uniformIterateBound_from_zero_of_nonexpansive_of_HGamma_kappa_of_budget_nonpos_succ
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hbudgetNonpos : hGammaKappaBudget kappa cost gamma hGamma ≤ 0)
    (hzero : p u0 = 0)
    (k : ℕ) :
    p ((Psi^[k + 1]) u0) ≤ 0 := by
  simpa [Nat.succ_eq_add_one] using
    uniformIterateBound_from_zero_of_nonexpansive_of_HGamma_kappa_of_budget_nonpos
      p Psi hPsi hfix hbound hbudgetNonpos hzero (k + 1)

/--
Successor-index equality form of the nonpositive-budget iterate control at zero seed.
-/
theorem uniformIterateEq_zero_of_nonexpansive_of_HGamma_kappa_of_budget_nonpos_succ
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hbudgetNonpos : hGammaKappaBudget kappa cost gamma hGamma ≤ 0)
    (hzero : p u0 = 0)
    (k : ℕ) :
    p ((Psi^[k + 1]) u0) = 0 := by
  simpa [Nat.succ_eq_add_one] using
    uniformIterateEq_zero_of_nonexpansive_of_HGamma_kappa_of_budget_nonpos
      p Psi hPsi hfix hbound hbudgetNonpos hzero (k + 1)

/--
Index-relaxed nonpositive-budget iterate control.

This is convenient when downstream arguments maintain an ambient upper index `kMax`
but only need the bound at some `k ≤ kMax`.
-/
theorem uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_budget_nonpos_of_le_index
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hbudgetNonpos : hGammaKappaBudget kappa cost gamma hGamma ≤ 0)
    (kMax k : ℕ)
    (_hk : k ≤ kMax) :
    p ((Psi^[k]) u0) ≤ p u0 := by
  exact uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_budget_nonpos
    p Psi hPsi hfix hbound hbudgetNonpos k

/--
Index-relaxed zero-seed nonpositive-budget iterate control.
-/
theorem uniformIterateBound_from_zero_of_nonexpansive_of_HGamma_kappa_of_budget_nonpos_of_le_index
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hbudgetNonpos : hGammaKappaBudget kappa cost gamma hGamma ≤ 0)
    (hzero : p u0 = 0)
    (kMax k : ℕ)
    (_hk : k ≤ kMax) :
    p ((Psi^[k]) u0) ≤ 0 := by
  exact uniformIterateBound_from_zero_of_nonexpansive_of_HGamma_kappa_of_budget_nonpos
    p Psi hPsi hfix hbound hbudgetNonpos hzero k

/--
`budget_le` + nonpositive target budget imply iterate control by the initial seminorm.

This bridges explicit upper-budget estimates `hGammaKappaBudget ≤ B` with sign information
`B ≤ 0`, avoiding a separate proof that `hGammaKappaBudget ≤ 0`.
-/
theorem uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_budget_le_of_nonpos
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma B : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hbudget : hGammaKappaBudget kappa cost gamma hGamma ≤ B)
    (hBnonpos : B ≤ 0)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ p u0 := by
  have hiter :
      p ((Psi^[k]) u0) ≤ p u0 + 2 * B :=
    uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_budget_le
      p Psi hPsi hfix hbound hbudget k
  have hshift : p u0 + 2 * B ≤ p u0 := by
    nlinarith
  exact hiter.trans hshift

/--
Zero-seed consequence of `budget_le` + nonpositive target budget.
-/
theorem uniformIterateBound_from_zero_of_nonexpansive_of_HGamma_kappa_of_budget_le_of_nonpos
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma B : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hbudget : hGammaKappaBudget kappa cost gamma hGamma ≤ B)
    (hBnonpos : B ≤ 0)
    (hzero : p u0 = 0)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ 0 := by
  have hiter :
      p ((Psi^[k]) u0) ≤ p u0 :=
    uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_budget_le_of_nonpos
      p Psi hPsi hfix hbound hbudget hBnonpos k
  simpa [hzero] using hiter

/--
Zero-seed + `budget_le` + nonpositive target budget in equality form.
-/
theorem uniformIterateEq_zero_of_nonexpansive_of_HGamma_kappa_of_budget_le_of_nonpos
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma B : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hbudget : hGammaKappaBudget kappa cost gamma hGamma ≤ B)
    (hBnonpos : B ≤ 0)
    (hzero : p u0 = 0)
    (k : ℕ) :
    p ((Psi^[k]) u0) = 0 := by
  have hle : p ((Psi^[k]) u0) ≤ 0 :=
    uniformIterateBound_from_zero_of_nonexpansive_of_HGamma_kappa_of_budget_le_of_nonpos
      p Psi hPsi hfix hbound hbudget hBnonpos hzero k
  exact le_antisymm hle (apply_nonneg p ((Psi^[k]) u0))

/--
Successor-index form of `budget_le` + nonpositive-target iterate control.
-/
theorem uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_budget_le_of_nonpos_succ
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma B : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hbudget : hGammaKappaBudget kappa cost gamma hGamma ≤ B)
    (hBnonpos : B ≤ 0)
    (k : ℕ) :
    p ((Psi^[k + 1]) u0) ≤ p u0 := by
  simpa [Nat.succ_eq_add_one] using
    uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_budget_le_of_nonpos
      p Psi hPsi hfix hbound hbudget hBnonpos (k + 1)

/--
Successor-index zero-seed consequence of `budget_le` + nonpositive target budget.
-/
theorem uniformIterateBound_from_zero_of_nonexpansive_of_HGamma_kappa_of_budget_le_of_nonpos_succ
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma B : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hbudget : hGammaKappaBudget kappa cost gamma hGamma ≤ B)
    (hBnonpos : B ≤ 0)
    (hzero : p u0 = 0)
    (k : ℕ) :
    p ((Psi^[k + 1]) u0) ≤ 0 := by
  simpa [Nat.succ_eq_add_one] using
    uniformIterateBound_from_zero_of_nonexpansive_of_HGamma_kappa_of_budget_le_of_nonpos
      p Psi hPsi hfix hbound hbudget hBnonpos hzero (k + 1)

/--
Successor-index equality form of `budget_le` + nonpositive target budget at zero seed.
-/
theorem uniformIterateEq_zero_of_nonexpansive_of_HGamma_kappa_of_budget_le_of_nonpos_succ
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma B : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hbudget : hGammaKappaBudget kappa cost gamma hGamma ≤ B)
    (hBnonpos : B ≤ 0)
    (hzero : p u0 = 0)
    (k : ℕ) :
    p ((Psi^[k + 1]) u0) = 0 := by
  simpa [Nat.succ_eq_add_one] using
    uniformIterateEq_zero_of_nonexpansive_of_HGamma_kappa_of_budget_le_of_nonpos
      p Psi hPsi hfix hbound hbudget hBnonpos hzero (k + 1)

/--
Inner-term upper bound with nonpositive target implies iterate control by the initial seminorm.

This composes the practical bridge `cost + γ H_γ ≤ C` with the sign condition `C ≤ 0`.
-/
theorem uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_inner_le_of_nonpos
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma C : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hkappa : 0 ≤ kappa)
    (hinner : cost + gamma * hGamma ≤ C)
    (hCnonpos : C ≤ 0)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ p u0 := by
  have hiter :
      p ((Psi^[k]) u0) ≤ p u0 + 2 * (kappa * C) :=
    uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_inner_le
      p Psi hPsi hfix hbound hkappa hinner k
  have hknonpos : kappa * C ≤ 0 := by
    nlinarith
  have hshift : p u0 + 2 * (kappa * C) ≤ p u0 := by
    nlinarith
  exact hiter.trans hshift

/--
Successor-index form of the inner-bound nonpositive-target iterate control.
-/
theorem uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_inner_le_of_nonpos_succ
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma C : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hkappa : 0 ≤ kappa)
    (hinner : cost + gamma * hGamma ≤ C)
    (hCnonpos : C ≤ 0)
    (k : ℕ) :
    p ((Psi^[k + 1]) u0) ≤ p u0 := by
  simpa [Nat.succ_eq_add_one] using
    uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_inner_le_of_nonpos
      p Psi hPsi hfix hbound hkappa hinner hCnonpos (k + 1)

/--
Successor-index form of the practical inner-term bridge `cost + γ H_γ ≤ C`.
-/
theorem uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_inner_le_succ
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma C : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hkappa : 0 ≤ kappa)
    (hinner : cost + gamma * hGamma ≤ C)
    (k : ℕ) :
    p ((Psi^[k + 1]) u0) ≤ p u0 + 2 * (kappa * C) := by
  simpa [Nat.succ_eq_add_one] using
    uniformIterateBound_of_nonexpansive_of_HGamma_kappa_of_inner_le
      p Psi hPsi hfix hbound hkappa hinner (k + 1)

/--
Zero-seed successor-index form of the practical inner-term bridge `cost + γ H_γ ≤ C`.
-/
theorem uniformIterateBound_from_zero_of_nonexpansive_of_HGamma_kappa_of_inner_le_succ
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma C : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hkappa : 0 ≤ kappa)
    (hinner : cost + gamma * hGamma ≤ C)
    (hzero : p u0 = 0)
    (k : ℕ) :
    p ((Psi^[k + 1]) u0) ≤ 2 * (kappa * C) := by
  simpa [Nat.succ_eq_add_one] using
    uniformIterateBound_from_zero_of_nonexpansive_of_HGamma_kappa_of_inner_le
      p Psi hPsi hfix hbound hkappa hinner hzero (k + 1)

/--
If the inner target `C` is nonpositive, the zero-seed inner-bridge gives a nonpositive bound.
-/
theorem uniformIterateBound_from_zero_of_nonexpansive_of_HGamma_kappa_of_inner_le_of_nonpos
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma C : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hkappa : 0 ≤ kappa)
    (hinner : cost + gamma * hGamma ≤ C)
    (hCnonpos : C ≤ 0)
    (hzero : p u0 = 0)
    (k : ℕ) :
    p ((Psi^[k]) u0) ≤ 0 := by
  have hiter :
      p ((Psi^[k]) u0) ≤ 2 * (kappa * C) :=
    uniformIterateBound_from_zero_of_nonexpansive_of_HGamma_kappa_of_inner_le
      p Psi hPsi hfix hbound hkappa hinner hzero k
  have htarget : 2 * (kappa * C) ≤ 0 := by
    nlinarith
  exact hiter.trans htarget

/--
Successor-index zero-seed form of the `inner_le` + nonpositive-target bridge.
-/
theorem uniformIterateBound_from_zero_of_nonexpansive_of_HGamma_kappa_of_inner_le_of_nonpos_succ
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma C : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hkappa : 0 ≤ kappa)
    (hinner : cost + gamma * hGamma ≤ C)
    (hCnonpos : C ≤ 0)
    (hzero : p u0 = 0)
    (k : ℕ) :
    p ((Psi^[k + 1]) u0) ≤ 0 := by
  simpa [Nat.succ_eq_add_one] using
    uniformIterateBound_from_zero_of_nonexpansive_of_HGamma_kappa_of_inner_le_of_nonpos
      p Psi hPsi hfix hbound hkappa hinner hCnonpos hzero (k + 1)

/--
Zero-seed equality form of the `inner_le` + nonpositive-target bridge.
-/
theorem uniformIterateEq_zero_of_nonexpansive_of_HGamma_kappa_of_inner_le_of_nonpos
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma C : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hkappa : 0 ≤ kappa)
    (hinner : cost + gamma * hGamma ≤ C)
    (hCnonpos : C ≤ 0)
    (hzero : p u0 = 0)
    (k : ℕ) :
    p ((Psi^[k]) u0) = 0 := by
  have hle : p ((Psi^[k]) u0) ≤ 0 :=
    uniformIterateBound_from_zero_of_nonexpansive_of_HGamma_kappa_of_inner_le_of_nonpos
      p Psi hPsi hfix hbound hkappa hinner hCnonpos hzero k
  exact le_antisymm hle (apply_nonneg p ((Psi^[k]) u0))

/--
Successor-index equality form of the `inner_le` + nonpositive-target bridge.
-/
theorem uniformIterateEq_zero_of_nonexpansive_of_HGamma_kappa_of_inner_le_of_nonpos_succ
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma C : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hkappa : 0 ≤ kappa)
    (hinner : cost + gamma * hGamma ≤ C)
    (hCnonpos : C ≤ 0)
    (hzero : p u0 = 0)
    (k : ℕ) :
    p ((Psi^[k + 1]) u0) = 0 := by
  simpa [Nat.succ_eq_add_one] using
    uniformIterateEq_zero_of_nonexpansive_of_HGamma_kappa_of_inner_le_of_nonpos
      p Psi hPsi hfix hbound hkappa hinner hCnonpos hzero (k + 1)

/--
Index-relaxed equality form of the `inner_le` + nonpositive-target bridge.

This keeps a larger bookkeeping index `kMax` available while certifying exact zero
seminorm at any subordinate iterate index `k ≤ kMax`.
-/
theorem uniformIterateEq_zero_of_nonexpansive_of_HGamma_kappa_of_inner_le_of_nonpos_of_le_index
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    {kappa cost gamma hGamma C : ℝ}
    (hbound : p uStar ≤ hGammaKappaBudget kappa cost gamma hGamma)
    (hkappa : 0 ≤ kappa)
    (hinner : cost + gamma * hGamma ≤ C)
    (hCnonpos : C ≤ 0)
    (hzero : p u0 = 0)
    (kMax k : ℕ)
    (_hk : k ≤ kMax) :
    p ((Psi^[k]) u0) = 0 := by
  exact uniformIterateEq_zero_of_nonexpansive_of_HGamma_kappa_of_inner_le_of_nonpos
    p Psi hPsi hfix hbound hkappa hinner hCnonpos hzero k

end PrimalDualBounds
end KLProjection
end FlowSinkhorn
