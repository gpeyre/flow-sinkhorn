import FlowSinkhorn.KLProjection.Setup
import Mathlib.Order.Monotone.Basic

/-!
# Primal bound from dual orbit radius

This module provides the monotone transfer lemma: if a primal quantity `X` is
controlled at a dual radius `d` (via a monotone function `Xbound`), and `d ≤ U`,
then `X ≤ Xbound U`.

## Note on reuse

This is a pure order-theoretic fact, independent of the specific problem. It can be
used by any application where primal feasibility is certified via a dual radius bound.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace PrimalDualBounds

/--
Abstract transport of a primal confinement estimate along a dual-radius bound.

This captures the generic blueprint shape:
if one has a primal control function `Xbound` known at a dual quantity `d`, then any upper bound
`d ≤ U` propagates the primal estimate to `U` by monotonicity.
-/
theorem primalBound_fromDualBound
    {d U X : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ U) :
    X ≤ Xbound U :=
  hprimal_at_d.trans (hmono hdual)

/--
Paper-facing alias with the upper dual quantity named as a budget.

This is the same transfer principle as `primalBound_fromDualBound`, but the statement shape is
often closer to the way Section 4 applications are written.
-/
theorem primalBound_fromDualBudget
    {d budget X : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget) :
    X ≤ Xbound budget := by
  calc
    X ≤ Xbound d := hprimal_at_d
    _ ≤ Xbound budget := hmono hdual

/--
Paper-facing one-line corollary when the dual radius is already bounded by `U`.
-/
theorem primalMassBound_fromDualRadius
    {U X : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    (hprimal_at_U : X ≤ Xbound U) :
    X ≤ Xbound U :=
  primalBound_fromDualBound hmono hprimal_at_U (le_rfl : U ≤ U)

/--
Primal confinement from an orbit upper bound.

This is the single-iterate specialisation of the primal transfer principle:
if `X ≤ Xbound orbit_k` holds for the current iterate seminorm value `orbit_k`,
and `orbit_k ≤ U_max` holds by some uniform orbit bound, then `X ≤ Xbound U_max`.

This is the form in which `prop:mass-bound-block` is applied: one substitutes the uniform
orbit estimate `p u_k ≤ U_max` and monotonicity of `Xbound` promotes the per-iterate control
to the uniform orbit budget.
-/
theorem primalBound_from_dualOrbitBound
    {X U_max : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {orbit_k : ℝ}
    (horbit : orbit_k ≤ U_max)
    (hprimal : X ≤ Xbound orbit_k) :
    X ≤ Xbound U_max :=
  primalBound_fromDualBound hmono hprimal horbit

/--
Uniform primal confinement from a pointwise orbit bound.

If every value in a sequence `orbit` is bounded by `U_max`, and `X ≤ Xbound (orbit k)` holds
for some index `k`, then `X ≤ Xbound U_max` for that same index.

This packages the typical usage pattern in Section 4: one has a uniform bound
`∀ k, orbit k ≤ U_max` (from the non-expansive iterate estimate) and a primal estimate that
holds at each iterate norm value; the conclusion is the primal bound evaluated at the budget.
-/
theorem primalBound_all_iterates
    {X U_max : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {orbit : ℕ → ℝ}
    (horbit : ∀ k, orbit k ≤ U_max)
    {k : ℕ}
    (hprimal_k : X ≤ Xbound (orbit k)) :
    X ≤ Xbound U_max :=
  primalBound_from_dualOrbitBound hmono (horbit k) hprimal_k

/--
Primal confinement through two chained dual bounds.

If `X ≤ Xbound d`, `d ≤ U1`, and `U1 ≤ U2`, then `X ≤ Xbound U2`.

This three-step chain is common in applications where a dual radius is first bounded by an
intermediate quantity and then the intermediate quantity is bounded by the final budget constant.
-/
theorem primalBound_trans
    {d U1 U2 X : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hd_U1 : d ≤ U1)
    (hU1_U2 : U1 ≤ U2) :
    X ≤ Xbound U2 :=
  primalBound_fromDualBound hmono hprimal_at_d (hd_U1.trans hU1_U2)

/--
Primal confinement from a pointwise dual `O(1/k)` bound.

If the iterate-level dual quantity `gap k` satisfies a rate bound
`gap k ≤ C / (k + 1)`, then any primal estimate stated at `gap k`
transfers to the explicit rate envelope `C / (k + 1)`.
-/
theorem primalBound_from_dualRateBound
    {X C : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {k : ℕ}
    (hprimal_k : X ≤ Xbound (gap k))
    (hrate_k : gap k ≤ C / (k + 1 : ℝ)) :
    X ≤ Xbound (C / (k + 1 : ℝ)) :=
  primalBound_fromDualBound hmono hprimal_k hrate_k

/--
Primal confinement from a pointwise dual rate plus a target threshold.

This is the direct optimization-ready bridge:
`gap k ≤ C/(k+1)` and `C/(k+1) ≤ eps` imply `X ≤ Xbound eps`
whenever `X ≤ Xbound (gap k)` holds.
-/
theorem primalBound_from_dualRate_threshold
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {k : ℕ}
    (hprimal_k : X ≤ Xbound (gap k))
    (hrate_k : gap k ≤ C / (k + 1 : ℝ))
    (hthreshold : C / (k + 1 : ℝ) ≤ eps) :
    X ≤ Xbound eps :=
  primalBound_trans hmono hprimal_k hrate_k hthreshold

/--
Primal confinement from a pointwise dual rate and ratio-form stopping rule.

If `C/eps ≤ k+1` (with `eps > 0`), this theorem derives the threshold
`C/(k+1) ≤ eps` and then applies `primalBound_from_dualRate_threshold`.
-/
theorem primalBound_from_dualRate_ratioBound
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {k : ℕ}
    (hprimal_k : X ≤ Xbound (gap k))
    (hrate_k : gap k ≤ C / (k + 1 : ℝ))
    (heps : 0 < eps)
    (hratio : C / eps ≤ (k + 1 : ℝ)) :
    X ≤ Xbound eps := by
  have hthreshold' : C ≤ (k + 1 : ℝ) * eps := (div_le_iff₀ heps).1 hratio
  have hthreshold : C / (k + 1 : ℝ) ≤ eps := by
    have hpos : 0 < (k + 1 : ℝ) := by
      exact_mod_cast Nat.succ_pos k
    have hthreshold'' : C ≤ eps * (k + 1 : ℝ) := by
      simpa [mul_comm] using hthreshold'
    exact (div_le_iff₀ hpos).2 hthreshold''
  exact primalBound_from_dualRate_threshold hmono hprimal_k hrate_k hthreshold

/--
Primal confinement from a pointwise dual rate and a natural-number ratio bound.

This is the `ℕ`-valued variant of `primalBound_from_dualRate_ratioBound`,
useful when iteration proofs provide `C/eps ≤ k+1` directly in naturals.
-/
theorem primalBound_from_dualRate_ratioBound_of_natBound
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {k : ℕ}
    (hprimal_k : X ≤ Xbound (gap k))
    (hrate_k : gap k ≤ C / (k + 1 : ℝ))
    (heps : 0 < eps)
    (hratio_nat : C / eps ≤ (k + 1 : ℕ)) :
    X ≤ Xbound eps := by
  have hratio : C / eps ≤ (k + 1 : ℝ) := by
    exact_mod_cast hratio_nat
  exact primalBound_from_dualRate_ratioBound
    (hmono := hmono) (hprimal_k := hprimal_k) (hrate_k := hrate_k) (heps := heps) hratio

/--
Primal confinement from a pointwise dual rate and ceiling-form stopping rule.

This matches paper-facing iteration complexity statements where one controls
`Nat.ceil (C/eps)` and needs a direct primal conclusion.
-/
theorem primalBound_from_dualRate_closedFormCeil
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {k : ℕ}
    (hprimal_k : X ≤ Xbound (gap k))
    (hrate_k : gap k ≤ C / (k + 1 : ℝ))
    (heps : 0 < eps)
    (hk : Nat.ceil (C / eps) ≤ k + 1) :
    X ≤ Xbound eps := by
  have hratio : C / eps ≤ (k + 1 : ℝ) := by
    simpa [Nat.cast_add, Nat.cast_one] using (Nat.le_of_ceil_le hk : C / eps ≤ (k + 1 : ℕ))
  exact primalBound_from_dualRate_ratioBound hmono hprimal_k hrate_k heps hratio

/--
Ceiling-form primal confinement through an intermediate natural-number budget.

This wrapper is convenient when complexity arguments first prove
`Nat.ceil (C/eps) ≤ N` and only afterwards relate `N ≤ k+1`.
-/
theorem primalBound_from_dualRate_closedFormCeil_of_natBound
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {k N : ℕ}
    (hprimal_k : X ≤ Xbound (gap k))
    (hrate_k : gap k ≤ C / (k + 1 : ℝ))
    (heps : 0 < eps)
    (hk_nat : Nat.ceil (C / eps) ≤ N)
    (hN : N ≤ k + 1) :
    X ≤ Xbound eps :=
  primalBound_from_dualRate_closedFormCeil
    hmono hprimal_k hrate_k heps (hk_nat.trans hN)

/--
Closed-form-ceil dual-rate transfer with an index comparison helper.

If the ceiling condition is known at index `m` and `m ≤ n`, this upgrades directly to `n`.
-/
theorem primalBound_from_dualRate_closedFormCeil_of_le_index
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n : ℕ}
    (hprimal_n : X ≤ Xbound (gap n))
    (hrate_n : gap n ≤ C / (n + 1 : ℝ))
    (heps : 0 < eps)
    (hk : Nat.ceil (C / eps) ≤ m + 1)
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hn : Nat.ceil (C / eps) ≤ n + 1 := by
    exact hk.trans (Nat.add_le_add_right hmn 1)
  exact primalBound_from_dualRate_closedFormCeil
    hmono hprimal_n hrate_n heps hn

/-! ### Successor-index dual-rate threshold bridges -/

/--
Successor-index dual-rate threshold transfer.

This is the `(k+1)` iterate convenience wrapper for
`primalBound_from_dualRate_threshold`.
-/
theorem primalBound_from_dualRate_threshold_succ
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {k : ℕ}
    (hprimal_k1 : X ≤ Xbound (gap (k + 1)))
    (hrate_k1 : gap (k + 1) ≤ C / ((k + 1 : ℝ) + 1))
    (hthreshold : C / ((k + 1 : ℝ) + 1) ≤ eps) :
    X ≤ Xbound eps := by
  exact primalBound_fromDualBound hmono hprimal_k1 (hrate_k1.trans hthreshold)

/--
Successor-index dual-rate ratio-form transfer.
-/
theorem primalBound_from_dualRate_ratioBound_succ
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {k : ℕ}
    (hprimal_k1 : X ≤ Xbound (gap (k + 1)))
    (hrate_k1 : gap (k + 1) ≤ C / ((k + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hratio : C / eps ≤ (k + 1 : ℝ) + 1) :
    X ≤ Xbound eps := by
  have hthreshold' : C ≤ ((k + 1 : ℝ) + 1) * eps := (div_le_iff₀ heps).1 hratio
  have hpos : 0 < ((k + 1 : ℝ) + 1) := by positivity
  have hthreshold : C / ((k + 1 : ℝ) + 1) ≤ eps := by
    have hthreshold'' : C ≤ eps * ((k + 1 : ℝ) + 1) := by
      simpa [mul_comm] using hthreshold'
    exact (div_le_iff₀ hpos).2 hthreshold''
  exact primalBound_from_dualRate_threshold_succ hmono hprimal_k1 hrate_k1 hthreshold

/--
Successor-index dual-rate ratio transfer from a natural-number bound.

This variant is convenient when complexity proofs provide the index bound in `ℕ`.
-/
theorem primalBound_from_dualRate_ratioBound_succ_of_natBound
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {k : ℕ}
    (hprimal_k1 : X ≤ Xbound (gap (k + 1)))
    (hrate_k1 : gap (k + 1) ≤ C / ((k + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hratio_nat : C / eps ≤ ((k + 1) + 1 : ℕ)) :
    X ≤ Xbound eps := by
  have hratio' : C / eps ≤ (((k + 1) + 1 : ℕ) : ℝ) := by
    exact_mod_cast hratio_nat
  have hratio : C / eps ≤ (k + 1 : ℝ) + 1 := by
    simpa [Nat.cast_add, Nat.cast_one] using hratio'
  exact primalBound_from_dualRate_ratioBound_succ
    (hmono := hmono) (hprimal_k1 := hprimal_k1) (hrate_k1 := hrate_k1) (heps := heps) hratio

/--
Successor-index dual-rate ceiling-form transfer.
-/
theorem primalBound_from_dualRate_closedFormCeil_succ
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {k : ℕ}
    (hprimal_k1 : X ≤ Xbound (gap (k + 1)))
    (hrate_k1 : gap (k + 1) ≤ C / ((k + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hk : Nat.ceil (C / eps) ≤ (k + 1) + 1) :
    X ≤ Xbound eps := by
  have hratio_nat : C / eps ≤ ((k + 1) + 1 : ℕ) :=
    Nat.le_of_ceil_le hk
  exact primalBound_from_dualRate_ratioBound_succ_of_natBound
    (hmono := hmono) (hprimal_k1 := hprimal_k1) (hrate_k1 := hrate_k1)
    (heps := heps) hratio_nat

/--
Successor-index closed-form-ceil dual-rate transfer via an intermediate natural bound.
-/
theorem primalBound_from_dualRate_closedFormCeil_succ_of_natBound
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {k N : ℕ}
    (hprimal_k1 : X ≤ Xbound (gap (k + 1)))
    (hrate_k1 : gap (k + 1) ≤ C / ((k + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hk_nat : Nat.ceil (C / eps) ≤ N)
    (hN : N ≤ (k + 1) + 1) :
    X ≤ Xbound eps :=
  primalBound_from_dualRate_closedFormCeil_succ
    hmono hprimal_k1 hrate_k1 heps (hk_nat.trans hN)

/-! ## Bridge from explicit `α*B/(n+1)` rate assumptions -/

/--
Primal confinement from an explicit dual rate envelope `α*B/(n+1)`.
-/
theorem primalBound_from_explicitRateBound
    {X alpha B : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n : ℕ}
    (hprimal_n : X ≤ Xbound (gap n))
    (hrate_n : gap n ≤ (alpha * B) / (n + 1 : ℝ)) :
    X ≤ Xbound ((alpha * B) / (n + 1 : ℝ)) :=
  primalBound_from_dualRateBound hmono hprimal_n hrate_n

/--
Explicit dual rate envelope plus budget threshold gives primal budget confinement.
-/
theorem primalBound_from_explicitRate_budget
    {X alpha B budget : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n : ℕ}
    (hprimal_n : X ≤ Xbound (gap n))
    (hrate_n : gap n ≤ (alpha * B) / (n + 1 : ℝ))
    (hbudget : (alpha * B) / (n + 1 : ℝ) ≤ budget) :
    X ≤ Xbound budget :=
  primalBound_from_dualRate_threshold hmono hprimal_n hrate_n hbudget

/--
Explicit dual rate envelope plus multiplicative threshold gives primal budget confinement.

This variant accepts the threshold in product form
`alpha * B ≤ budget * (n+1)`, often produced by algebraic manipulations in
complexity proofs before a division step.
-/
theorem primalBound_from_explicitRate_budget_of_mulThreshold
    {X alpha B budget : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n : ℕ}
    (hprimal_n : X ≤ Xbound (gap n))
    (hrate_n : gap n ≤ (alpha * B) / (n + 1 : ℝ))
    (hbudget_mul : alpha * B ≤ budget * (n + 1 : ℝ)) :
    X ≤ Xbound budget := by
  have hpos : 0 < (n + 1 : ℝ) := by
    exact_mod_cast Nat.succ_pos n
  have hbudget : (alpha * B) / (n + 1 : ℝ) ≤ budget := by
    exact (div_le_iff₀ hpos).2 hbudget_mul
  exact primalBound_from_explicitRate_budget hmono hprimal_n hrate_n hbudget

/--
Explicit dual rate envelope plus ratio-form stopping rule gives primal `eps`-confinement.
-/
theorem primalBound_from_explicitRate_ratioBound
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n : ℕ}
    (hprimal_n : X ≤ Xbound (gap n))
    (hrate_n : gap n ≤ (alpha * B) / (n + 1 : ℝ))
    (heps : 0 < eps)
    (hratio : (alpha * B) / eps ≤ (n + 1 : ℝ)) :
    X ≤ Xbound eps :=
  primalBound_from_dualRate_ratioBound hmono hprimal_n hrate_n heps hratio

/--
Explicit-rate ratio-form transfer from a natural-number bound.
-/
theorem primalBound_from_explicitRate_ratioBound_of_natBound
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n : ℕ}
    (hprimal_n : X ≤ Xbound (gap n))
    (hrate_n : gap n ≤ (alpha * B) / (n + 1 : ℝ))
    (heps : 0 < eps)
    (hratio_nat : (alpha * B) / eps ≤ (n + 1 : ℕ)) :
    X ≤ Xbound eps := by
  have hratio : (alpha * B) / eps ≤ (n + 1 : ℝ) := by
    exact_mod_cast hratio_nat
  exact primalBound_from_explicitRate_ratioBound
    (hmono := hmono) (hprimal_n := hprimal_n) (hrate_n := hrate_n) (heps := heps) hratio

/--
Successor-index explicit-rate budget transfer.

This is the `(n+1)` iterate convenience wrapper for
`primalBound_from_explicitRate_budget`.
-/
theorem primalBound_from_explicitRate_budget_succ
    {X alpha B budget : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ (alpha * B) / ((n + 1 : ℝ) + 1))
    (hbudget : (alpha * B) / ((n + 1 : ℝ) + 1) ≤ budget) :
    X ≤ Xbound budget := by
  exact primalBound_fromDualBound hmono hprimal_n1 (hrate_n1.trans hbudget)

/--
Successor-index explicit-rate ratio-form transfer.
-/
theorem primalBound_from_explicitRate_ratioBound_succ
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ (alpha * B) / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hratio : (alpha * B) / eps ≤ (n + 1 : ℝ) + 1) :
    X ≤ Xbound eps := by
  have hthreshold' : alpha * B ≤ eps * ((n + 1 : ℝ) + 1) := by
    nlinarith [(div_le_iff₀ heps).1 hratio]
  have hpos : 0 < ((n + 1 : ℝ) + 1) := by positivity
  have hthreshold : (alpha * B) / ((n + 1 : ℝ) + 1) ≤ eps := by
    have hthreshold'' : alpha * B ≤ eps * ((n + 1 : ℝ) + 1) := hthreshold'
    exact (div_le_iff₀ hpos).2 hthreshold''
  exact primalBound_fromDualBound hmono hprimal_n1 (hrate_n1.trans hthreshold)

/--
Successor-index explicit-rate ratio-form transfer from a natural-number bound.
-/
theorem primalBound_from_explicitRate_ratioBound_succ_of_natBound
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ (alpha * B) / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hratio_nat : (alpha * B) / eps ≤ ((n + 1) + 1 : ℕ)) :
    X ≤ Xbound eps := by
  have hratio' : (alpha * B) / eps ≤ (((n + 1) + 1 : ℕ) : ℝ) := by
    exact_mod_cast hratio_nat
  have hratio : (alpha * B) / eps ≤ (n + 1 : ℝ) + 1 := by
    simpa [Nat.cast_add, Nat.cast_one] using hratio'
  exact primalBound_from_explicitRate_ratioBound_succ
    (hmono := hmono) (hprimal_n1 := hprimal_n1) (hrate_n1 := hrate_n1) (heps := heps) hratio

/--
Explicit dual rate envelope plus ceiling-form stopping rule gives primal `eps`-confinement.
-/
theorem primalBound_from_explicitRate_closedFormCeil
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n : ℕ}
    (hprimal_n : X ≤ Xbound (gap n))
    (hrate_n : gap n ≤ (alpha * B) / (n + 1 : ℝ))
    (heps : 0 < eps)
    (hn : Nat.ceil ((alpha * B) / eps) ≤ n + 1) :
    X ≤ Xbound eps :=
  primalBound_from_dualRate_closedFormCeil hmono hprimal_n hrate_n heps hn

/--
Explicit-rate closed-form-ceil transfer via an intermediate natural bound.
-/
theorem primalBound_from_explicitRate_closedFormCeil_of_natBound
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n N : ℕ}
    (hprimal_n : X ≤ Xbound (gap n))
    (hrate_n : gap n ≤ (alpha * B) / (n + 1 : ℝ))
    (heps : 0 < eps)
    (hn_nat : Nat.ceil ((alpha * B) / eps) ≤ N)
    (hN : N ≤ n + 1) :
    X ≤ Xbound eps :=
  primalBound_from_explicitRate_closedFormCeil
    hmono hprimal_n hrate_n heps (hn_nat.trans hN)

/--
Explicit-rate closed-form-ceil transfer with an index comparison helper.
-/
theorem primalBound_from_explicitRate_closedFormCeil_of_le_index
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n : ℕ}
    (hprimal_n : X ≤ Xbound (gap n))
    (hrate_n : gap n ≤ (alpha * B) / (n + 1 : ℝ))
    (heps : 0 < eps)
    (hn : Nat.ceil ((alpha * B) / eps) ≤ m + 1)
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hn' : Nat.ceil ((alpha * B) / eps) ≤ n + 1 := by
    exact hn.trans (Nat.add_le_add_right hmn 1)
  exact primalBound_from_explicitRate_closedFormCeil
    hmono hprimal_n hrate_n heps hn'

/--
Successor-index explicit-rate ceiling-form transfer.

This is the `(n+1)` iterate convenience form of
`primalBound_from_explicitRate_closedFormCeil`.
-/
theorem primalBound_from_explicitRate_closedFormCeil_succ
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ (alpha * B) / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hn : Nat.ceil ((alpha * B) / eps) ≤ (n + 1) + 1) :
    X ≤ Xbound eps := by
  have hratio : (alpha * B) / eps ≤ (n + 1 : ℝ) + 1 := by
    have hratio_nat : (alpha * B) / eps ≤ ((n + 1) + 1 : ℕ) :=
      Nat.le_of_ceil_le hn
    exact_mod_cast hratio_nat
  exact primalBound_from_explicitRate_ratioBound_succ
    (hmono := hmono) (hprimal_n1 := hprimal_n1) (hrate_n1 := hrate_n1)
    (heps := heps) hratio

/--
Successor-index dual-rate closed-form-ceil transfer with an index comparison helper.

If the ceiling bound is known at successor index `m+1` and `m ≤ n`, this upgrades to `n+1`.
-/
theorem primalBound_from_dualRate_closedFormCeil_succ_of_le_index
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ C / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hk : Nat.ceil (C / eps) ≤ (m + 1) + 1)
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hmn_nat : (m + 1) + 1 ≤ (n + 1) + 1 := by
    exact Nat.add_le_add_right (Nat.add_le_add_right hmn 1) 1
  have hn : Nat.ceil (C / eps) ≤ (n + 1) + 1 := hk.trans hmn_nat
  exact primalBound_from_dualRate_closedFormCeil_succ
    hmono hprimal_n1 hrate_n1 heps hn

/--
Successor-index explicit-rate ratio transfer with an index comparison helper.

If the ratio bound holds at index `m+1` and `m ≤ n`, this upgrades directly to `n+1`.
-/
theorem primalBound_from_explicitRate_ratioBound_succ_of_le_index
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ (alpha * B) / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hratio_m : (alpha * B) / eps ≤ (m + 1 : ℝ) + 1)
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hmn_nat : (m + 1) + 1 ≤ (n + 1) + 1 := by
    exact Nat.add_le_add_right (Nat.add_le_add_right hmn 1) 1
  have hmn_real : (m + 1 : ℝ) + 1 ≤ (n + 1 : ℝ) + 1 := by
    exact_mod_cast hmn_nat
  have hratio_n : (alpha * B) / eps ≤ (n + 1 : ℝ) + 1 := hratio_m.trans hmn_real
  exact primalBound_from_explicitRate_ratioBound_succ
    hmono hprimal_n1 hrate_n1 heps hratio_n

/--
Successor-index explicit-rate closed-form-ceil transfer with an index comparison helper.
-/
theorem primalBound_from_explicitRate_closedFormCeil_succ_of_le_index
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ (alpha * B) / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hn : Nat.ceil ((alpha * B) / eps) ≤ (m + 1) + 1)
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hmn_nat : (m + 1) + 1 ≤ (n + 1) + 1 := by
    exact Nat.add_le_add_right (Nat.add_le_add_right hmn 1) 1
  have hn' : Nat.ceil ((alpha * B) / eps) ≤ (n + 1) + 1 := hn.trans hmn_nat
  exact primalBound_from_explicitRate_closedFormCeil_succ
    hmono hprimal_n1 hrate_n1 heps hn'

/--
Successor-index explicit-rate closed-form-ceil transfer via an intermediate natural bound.
-/
theorem primalBound_from_explicitRate_closedFormCeil_succ_of_natBound
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n N : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ (alpha * B) / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hn_nat : Nat.ceil ((alpha * B) / eps) ≤ N)
    (hN : N ≤ (n + 1) + 1) :
    X ≤ Xbound eps :=
  primalBound_from_explicitRate_closedFormCeil_succ
    hmono hprimal_n1 hrate_n1 heps (hn_nat.trans hN)

/--
Successor-index dual-rate ratio-form transfer with an index comparison helper.
-/
theorem primalBound_from_dualRate_ratioBound_succ_of_le_index
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ C / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hratio_m : C / eps ≤ (m + 1 : ℝ) + 1)
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hmn_nat : (m + 1) + 1 ≤ (n + 1) + 1 := by
    exact Nat.add_le_add_right (Nat.add_le_add_right hmn 1) 1
  have hmn_real : (m + 1 : ℝ) + 1 ≤ (n + 1 : ℝ) + 1 := by
    exact_mod_cast hmn_nat
  have hratio_n : C / eps ≤ (n + 1 : ℝ) + 1 := hratio_m.trans hmn_real
  exact primalBound_from_dualRate_ratioBound_succ
    hmono hprimal_n1 hrate_n1 heps hratio_n

/--
Successor-index dual-rate ratio-form transfer from a natural-number bound
with an index comparison helper.
-/
theorem primalBound_from_dualRate_ratioBound_succ_of_natBound_of_le_index
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ C / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hratio_nat_m : C / eps ≤ ((m + 1) + 1 : ℕ))
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hratio_m' : C / eps ≤ (((m + 1) + 1 : ℕ) : ℝ) := by
    exact_mod_cast hratio_nat_m
  have hratio_m : C / eps ≤ (m + 1 : ℝ) + 1 := by
    simpa [Nat.cast_add, Nat.cast_one] using hratio_m'
  exact primalBound_from_dualRate_ratioBound_succ_of_le_index
    hmono hprimal_n1 hrate_n1 heps hratio_m hmn

/--
Successor-index dual-rate closed-form-ceil transfer from a natural-number bound
with an index comparison helper.
-/
theorem primalBound_from_dualRate_closedFormCeil_succ_of_natBound_of_le_index
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n N : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ C / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hk_nat : Nat.ceil (C / eps) ≤ N)
    (hN : N ≤ (m + 1) + 1)
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hk : Nat.ceil (C / eps) ≤ (m + 1) + 1 := hk_nat.trans hN
  exact primalBound_from_dualRate_closedFormCeil_succ_of_le_index
    hmono hprimal_n1 hrate_n1 heps hk hmn

/--
Successor-index explicit-rate closed-form-ceil transfer from a natural-number bound
with an index comparison helper.
-/
theorem primalBound_from_explicitRate_closedFormCeil_succ_of_natBound_of_le_index
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n N : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ (alpha * B) / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hn_nat : Nat.ceil ((alpha * B) / eps) ≤ N)
    (hN : N ≤ (m + 1) + 1)
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hn : Nat.ceil ((alpha * B) / eps) ≤ (m + 1) + 1 := hn_nat.trans hN
  exact primalBound_from_explicitRate_closedFormCeil_succ_of_le_index
    hmono hprimal_n1 hrate_n1 heps hn hmn

/--
Successor-index explicit-rate ratio-form transfer from a natural-number bound
with an index comparison helper.
-/
theorem primalBound_from_explicitRate_ratioBound_succ_of_natBound_of_le_index
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ (alpha * B) / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hratio_nat_m : (alpha * B) / eps ≤ ((m + 1) + 1 : ℕ))
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hratio_m' : (alpha * B) / eps ≤ (((m + 1) + 1 : ℕ) : ℝ) := by
    exact_mod_cast hratio_nat_m
  have hratio_m : (alpha * B) / eps ≤ (m + 1 : ℝ) + 1 := by
    simpa [Nat.cast_add, Nat.cast_one] using hratio_m'
  exact primalBound_from_explicitRate_ratioBound_succ_of_le_index
    hmono hprimal_n1 hrate_n1 heps hratio_m hmn

/--
Dual-rate closed-form-ceil transfer from a natural bound and an index comparison helper.
-/
theorem primalBound_from_dualRate_closedFormCeil_of_natBound_of_le_index
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n N : ℕ}
    (hprimal_n : X ≤ Xbound (gap n))
    (hrate_n : gap n ≤ C / (n + 1 : ℝ))
    (heps : 0 < eps)
    (hk_nat : Nat.ceil (C / eps) ≤ N)
    (hN : N ≤ m + 1)
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hk : Nat.ceil (C / eps) ≤ m + 1 := hk_nat.trans hN
  exact primalBound_from_dualRate_closedFormCeil_of_le_index
    hmono hprimal_n hrate_n heps hk hmn

/--
Explicit-rate closed-form-ceil transfer from a natural bound and an index comparison helper.
-/
theorem primalBound_from_explicitRate_closedFormCeil_of_natBound_of_le_index
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n N : ℕ}
    (hprimal_n : X ≤ Xbound (gap n))
    (hrate_n : gap n ≤ (alpha * B) / (n + 1 : ℝ))
    (heps : 0 < eps)
    (hn_nat : Nat.ceil ((alpha * B) / eps) ≤ N)
    (hN : N ≤ m + 1)
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hn : Nat.ceil ((alpha * B) / eps) ≤ m + 1 := hn_nat.trans hN
  exact primalBound_from_explicitRate_closedFormCeil_of_le_index
    hmono hprimal_n hrate_n heps hn hmn

end PrimalDualBounds
end KLProjection
end FlowSinkhorn
