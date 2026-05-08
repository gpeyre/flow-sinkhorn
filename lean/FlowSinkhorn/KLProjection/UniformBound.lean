import Mathlib.Analysis.Seminorm
import Mathlib.Tactic

noncomputable section

/-!
# Seminorm-nonexpansive maps and uniform orbit bounds

This module defines the `SeminormNonexpansive` predicate for maps on normed modules
and provides the key dynamical corollary: nonexpansive maps are uniformly bounded
near any fixed point.

## Note on reuse

This module is independent of the specific seminorm and algorithm. It can be used
with any `Seminorm 𝕜 E` — the variation seminorm is just one instance. For the
connection to topical maps (monotone + translation-equivariant), see `Topical.lean`.
-/

namespace FlowSinkhorn
namespace KLProjection

open Function

variable {𝕜 E : Type*}
variable [NormedField 𝕜] [AddCommGroup E] [Module 𝕜 E]

/-! ## The SeminormNonexpansive predicate -/

/-- Non-expansiveness of a map with respect to a seminorm-induced difference distance. -/
def SeminormNonexpansive (p : Seminorm 𝕜 E) (Ψ : E → E) : Prop :=
  ∀ x y, p (Ψ x - Ψ y) ≤ p (x - y)

/--
Abstract fixed-point contraction estimate behind the first step of Proposition 4.2.
-/
theorem seminorm_iterate_sub_fixed_le_of_nonexpansive
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar) :
    ∀ k : ℕ, p ((Ψ^[k]) u0 - uStar) ≤ p (u0 - uStar) := by
  intro k
  induction k with
  | zero =>
      simp
  | succ k ih =>
      calc
        p ((Ψ^[k + 1]) u0 - uStar) = p (Ψ ((Ψ^[k]) u0) - Ψ uStar) := by
          simp [Function.iterate_succ_apply', hfix]
        _ ≤ p ((Ψ^[k]) u0 - uStar) := hΨ ((Ψ^[k]) u0) uStar
        _ ≤ p (u0 - uStar) := ih

/--
Abstract uniform iterate bound: a non-expansive map is uniformly controlled by any fixed point.
This is the seminorm-level core of Proposition 4.2 before the optimizer-specific bound on `u_γ`.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar) (k : ℕ) :
    p ((Ψ^[k]) u0) ≤ p u0 + 2 * p uStar := by
  have hiter :=
    seminorm_iterate_sub_fixed_le_of_nonexpansive
      p Ψ hΨ (uStar := uStar) (u0 := u0) hfix k
  calc
    p ((Ψ^[k]) u0) = p (((Ψ^[k]) u0 - uStar) + uStar) := by rw [sub_add_cancel]
    _ ≤ p ((Ψ^[k]) u0 - uStar) + p uStar := map_add_le_add p _ _
    _ ≤ p (u0 - uStar) + p uStar := by
          simpa [add_comm, add_left_comm, add_assoc] using add_le_add_left hiter (p uStar)
    _ ≤ (p u0 + p uStar) + p uStar := by
          gcongr
          exact map_sub_le_add p u0 uStar
    _ = p u0 + 2 * p uStar := by ring

/--
Paper-facing corollary: once the fixed point has seminorm at most `B`, every iterate is bounded by
`p u₀ + 2 B`.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {B : ℝ} (hB : p uStar ≤ B) (k : ℕ) :
    p ((Ψ^[k]) u0) ≤ p u0 + 2 * B := by
  calc
    p ((Ψ^[k]) u0) ≤ p u0 + 2 * p uStar :=
      seminorm_iterate_le_of_nonexpansive_fixedPoint p Ψ hΨ hfix k
    _ ≤ p u0 + 2 * B := by nlinarith

/--
Budget-lifting convenience: if `p u⋆ ≤ B` and `B ≤ U`, the iterate bound upgrades from `B` to `U`.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_bound_le
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {B U : ℝ} (hB : p uStar ≤ B) (hBU : B ≤ U) (k : ℕ) :
    p ((Ψ^[k]) u0) ≤ p u0 + 2 * U := by
  have hiter :
      p ((Ψ^[k]) u0) ≤ p u0 + 2 * B :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
      p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hB k
  have hshift : p u0 + 2 * B ≤ p u0 + 2 * U := by
    nlinarith
  exact hiter.trans hshift

/--
Successor-index budget-lifting convenience form.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_succ_of_bound_le
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {B U : ℝ} (hB : p uStar ≤ B) (hBU : B ≤ U) (k : ℕ) :
    p ((Ψ^[k + 1]) u0) ≤ p u0 + 2 * U := by
  simpa [Nat.succ_eq_add_one] using
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_bound_le
      p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hB hBU (k + 1)

/--
Zero-base budget-lifting convenience form.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_zero_base_of_bound_le
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {B U : ℝ} (hzero : p u0 = 0) (hB : p uStar ≤ B) (hBU : B ≤ U) (k : ℕ) :
    p ((Ψ^[k]) u0) ≤ 2 * U := by
  have hiter :
      p ((Ψ^[k]) u0) ≤ p u0 + 2 * U :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_bound_le
      p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hB hBU k
  rw [hzero, zero_add] at hiter
  exact hiter

/--
Successor-index zero-base budget-lifting convenience form.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_zero_base_succ_of_bound_le
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {B U : ℝ} (hzero : p u0 = 0) (hB : p uStar ≤ B) (hBU : B ≤ U) (k : ℕ) :
    p ((Ψ^[k + 1]) u0) ≤ 2 * U := by
  simpa [Nat.succ_eq_add_one] using
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_zero_base_of_bound_le
      p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hzero hB hBU (k + 1)

/-! ## Closure properties -/

/--
The identity map is always `SeminormNonexpansive`.
-/
theorem SeminormNonexpansive_id (p : Seminorm 𝕜 E) : SeminormNonexpansive p id :=
  fun _ _ => le_rfl

/--
The composition of two `SeminormNonexpansive` maps is `SeminormNonexpansive`.

If `T₁` and `T₂` are both nonexpansive for the seminorm `p`, then so is their composition
`T₂ ∘ T₁`.
-/
theorem SeminormNonexpansive_comp
    (p : Seminorm 𝕜 E) (T₁ T₂ : E → E)
    (hT₁ : SeminormNonexpansive p T₁)
    (hT₂ : SeminormNonexpansive p T₂) :
    SeminormNonexpansive p (T₂ ∘ T₁) :=
  fun x y => calc
    p ((T₂ ∘ T₁) x - (T₂ ∘ T₁) y) = p (T₂ (T₁ x) - T₂ (T₁ y)) := rfl
    _ ≤ p (T₁ x - T₁ y) := hT₂ (T₁ x) (T₁ y)
    _ ≤ p (x - y) := hT₁ x y

/--
Any iterate of a `SeminormNonexpansive` map is itself `SeminormNonexpansive`.
-/
theorem SeminormNonexpansive_iterate
    (p : Seminorm 𝕜 E) (T : E → E)
    (hT : SeminormNonexpansive p T) :
    ∀ k : ℕ, SeminormNonexpansive p (T^[k]) := by
  intro k
  induction k with
  | zero => simpa using SeminormNonexpansive_id p
  | succ k ih =>
      rw [Function.iterate_succ']
      exact SeminormNonexpansive_comp p (T^[k]) T ih hT

/--
Pointwise nonexpansive estimate for iterates.

This is the inequality form of `SeminormNonexpansive_iterate`, convenient when downstream
proofs need an immediate bound on a pair `(x, y)`.
-/
theorem seminorm_iterate_nonexpansive
    (p : Seminorm 𝕜 E) (T : E → E)
    (hT : SeminormNonexpansive p T)
    (k : ℕ) (x y : E) :
    p ((T^[k]) x - (T^[k]) y) ≤ p (x - y) :=
  (SeminormNonexpansive_iterate p T hT k) x y

/--
Successor-step pointwise nonexpansive iterate estimate.
-/
theorem seminorm_iterate_nonexpansive_succ
    (p : Seminorm 𝕜 E) (T : E → E)
    (hT : SeminormNonexpansive p T)
    (k : ℕ) (x y : E) :
    p ((T^[k + 1]) x - (T^[k + 1]) y) ≤ p (x - y) := by
  simpa [Nat.succ_eq_add_one] using
    seminorm_iterate_nonexpansive p T hT (k + 1) x y

/-! ## Sufficient conditions for nonexpansiveness -/

/--
A contraction with factor `c ≤ 1` is nonexpansive.
-/
theorem seminorm_nonexpansive_of_contract
    (p : Seminorm 𝕜 E) (T : E → E) {c : ℝ}
    (hc : c ≤ 1)
    (hT : ∀ x y : E, p (T x - T y) ≤ c * p (x - y)) :
    SeminormNonexpansive p T := by
  intro x y
  calc p (T x - T y) ≤ c * p (x - y) := hT x y
    _ ≤ 1 * p (x - y) := by nlinarith [apply_nonneg p (x - y)]
    _ = p (x - y) := one_mul _

/-! ## Monotonicity and convergence of iterates -/

/--
The sequence `p (Ψ^[k] u0 - uStar)` is nonincreasing in `k` when `Ψ` is nonexpansive and
`uStar` is a fixed point.
-/
theorem seminorm_iterate_convergence_to_fixed
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    (k1 k2 : ℕ) (h : k1 ≤ k2) :
    p ((Psi^[k2]) u0 - uStar) ≤ p ((Psi^[k1]) u0 - uStar) := by
  have hmono : ∀ k : ℕ, p ((Psi^[k + 1]) u0 - uStar) ≤ p ((Psi^[k]) u0 - uStar) := by
    intro k
    rw [Function.iterate_succ_apply']
    calc p (Psi ((Psi^[k]) u0) - uStar)
        = p (Psi ((Psi^[k]) u0) - Psi uStar) := by rw [hfix]
      _ ≤ p ((Psi^[k]) u0 - uStar) := hPsi _ _
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le h
  induction d with
  | zero => simp
  | succ d ih =>
      have ih' : p ((Psi^[k1 + d]) u0 - uStar) ≤ p ((Psi^[k1]) u0 - uStar) :=
        ih (Nat.le_add_right k1 d)
      calc p ((Psi^[k1 + (d + 1)]) u0 - uStar)
          = p ((Psi^[k1 + d + 1]) u0 - uStar) := by ring_nf
        _ ≤ p ((Psi^[k1 + d]) u0 - uStar) := hmono _
        _ ≤ p ((Psi^[k1]) u0 - uStar) := ih'

/--
The constant map is always nonexpansive: `p(c - c) = 0 ≤ p(x - y)`.
-/
theorem SeminormNonexpansive_const
    (p : Seminorm 𝕜 E) (c : E) :
    SeminormNonexpansive p (fun _ => c) := by
  intro x y
  simp [map_zero]

/--
Triangle-inequality bound on iterates: `p (Ψ^[k] u0 - uStar) ≤ p u0 + p uStar`.
-/
theorem seminorm_iterate_le_triangle
    (p : Seminorm 𝕜 E) (Psi : E → E)
    (hPsi : SeminormNonexpansive p Psi)
    {uStar u0 : E} (hfix : Psi uStar = uStar)
    (k : ℕ) :
    p ((Psi^[k]) u0 - uStar) ≤ p u0 + p uStar := by
  have h1 := seminorm_iterate_sub_fixed_le_of_nonexpansive p Psi hPsi
    (uStar := uStar) (u0 := u0) hfix k
  have h2 : p (u0 - uStar) ≤ p u0 + p uStar := by
    calc p (u0 - uStar)
        = p (u0 + (-uStar)) := by simp [sub_eq_add_neg]
      _ ≤ p u0 + p (-uStar) := map_add_le_add p u0 (-uStar)
      _ = p u0 + p uStar := by rw [map_neg_eq_map]
  linarith

/--
If the initial distance to a fixed point is bounded by `R`, every iterate stays within `R`.
-/
theorem seminorm_iterate_sub_fixed_le_of_nonexpansive_radius
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {R : ℝ} (hR : p (u0 - uStar) ≤ R) (k : ℕ) :
    p ((Ψ^[k]) u0 - uStar) ≤ R := by
  exact
    (seminorm_iterate_sub_fixed_le_of_nonexpansive p Ψ hΨ
      (uStar := uStar) (u0 := u0) hfix k).trans hR

/--
Bounded-base formulation: combine bounds on `p u₀` and `p u⋆` into a uniform iterate bound.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_baseBound
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {U0 B : ℝ} (hU0 : p u0 ≤ U0) (hB : p uStar ≤ B) (k : ℕ) :
    p ((Ψ^[k]) u0) ≤ U0 + 2 * B := by
  calc
    p ((Ψ^[k]) u0) ≤ p u0 + 2 * B :=
      seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
        p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hB k
    _ ≤ U0 + 2 * B := by gcongr

/--
Successor variant of the bounded-base uniform iterate bound.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_baseBound_succ
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {U0 B : ℝ} (hU0 : p u0 ≤ U0) (hB : p uStar ≤ B) (k : ℕ) :
    p ((Ψ^[k + 1]) u0) ≤ U0 + 2 * B := by
  simpa [Nat.succ_eq_add_one] using
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_baseBound
      p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hU0 hB (k + 1)

/--
Zero-base variant: if `p u₀ = 0`, the uniform iterate bound simplifies to `2 * B`.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_zero_base
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {B : ℝ} (hzero : p u0 = 0) (hB : p uStar ≤ B) (k : ℕ) :
    p ((Ψ^[k]) u0) ≤ 2 * B := by
  have h :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
      p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hB k
  rw [hzero, zero_add] at h
  exact h

/--
Successor variant of the fixed-point uniform iterate bound with explicit base term.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_succ
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {B : ℝ} (hB : p uStar ≤ B) (k : ℕ) :
    p ((Ψ^[k + 1]) u0) ≤ p u0 + 2 * B := by
  simpa [Nat.succ_eq_add_one] using
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
      p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hB (k + 1)

/--
Zero-base successor variant of the uniform iterate bound.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_zero_base_succ
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {B : ℝ} (hzero : p u0 = 0) (hB : p uStar ≤ B) (k : ℕ) :
    p ((Ψ^[k + 1]) u0) ≤ 2 * B := by
  simpa [Nat.succ_eq_add_one] using
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_zero_base
      p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hzero hB (k + 1)

/--
Nonpositive-base variant: if `p u₀ ≤ 0`, the uniform iterate bound simplifies to `2 * B`.

This is a relaxed form of the zero-base statement, useful when the base seminorm is known
to be nonpositive (hence necessarily zero by seminorm nonnegativity).
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_nonpos_base
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {B : ℝ} (hU0 : p u0 ≤ 0) (hB : p uStar ≤ B) (k : ℕ) :
    p ((Ψ^[k]) u0) ≤ 2 * B := by
  have h :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_baseBound
      p Ψ hΨ (uStar := uStar) (u0 := u0) hfix
      (U0 := 0) hU0 hB k
  simpa [zero_add] using h

/--
Successor-index nonpositive-base variant of the uniform iterate bound.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_nonpos_base_succ
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {B : ℝ} (hU0 : p u0 ≤ 0) (hB : p uStar ≤ B) (k : ℕ) :
    p ((Ψ^[k + 1]) u0) ≤ 2 * B := by
  simpa [Nat.succ_eq_add_one] using
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_nonpos_base
      p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hU0 hB (k + 1)

/--
Iterate-composition bounded-base corollary.

Applying the same nonexpansive map in `m`-step blocks is still uniformly bounded:
if `p u₀ ≤ U₀` and `p u⋆ ≤ B`, then
`p (((Ψ^[m])^[k]) u₀) ≤ U₀ + 2 * B`.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_iterateComposition_baseBound
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {U0 B : ℝ} (hU0 : p u0 ≤ U0) (hB : p uStar ≤ B)
    (m k : ℕ) :
    p (((Ψ^[m])^[k]) u0) ≤ U0 + 2 * B := by
  have hfixm : (Ψ^[m]) uStar = uStar := by
    induction m with
    | zero =>
        simp
    | succ m ih =>
        calc
          (Ψ^[m + 1]) uStar = Ψ ((Ψ^[m]) uStar) := by
            simp [Function.iterate_succ_apply']
          _ = Ψ uStar := by rw [ih]
          _ = uStar := hfix
  have hΨm : SeminormNonexpansive p (Ψ^[m]) :=
    SeminormNonexpansive_iterate p Ψ hΨ m
  exact seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_baseBound
    p (Ψ^[m]) hΨm (uStar := uStar) (u0 := u0) hfixm hU0 hB k

/--
Successor-index iterate-composition bounded-base corollary.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_iterateComposition_baseBound_succ
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {U0 B : ℝ} (hU0 : p u0 ≤ U0) (hB : p uStar ≤ B)
    (m k : ℕ) :
    p (((Ψ^[m])^[k + 1]) u0) ≤ U0 + 2 * B := by
  simpa [Nat.succ_eq_add_one] using
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_iterateComposition_baseBound
      p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hU0 hB m (k + 1)

/--
Iterate-composition bounded-base corollary with budget lifting.

If `p u⋆ ≤ B` and `B ≤ U`, the `m`-block iterate bound upgrades from `B` to `U`.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_iterateComposition_budget_le
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {U0 B U : ℝ} (hU0 : p u0 ≤ U0) (hB : p uStar ≤ B) (hBU : B ≤ U)
    (m k : ℕ) :
    p (((Ψ^[m])^[k]) u0) ≤ U0 + 2 * U := by
  have hiter :
      p (((Ψ^[m])^[k]) u0) ≤ U0 + 2 * B :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_iterateComposition_baseBound
      p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hU0 hB m k
  have hshift : U0 + 2 * B ≤ U0 + 2 * U := by nlinarith
  exact hiter.trans hshift

/--
Successor-index iterate-composition bounded-base corollary with budget lifting.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_iterateComposition_budget_le_succ
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {U0 B U : ℝ} (hU0 : p u0 ≤ U0) (hB : p uStar ≤ B) (hBU : B ≤ U)
    (m k : ℕ) :
    p (((Ψ^[m])^[k + 1]) u0) ≤ U0 + 2 * U := by
  simpa [Nat.succ_eq_add_one] using
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_iterateComposition_budget_le
      p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hU0 hB hBU m (k + 1)

/--
Zero-step iterate-composition bounded-base corollary with budget lifting.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_iterateComposition_budget_le_zeroStep
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {U0 B U : ℝ} (hU0 : p u0 ≤ U0) (hB : p uStar ≤ B) (hBU : B ≤ U)
    (m : ℕ) :
    p (((Ψ^[m])^[0]) u0) ≤ U0 + 2 * U :=
  seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_iterateComposition_budget_le
    p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hU0 hB hBU m 0

/--
Nonpositive-base iterate-composition corollary with budget lifting.

This is a relaxed form of the zero-base budget-lifted statement.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_iterComp_nonpos_base_budget_le
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {B U : ℝ} (hU0 : p u0 ≤ 0) (hB : p uStar ≤ B) (hBU : B ≤ U)
    (m k : ℕ) :
    p (((Ψ^[m])^[k]) u0) ≤ 2 * U := by
  have hiter :
      p (((Ψ^[m])^[k]) u0) ≤ 0 + 2 * U :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_iterateComposition_budget_le
      p Ψ hΨ (uStar := uStar) (u0 := u0) hfix (U0 := 0) hU0 hB hBU m k
  simpa [zero_add] using hiter

/--
Successor-index nonpositive-base iterate-composition corollary with budget lifting.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_iterComp_nonpos_base_budget_le_succ
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {B U : ℝ} (hU0 : p u0 ≤ 0) (hB : p uStar ≤ B) (hBU : B ≤ U)
    (m k : ℕ) :
    p (((Ψ^[m])^[k + 1]) u0) ≤ 2 * U := by
  simpa [Nat.succ_eq_add_one] using
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_iterComp_nonpos_base_budget_le
      p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hU0 hB hBU m (k + 1)

/--
`of_le_index` wrapper for the nonpositive-base iterate-composition budget-lifted bound.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_iterComp_nonpos_base_budget_le_of_le_index
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {B U : ℝ} (hU0 : p u0 ≤ 0) (hB : p uStar ≤ B) (hBU : B ≤ U)
    (m k n : ℕ) (_hk : k ≤ n) :
    p (((Ψ^[m])^[n]) u0) ≤ 2 * U :=
  seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_iterComp_nonpos_base_budget_le
    p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hU0 hB hBU m n

/--
`of_natBound` wrapper for the nonpositive-base iterate-composition budget-lifted bound.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_iterComp_nonpos_base_budget_le_of_natBound
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {B U : ℝ} (hU0 : p u0 ≤ 0) (hB : p uStar ≤ B) (hBU : B ≤ U)
    (m n N : ℕ) (hn : n ≤ N) :
    p (((Ψ^[m])^[N]) u0) ≤ 2 * U :=
  seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_iterComp_nonpos_base_budget_le_of_le_index
    p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hU0 hB hBU m n N hn

/--
Successor-index `of_le_index` wrapper for the nonpositive-base
iterate-composition budget-lifted bound.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_iterComp_nonpos_base_budget_le_succ_of_le_index
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {B U : ℝ} (hU0 : p u0 ≤ 0) (hB : p uStar ≤ B) (hBU : B ≤ U)
    (m k n : ℕ) (hk : k ≤ n) :
    p (((Ψ^[m])^[n + 1]) u0) ≤ 2 * U := by
  have hk' : k + 1 ≤ n + 1 := Nat.succ_le_succ hk
  simpa [Nat.succ_eq_add_one] using
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_iterComp_nonpos_base_budget_le_of_le_index
      p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hU0 hB hBU m (k + 1) (n + 1) hk'

/--
Zero-base iterate-composition corollary with budget lifting.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_iterComp_zero_base_budget_le
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {B U : ℝ} (hzero : p u0 = 0) (hB : p uStar ≤ B) (hBU : B ≤ U)
    (m k : ℕ) :
    p (((Ψ^[m])^[k]) u0) ≤ 2 * U := by
  have hiter :
      p (((Ψ^[m])^[k]) u0) ≤ 0 + 2 * U :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_iterateComposition_budget_le
      p Ψ hΨ (uStar := uStar) (u0 := u0) hfix
      (U0 := 0) (by simp [hzero]) hB hBU m k
  simpa [zero_add] using hiter

/--
Successor-index zero-base iterate-composition corollary with budget lifting.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_iterComp_zero_base_budget_le_succ
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {B U : ℝ} (hzero : p u0 = 0) (hB : p uStar ≤ B) (hBU : B ≤ U)
    (m k : ℕ) :
    p (((Ψ^[m])^[k + 1]) u0) ≤ 2 * U := by
  simpa [Nat.succ_eq_add_one] using
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_iterComp_zero_base_budget_le
      p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hzero hB hBU m (k + 1)

/--
`of_le_index` wrapper for the zero-base iterate-composition budget-lifted bound.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_iterComp_zero_base_budget_le_of_le_index
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {B U : ℝ} (hzero : p u0 = 0) (hB : p uStar ≤ B) (hBU : B ≤ U)
    (m k n : ℕ) (_hk : k ≤ n) :
    p (((Ψ^[m])^[n]) u0) ≤ 2 * U :=
  seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_iterComp_zero_base_budget_le
    p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hzero hB hBU m n

/--
Natural-bound wrapper for the zero-base iterate-composition budget-lifted bound.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_iterComp_zero_base_budget_le_of_natBound
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {B U : ℝ} (hzero : p u0 = 0) (hB : p uStar ≤ B) (hBU : B ≤ U)
    (m n N : ℕ) (hNn : N ≤ n) :
    p (((Ψ^[m])^[n]) u0) ≤ 2 * U :=
  seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_iterComp_zero_base_budget_le_of_le_index
    p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hzero hB hBU m N n hNn

/--
Successor-index `of_le_index` wrapper for the zero-base iterate-composition budget-lifted bound.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_iterComp_zero_base_budget_le_succ_of_le_index
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {B U : ℝ} (hzero : p u0 = 0) (hB : p uStar ≤ B) (hBU : B ≤ U)
    (m k n : ℕ) (_hk : k ≤ n) :
    p (((Ψ^[m])^[n + 1]) u0) ≤ 2 * U := by
  simpa [Nat.succ_eq_add_one] using
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_iterComp_zero_base_budget_le_succ
      p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hzero hB hBU m n

/--
Zero-step zero-base iterate-composition corollary with budget lifting.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_iterComp_zero_base_budget_le_zeroStep
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {B U : ℝ} (hzero : p u0 = 0) (hB : p uStar ≤ B) (hBU : B ≤ U)
    (m : ℕ) :
    p (((Ψ^[m])^[0]) u0) ≤ 2 * U :=
  seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_iterComp_zero_base_budget_le
    p Ψ hΨ (uStar := uStar) (u0 := u0) hfix hzero hB hBU m 0

/--
Zero-base iterate-composition corollary.

If `p u₀ = 0`, then every `m`-block iterate also satisfies
`p (((Ψ^[m])^[k]) u₀) ≤ 2 * B`.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_iterateComposition_zero_base
    (p : Seminorm 𝕜 E) (Ψ : E → E)
    (hΨ : SeminormNonexpansive p Ψ)
    {uStar u0 : E} (hfix : Ψ uStar = uStar)
    {B : ℝ} (hzero : p u0 = 0) (hB : p uStar ≤ B)
    (m k : ℕ) :
    p (((Ψ^[m])^[k]) u0) ≤ 2 * B := by
  have h :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_iterateComposition_baseBound
      p Ψ hΨ (uStar := uStar) (u0 := u0) hfix
      (U0 := 0) (by simp [hzero]) hB m k
  simpa [zero_add] using h

/--
Composed-map bounded-base corollary.

If both `T₁` and `T₂` are nonexpansive and `u⋆` is fixed by `T₂ ∘ T₁`, then
iterates of the composed map satisfy the same uniform base bound.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_comp_baseBound
    (p : Seminorm 𝕜 E) (T₁ T₂ : E → E)
    (hT₁ : SeminormNonexpansive p T₁)
    (hT₂ : SeminormNonexpansive p T₂)
    {uStar u0 : E} (hfix : (T₂ ∘ T₁) uStar = uStar)
    {U0 B : ℝ} (hU0 : p u0 ≤ U0) (hB : p uStar ≤ B) (k : ℕ) :
    p (((T₂ ∘ T₁)^[k]) u0) ≤ U0 + 2 * B := by
  exact
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_baseBound
      p (T₂ ∘ T₁) (SeminormNonexpansive_comp p T₁ T₂ hT₁ hT₂)
      (uStar := uStar) (u0 := u0) hfix hU0 hB k

/--
Successor variant of the composed-map bounded-base corollary.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_comp_baseBound_succ
    (p : Seminorm 𝕜 E) (T₁ T₂ : E → E)
    (hT₁ : SeminormNonexpansive p T₁)
    (hT₂ : SeminormNonexpansive p T₂)
    {uStar u0 : E} (hfix : (T₂ ∘ T₁) uStar = uStar)
    {U0 B : ℝ} (hU0 : p u0 ≤ U0) (hB : p uStar ≤ B) (k : ℕ) :
    p (((T₂ ∘ T₁)^[k + 1]) u0) ≤ U0 + 2 * B := by
  simpa [Nat.succ_eq_add_one] using
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_comp_baseBound
      p T₁ T₂ hT₁ hT₂ (uStar := uStar) (u0 := u0) hfix hU0 hB (k + 1)

/--
Zero-base composed-map corollary.

If `p u₀ = 0`, then every iterate of `T₂ ∘ T₁` is bounded by `2 * B`.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_comp_zero_base
    (p : Seminorm 𝕜 E) (T₁ T₂ : E → E)
    (hT₁ : SeminormNonexpansive p T₁)
    (hT₂ : SeminormNonexpansive p T₂)
    {uStar u0 : E} (hfix : (T₂ ∘ T₁) uStar = uStar)
    {B : ℝ} (hzero : p u0 = 0) (hB : p uStar ≤ B) (k : ℕ) :
    p (((T₂ ∘ T₁)^[k]) u0) ≤ 2 * B := by
  have h :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_comp_baseBound
      p T₁ T₂ hT₁ hT₂ (uStar := uStar) (u0 := u0) hfix
      (U0 := 0) (by simp [hzero]) hB k
  simpa [zero_add] using h

/--
Successor variant of the zero-base composed-map corollary.
-/
theorem seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_comp_zero_base_succ
    (p : Seminorm 𝕜 E) (T₁ T₂ : E → E)
    (hT₁ : SeminormNonexpansive p T₁)
    (hT₂ : SeminormNonexpansive p T₂)
    {uStar u0 : E} (hfix : (T₂ ∘ T₁) uStar = uStar)
    {B : ℝ} (hzero : p u0 = 0) (hB : p uStar ≤ B) (k : ℕ) :
    p (((T₂ ∘ T₁)^[k + 1]) u0) ≤ 2 * B := by
  simpa [Nat.succ_eq_add_one] using
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound_of_comp_zero_base
      p T₁ T₂ hT₁ hT₂ (uStar := uStar) (u0 := u0) hfix hzero hB (k + 1)

end KLProjection
end FlowSinkhorn
