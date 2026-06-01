import FlowSinkhorn.KLProjection.VariationVocabulary
import Mathlib.Data.Real.Basic
import Mathlib.Order.ConditionallyCompleteLattice.Finset
import Mathlib.Order.Monotone.Basic
import Mathlib.Tactic

noncomputable section

/-!
# Variation seminorm on finite real-valued functions

This module defines the **variation seminorm** (half the oscillation) on functions
`ι → ℝ` for a finite nonempty index type `ι`, and establishes its key properties.

## Note on reuse

This file is independent of KL divergence, optimal transport, and graph structure.
It can be imported by any module that needs the variation seminorm on finite functions.

For the bundled `IsTopical` predicate and its consequences (composition, iteration,
nonexpansiveness), see `Topical.lean`.
-/

namespace FlowSinkhorn
namespace KLProjection

variable {ι : Type*} [Fintype ι] [Nonempty ι]

lemma coordMin_le (x : ι → ℝ) (i : ι) : coordMin x ≤ x i := by
  unfold coordMin
  exact Finset.inf'_le (s := Finset.univ) (f := x) (by simp)

lemma le_coordMax (x : ι → ℝ) (i : ι) : x i ≤ coordMax x := by
  unfold coordMax
  exact Finset.le_sup' (s := Finset.univ) (f := x) (by simp)

lemma oscillation_nonneg (x : ι → ℝ) : 0 ≤ oscillation x := by
  unfold oscillation
  let i : ι := Classical.choice inferInstance
  exact sub_nonneg.mpr <| le_trans (coordMin_le x i) (le_coordMax x i)

lemma variationSeminorm_nonneg (x : ι → ℝ) : 0 ≤ variationSeminorm x := by
  unfold variationSeminorm
  have h : 0 ≤ oscillation x := oscillation_nonneg x
  linarith

/-- Adding a constant to every coordinate shifts the maximum by the same constant. -/
lemma coordMax_add_const (x : ι → ℝ) (c : ℝ) :
    coordMax (fun i => x i + c) = coordMax x + c := by
  unfold coordMax
  simpa [Function.comp, max_add_add_right] using
    (Finset.comp_sup'_eq_sup'_comp (s := Finset.univ) (H := Finset.univ_nonempty)
      (f := x) (g := fun t : ℝ => t + c) fun a b => by
        simp [max_add_add_right]).symm

/-- Adding a constant to every coordinate shifts the minimum by the same constant. -/
lemma coordMin_add_const (x : ι → ℝ) (c : ℝ) :
    coordMin (fun i => x i + c) = coordMin x + c := by
  unfold coordMin
  simpa [Function.comp, min_add_add_right] using
    (Finset.comp_inf'_eq_inf'_comp (s := Finset.univ) (H := Finset.univ_nonempty)
      (f := x) (g := fun t : ℝ => t + c) fun a b => by
        simp [min_add_add_right]).symm

/-- Oscillation is invariant under translation by constant vectors. -/
theorem oscillation_add_const (x : ι → ℝ) (c : ℝ) :
    oscillation (fun i => x i + c) = oscillation x := by
  unfold oscillation
  rw [coordMax_add_const, coordMin_add_const]
  ring

/-- Variation seminorm is invariant under translation by constant vectors. -/
theorem variationSeminorm_add_const (x : ι → ℝ) (c : ℝ) :
    variationSeminorm (fun i => x i + c) = variationSeminorm x := by
  unfold variationSeminorm
  rw [oscillation_add_const]

/--
The centering shift places every coordinate in the interval
`[-variationSeminorm x, variationSeminorm x]`.
-/
theorem abs_add_centeringShift_le_variationSeminorm (x : ι → ℝ) (i : ι) :
    |x i + centeringShift x| ≤ variationSeminorm x := by
  have hmin : coordMin x ≤ x i := coordMin_le x i
  have hmax : x i ≤ coordMax x := le_coordMax x i
  unfold centeringShift variationSeminorm oscillation
  rw [abs_le]
  constructor <;> linarith

/--
There is always a translate of `x` whose coordinates are bounded in absolute value by
`variationSeminorm x`.
-/
theorem exists_shift_abs_le_variationSeminorm (x : ι → ℝ) :
    ∃ c, ∀ i, |x i + c| ≤ variationSeminorm x := by
  refine ⟨centeringShift x, ?_⟩
  intro i
  exact abs_add_centeringShift_le_variationSeminorm x i

/-- Paper-facing short alias for the centered representative theorem. -/
theorem variationSeminorm_centered_representative (x : ι → ℝ) :
    ∃ c, ∀ i, |x i + c| ≤ variationSeminorm x :=
  exists_shift_abs_le_variationSeminorm x

/--
Any uniform absolute bound on a translated copy of `x` bounds the variation seminorm of `x`.

This is a practical one-sided quotient characterization:
`variationSeminorm x` is the optimal radius of a translated representative.
-/
theorem variationSeminorm_le_of_forall_abs_add_const_le
    (x : ι → ℝ) {c M : ℝ}
    (hM : ∀ i, |x i + c| ≤ M) :
    variationSeminorm x ≤ M := by
  let y : ι → ℝ := fun i => x i + c
  have hM_nonneg : 0 ≤ M := by
    let i : ι := Classical.choice inferInstance
    exact (abs_nonneg (y i)).trans (by simpa [y] using hM i)
  have hmax : coordMax y ≤ M := by
    unfold coordMax y
    refine Finset.sup'_le (s := Finset.univ) (H := Finset.univ_nonempty)
      (f := fun i : ι => x i + c) ?_
    intro i hi
    exact (abs_le.mp (hM i)).2
  have hmin : -M ≤ coordMin y := by
    unfold coordMin y
    refine Finset.le_inf' (s := Finset.univ) (H := Finset.univ_nonempty)
      (f := fun i : ι => x i + c) ?_
    intro i hi
    exact (abs_le.mp (hM i)).1
  have hy : variationSeminorm y ≤ M := by
    unfold variationSeminorm oscillation
    linarith
  simpa [y, variationSeminorm_add_const] using hy

/-- Paper-facing short alias for the quotient upper-bound lemma. -/
theorem variationSeminorm_le_of_shifted_supnorm_bound
    (x : ι → ℝ) {c M : ℝ}
    (hM : ∀ i, |x i + c| ≤ M) :
    variationSeminorm x ≤ M :=
  variationSeminorm_le_of_forall_abs_add_const_le x hM

/-- The variation seminorm satisfies the triangle inequality. -/
theorem variationSeminorm_add (x y : ι → ℝ) :
    variationSeminorm (x + y) ≤ variationSeminorm x + variationSeminorm y := by
  unfold variationSeminorm oscillation
  suffices h : coordMax (x + y) - coordMin (x + y) ≤
      (coordMax x - coordMin x) + (coordMax y - coordMin y) by linarith
  have hmax : coordMax (x + y) ≤ coordMax x + coordMax y := by
    unfold coordMax
    refine Finset.sup'_le (s := Finset.univ) Finset.univ_nonempty (f := x + y) ?_
    intro i _
    simp only [Pi.add_apply]
    exact add_le_add (le_coordMax x i) (le_coordMax y i)
  have hmin : coordMin x + coordMin y ≤ coordMin (x + y) := by
    unfold coordMin
    refine Finset.le_inf' (s := Finset.univ) Finset.univ_nonempty (f := x + y) ?_
    intro i _
    simp only [Pi.add_apply]
    exact add_le_add (coordMin_le x i) (coordMin_le y i)
  linarith

/-- The variation seminorm is absolutely homogeneous under scalar multiplication. -/
theorem variationSeminorm_smul (a : ℝ) (x : ι → ℝ) :
    variationSeminorm (a • x) = |a| * variationSeminorm x := by
  unfold variationSeminorm oscillation
  suffices h : coordMax (a • x) - coordMin (a • x) = |a| * (coordMax x - coordMin x) by
    linarith
  by_cases ha : 0 ≤ a
  · -- a ≥ 0 case: coordMax (a • x) = a * coordMax x and coordMin (a • x) = a * coordMin x
    have hmax : coordMax (a • x) = a * coordMax x := by
      apply le_antisymm
      · -- each (a • x) i ≤ a * coordMax x
        unfold coordMax
        refine Finset.sup'_le Finset.univ_nonempty (f := a • x) (fun i _ => ?_)
        simp only [Pi.smul_apply, smul_eq_mul]
        exact mul_le_mul_of_nonneg_left (le_coordMax x i) ha
      · -- a * coordMax x is attained: find j with x j = coordMax x
        obtain ⟨j, _, hj⟩ := Finset.exists_mem_eq_sup' Finset.univ_nonempty (f := x)
        have heq : a * coordMax x = (a • x) j := by
          simp only [Pi.smul_apply, smul_eq_mul, coordMax, hj]
        rw [heq]
        exact le_coordMax (a • x) j
    have hmin : coordMin (a • x) = a * coordMin x := by
      apply le_antisymm
      · -- a * coordMin x is attained: find j with x j = coordMin x
        obtain ⟨j, _, hj⟩ := Finset.exists_mem_eq_inf' Finset.univ_nonempty (f := x)
        have heq : a * coordMin x = (a • x) j := by
          simp only [Pi.smul_apply, smul_eq_mul, coordMin, hj]
        rw [heq]
        exact coordMin_le (a • x) j
      · -- each (a • x) i ≥ a * coordMin x
        unfold coordMin
        refine Finset.le_inf' Finset.univ_nonempty (f := a • x) (fun i _ => ?_)
        simp only [Pi.smul_apply, smul_eq_mul]
        exact mul_le_mul_of_nonneg_left (coordMin_le x i) ha
    rw [hmax, hmin, abs_of_nonneg ha]; ring
  · -- a < 0 case: coordMax (a • x) = a * coordMin x and coordMin (a • x) = a * coordMax x
    have ha_neg : a < 0 := not_le.mp ha
    have ha' : a ≤ 0 := le_of_lt ha_neg
    have hmax : coordMax (a • x) = a * coordMin x := by
      apply le_antisymm
      · -- each (a • x) i ≤ a * coordMin x (a < 0 flips order)
        unfold coordMax
        refine Finset.sup'_le Finset.univ_nonempty (f := a • x) (fun i _ => ?_)
        simp only [Pi.smul_apply, smul_eq_mul]
        exact mul_le_mul_of_nonpos_left (coordMin_le x i) ha'
      · -- a * coordMin x is attained at argmin of x
        obtain ⟨j, _, hj⟩ := Finset.exists_mem_eq_inf' Finset.univ_nonempty (f := x)
        have heq : a * coordMin x = (a • x) j := by
          simp only [Pi.smul_apply, smul_eq_mul, coordMin, hj]
        rw [heq]
        exact le_coordMax (a • x) j
    have hmin : coordMin (a • x) = a * coordMax x := by
      apply le_antisymm
      · -- a * coordMax x is attained at argmax of x
        obtain ⟨j, _, hj⟩ := Finset.exists_mem_eq_sup' Finset.univ_nonempty (f := x)
        have heq : a * coordMax x = (a • x) j := by
          simp only [Pi.smul_apply, smul_eq_mul, coordMax, hj]
        rw [heq]
        exact coordMin_le (a • x) j
      · -- each (a • x) i ≥ a * coordMax x (a < 0 flips order)
        unfold coordMin
        refine Finset.le_inf' Finset.univ_nonempty (f := a • x) (fun i _ => ?_)
        simp only [Pi.smul_apply, smul_eq_mul]
        exact mul_le_mul_of_nonpos_left (le_coordMax x i) ha'
    rw [hmax, hmin, abs_of_neg ha_neg]; ring

/--
The oscillation scales with the absolute value of the scalar.

`oscillation (a • x) = |a| * oscillation x`.
-/
theorem oscillation_smul (a : ℝ) (x : ι → ℝ) :
    oscillation (a • x) = |a| * oscillation x := by
  have h := variationSeminorm_smul a x
  unfold variationSeminorm at h
  linarith

/-- The variation seminorm is at most the oscillation. -/
theorem variationSeminorm_le_oscillation (x : ι → ℝ) :
    variationSeminorm x ≤ oscillation x := by
  unfold variationSeminorm
  have h := oscillation_nonneg x
  linarith

/--
The oscillation equals twice the variation seminorm.
-/
theorem oscillation_eq_two_mul_variationSeminorm (x : ι → ℝ) :
    oscillation x = 2 * variationSeminorm x := by
  unfold variationSeminorm
  ring

/-- The variation seminorm vanishes at zero. -/
theorem variationSeminorm_zero : variationSeminorm (0 : ι → ℝ) = 0 := by
  have hmax : coordMax (0 : ι → ℝ) = 0 := by
    apply le_antisymm
    · apply Finset.sup'_le Finset.univ_nonempty
      intros i _
      rfl
    · exact le_coordMax (0 : ι → ℝ) (Classical.choice inferInstance)
  have hmin : coordMin (0 : ι → ℝ) = 0 := by
    apply le_antisymm
    · exact coordMin_le (0 : ι → ℝ) (Classical.choice inferInstance)
    · apply Finset.le_inf' Finset.univ_nonempty
      intros i _
      rfl
  simp [variationSeminorm, oscillation, hmax, hmin]

/-- The variation seminorm is even: `variationSeminorm (-x) = variationSeminorm x`. -/
theorem variationSeminorm_neg (x : ι → ℝ) : variationSeminorm (-x) = variationSeminorm x := by
  have hmax : coordMax (-x) = -coordMin x := by
    apply le_antisymm
    · -- each (-x) i ≤ -coordMin x
      apply Finset.sup'_le Finset.univ_nonempty
      intro i _
      simp only [Pi.neg_apply]
      have h : coordMin x ≤ x i := coordMin_le x i
      linarith
    · -- -coordMin x is attained: find j with coordMin x = x j
      obtain ⟨j, _, hj⟩ := Finset.exists_mem_eq_inf' Finset.univ_nonempty (f := x)
      -- hj : Finset.univ.inf' ... x = x j, so coordMin x = x j
      have hcm : coordMin x = x j := by unfold coordMin; exact hj
      have hval : -coordMin x = (-x) j := by
        simp only [Pi.neg_apply]; linarith
      rw [hval]
      exact le_coordMax (-x) j
  have hmin : coordMin (-x) = -coordMax x := by
    apply le_antisymm
    · -- -coordMax x is attained: find j with coordMax x = x j
      obtain ⟨j, _, hj⟩ := Finset.exists_mem_eq_sup' Finset.univ_nonempty (f := x)
      -- hj : Finset.univ.sup' ... x = x j, so coordMax x = x j
      have hcM : coordMax x = x j := by unfold coordMax; exact hj
      have hval : -coordMax x = (-x) j := by
        simp only [Pi.neg_apply]; linarith
      rw [hval]
      exact coordMin_le (-x) j
    · -- each (-x) i ≥ -coordMax x
      apply Finset.le_inf' Finset.univ_nonempty
      intro i _
      simp only [Pi.neg_apply]
      have h : x i ≤ coordMax x := le_coordMax x i
      linarith
  unfold variationSeminorm oscillation
  linarith [show coordMax (-x) - coordMin (-x) = coordMax x - coordMin x by linarith]

/-- The variation seminorm is symmetric: `variationSeminorm (x - y) = variationSeminorm (y - x)`. -/
theorem variationSeminorm_sub_comm (x y : ι → ℝ) :
    variationSeminorm (x - y) = variationSeminorm (y - x) := by
  have h : y - x = -(x - y) := by simp [neg_sub]
  rw [h, variationSeminorm_neg]

/-- If `variationSeminorm x ≤ M`, then `M ≥ 0`. -/
theorem variationSeminorm_nonneg_of_le {x : ι → ℝ} {M : ℝ} (h : variationSeminorm x ≤ M) :
    0 ≤ M :=
  (variationSeminorm_nonneg x).trans h

/-- A constant vector has zero variation seminorm. -/
theorem variationSeminorm_const (c : ℝ) :
    variationSeminorm (fun _ : ι => c) = 0 := by
  have h : (fun _ : ι => c) = fun i => (0 : ι → ℝ) i + c := by simp
  simp only [h, variationSeminorm_add_const, variationSeminorm_zero]

/--
`variationSeminorm x = 0` if and only if `x` is constant (i.e. all coordinates are equal).

Forward direction: if the oscillation is 0 then `coordMax x = coordMin x`, so all values
lie in a degenerate interval.
Backward direction: a constant function has zero oscillation.
-/
theorem variationSeminorm_eq_zero_iff (x : ι → ℝ) :
    variationSeminorm x = 0 ↔ ∃ c : ℝ, ∀ i, x i = c := by
  constructor
  · intro h
    -- variationSeminorm x = 0 means coordMax x = coordMin x
    have hosc : oscillation x = 0 := by unfold variationSeminorm at h; linarith
    have heq : coordMax x = coordMin x := by
      unfold oscillation at hosc; linarith
    -- Use coordMin x as the constant
    refine ⟨coordMin x, fun i => le_antisymm ?_ (coordMin_le x i)⟩
    calc x i ≤ coordMax x := le_coordMax x i
      _ = coordMin x := heq
  · rintro ⟨c, hc⟩
    have h : x = fun _ => c := funext hc
    rw [h]
    exact variationSeminorm_const c

/--
If `variationSeminorm u₀ = 0`, then `u₀` is a constant function.
This is the useful direction for orbit bound theorems where we start at `u₀ = 0`.
-/
theorem exists_const_of_variationSeminorm_zero (x : ι → ℝ) (h : variationSeminorm x = 0) :
    ∃ c : ℝ, ∀ i, x i = c :=
  (variationSeminorm_eq_zero_iff x).mp h

/-! ## Mathlib Seminorm instance -/

/-- `variationSeminorm` as a Mathlib `Seminorm ℝ (ι → ℝ)`. -/
noncomputable def variationSeminormAsSeminorm : Seminorm ℝ (ι → ℝ) :=
  Seminorm.of variationSeminorm variationSeminorm_add (fun a x => by
    rw [variationSeminorm_smul, Real.norm_eq_abs])

/--
If all pairwise differences `|x i - x j|` are bounded by `2*M`, then
`variationSeminorm x ≤ M`.

This is the direct "bounded differences imply bounded variation" lemma.
-/
theorem variationSeminorm_le_of_pairwise_diff_bound
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {x : ι → ℝ} {M : ℝ}
    (h : ∀ i j : ι, |x i - x j| ≤ 2 * M) :
    variationSeminorm x ≤ M := by
  unfold variationSeminorm oscillation
  obtain ⟨imax, _, himax⟩ := Finset.exists_mem_eq_sup' Finset.univ_nonempty (f := x)
  obtain ⟨imin, _, himin⟩ := Finset.exists_mem_eq_inf' Finset.univ_nonempty (f := x)
  have hcm : coordMax x = x imax := by unfold coordMax; exact himax
  have hcmin : coordMin x = x imin := by unfold coordMin; exact himin
  rw [hcm, hcmin]
  have hpos : 0 ≤ x imax - x imin := by
    linarith [coordMin_le x imax, le_coordMax x imin]
  have hbound : x imax - x imin ≤ 2 * M := by
    have := h imax imin
    rwa [abs_of_nonneg hpos] at this
  linarith

/--
For any two indices `i j`, `|x i - x j| / 2 ≤ variationSeminorm x`.
-/
theorem variationSeminorm_half_abs_diff_le {x : ι → ℝ} (i j : ι) :
    |x i - x j| / 2 ≤ variationSeminorm x := by
  unfold variationSeminorm oscillation
  have hi_max : x i ≤ coordMax x := le_coordMax x i
  have hj_min : coordMin x ≤ x j := coordMin_le x j
  have hi_min : coordMin x ≤ x i := coordMin_le x i
  have hj_max : x j ≤ coordMax x := le_coordMax x j
  have hle : |x i - x j| ≤ coordMax x - coordMin x := by
    rw [abs_le]
    constructor
    · linarith
    · linarith
  linarith [abs_nonneg (x i - x j)]

/--
For any two indices `i j`, `|x i - x j| ≤ 2 * variationSeminorm x`.
-/
theorem variationSeminorm_coord_diff_le {x : ι → ℝ} (i j : ι) :
    |x i - x j| ≤ 2 * variationSeminorm x := by
  have h := variationSeminorm_half_abs_diff_le i j (x := x)
  linarith [abs_nonneg (x i - x j)]

/--
For any two indices `i j`, `|x i - x j|` is bounded by the oscillation.
-/
theorem abs_coord_diff_le_oscillation {x : ι → ℝ} (i j : ι) :
    |x i - x j| ≤ oscillation x := by
  calc
    |x i - x j| ≤ 2 * variationSeminorm x := variationSeminorm_coord_diff_le i j
    _ = oscillation x := (oscillation_eq_two_mul_variationSeminorm x).symm

/--
If `a ≤ x i ≤ b` for all `i`, then `variationSeminorm x ≤ (b - a) / 2`.
-/
theorem variationSeminorm_le_of_range_bound {x : ι → ℝ} {a b : ℝ}
    (h : ∀ i, a ≤ x i ∧ x i ≤ b) :
    variationSeminorm x ≤ (b - a) / 2 := by
  unfold variationSeminorm oscillation
  have hmax : coordMax x ≤ b := by
    unfold coordMax
    apply Finset.sup'_le
    intro i _
    exact (h i).2
  have hmin : a ≤ coordMin x := by
    unfold coordMin
    apply Finset.le_inf'
    intro i _
    exact (h i).1
  linarith

/--
If `|x i| ≤ M` for all `i`, then `variationSeminorm x ≤ M`.
-/
theorem variationSeminorm_le_forall_abs_le {x : ι → ℝ} {M : ℝ}
    (h : ∀ i, |x i| ≤ M) :
    variationSeminorm x ≤ M := by
  have hbound : variationSeminorm x ≤ (M - (-M)) / 2 := by
    apply variationSeminorm_le_of_range_bound
    intro i
    have hi := h i
    rw [abs_le] at hi
    exact ⟨hi.1, hi.2⟩
  linarith

/-- Triangle inequality for subtraction of the variation seminorm. -/
theorem variationSeminorm_sub_le (x y : ι → ℝ) :
    variationSeminorm (x - y) ≤ variationSeminorm x + variationSeminorm y := by
  calc variationSeminorm (x - y) = variationSeminorm (x + (-y)) := by simp [sub_eq_add_neg]
    _ ≤ variationSeminorm x + variationSeminorm (-y) := variationSeminorm_add x (-y)
    _ = variationSeminorm x + variationSeminorm y := by rw [variationSeminorm_neg]

/--
Shifting the right operand of a subtraction by a constant does not change variation seminorm.
-/
theorem variationSeminorm_sub_add_const_right (x y : ι → ℝ) (c : ℝ) :
    variationSeminorm (fun i => x i - (y i + c)) = variationSeminorm (x - y) := by
  have hrepr : (fun i => x i - (y i + c)) = fun i => (x - y) i + (-c) := by
    funext i
    simp [sub_eq_add_neg, add_left_comm, add_comm]
  rw [hrepr, variationSeminorm_add_const (x := x - y) (c := -c)]

/--
Shifting the left operand of a subtraction by a constant does not change variation seminorm.
-/
theorem variationSeminorm_add_const_sub (x y : ι → ℝ) (c : ℝ) :
    variationSeminorm (fun i => (x i + c) - y i) = variationSeminorm (x - y) := by
  have hrepr : (fun i => (x i + c) - y i) = fun i => (x - y) i + c := by
    funext i
    simp [sub_eq_add_neg, add_assoc, add_comm]
  rw [hrepr, variationSeminorm_add_const (x := x - y) (c := c)]

/--
Three-point triangle inequality:
`variationSeminorm (x - z) ≤ variationSeminorm (x - y) + variationSeminorm (y - z)`.
-/
theorem variationSeminorm_sub_triangle (x y z : ι → ℝ) :
    variationSeminorm (x - z) ≤ variationSeminorm (x - y) + variationSeminorm (y - z) := by
  have hdecomp : x - z = (x - y) + (y - z) := by
    funext i
    have hscalar : x i - z i = (x i - y i) + (y i - z i) := by linarith
    simpa [Pi.sub_apply, Pi.add_apply] using hscalar
  rw [hdecomp]
  exact variationSeminorm_add (x - y) (y - z)

/--
Reverse triangle inequality for the variation seminorm.
-/
theorem variationSeminorm_abs_sub_le (x y : ι → ℝ) :
    |variationSeminorm x - variationSeminorm y| ≤ variationSeminorm (x - y) := by
  have hxy_add : (x - y) + y = x := by
    funext i
    simp [sub_eq_add_neg, add_comm]
  have hyx_add : (y - x) + x = y := by
    funext i
    simp [sub_eq_add_neg, add_comm]
  have hx : variationSeminorm x ≤ variationSeminorm (x - y) + variationSeminorm y := by
    calc
      variationSeminorm x = variationSeminorm ((x - y) + y) := by rw [hxy_add]
      _ ≤ variationSeminorm (x - y) + variationSeminorm y := variationSeminorm_add (x - y) y
  have hy : variationSeminorm y ≤ variationSeminorm (y - x) + variationSeminorm x := by
    calc
      variationSeminorm y = variationSeminorm ((y - x) + x) := by rw [hyx_add]
      _ ≤ variationSeminorm (y - x) + variationSeminorm x := variationSeminorm_add (y - x) x
  have hright : variationSeminorm x - variationSeminorm y ≤ variationSeminorm (x - y) := by
    linarith [hx]
  have hy' : variationSeminorm y - variationSeminorm x ≤ variationSeminorm (x - y) := by
    have htmp : variationSeminorm y - variationSeminorm x ≤ variationSeminorm (y - x) := by
      linarith [hy]
    simpa [variationSeminorm_sub_comm] using htmp
  have hleft : -variationSeminorm (x - y) ≤ variationSeminorm x - variationSeminorm y := by
    linarith [hy']
  exact (abs_le.mpr ⟨hleft, hright⟩)

/--
Triangle inequality for subtraction of oscillation.
-/
theorem oscillation_sub_le (x y : ι → ℝ) :
    oscillation (x - y) ≤ oscillation x + oscillation y := by
  have h := variationSeminorm_sub_le x y
  calc
    oscillation (x - y) = 2 * variationSeminorm (x - y) :=
      oscillation_eq_two_mul_variationSeminorm (x - y)
    _ ≤ 2 * (variationSeminorm x + variationSeminorm y) := by nlinarith
    _ = oscillation x + oscillation y := by
      rw [oscillation_eq_two_mul_variationSeminorm x, oscillation_eq_two_mul_variationSeminorm y]
      ring

/--
Reverse triangle inequality for oscillation.
-/
theorem oscillation_abs_sub_le (x y : ι → ℝ) :
    |oscillation x - oscillation y| ≤ oscillation (x - y) := by
  have hvar := variationSeminorm_abs_sub_le x y
  have hmul :
      2 * |variationSeminorm x - variationSeminorm y| ≤ 2 * variationSeminorm (x - y) :=
    mul_le_mul_of_nonneg_left hvar (by positivity)
  have hrepr : oscillation x - oscillation y = 2 * (variationSeminorm x - variationSeminorm y) := by
    rw [oscillation_eq_two_mul_variationSeminorm x, oscillation_eq_two_mul_variationSeminorm y]
    ring
  calc
    |oscillation x - oscillation y|
        = |2 * (variationSeminorm x - variationSeminorm y)| := by rw [hrepr]
    _ = 2 * |variationSeminorm x - variationSeminorm y| := by simp [abs_mul]
    _ ≤ 2 * variationSeminorm (x - y) := hmul
    _ = oscillation (x - y) := by
      rw [oscillation_eq_two_mul_variationSeminorm]

/--
Three-point triangle inequality:
`oscillation (x - z) ≤ oscillation (x - y) + oscillation (y - z)`.
-/
theorem oscillation_sub_triangle (x y z : ι → ℝ) :
    oscillation (x - z) ≤ oscillation (x - y) + oscillation (y - z) := by
  have hvar : variationSeminorm (x - z) ≤
      variationSeminorm (x - y) + variationSeminorm (y - z) :=
    variationSeminorm_sub_triangle x y z
  calc
    oscillation (x - z) = 2 * variationSeminorm (x - z) :=
      oscillation_eq_two_mul_variationSeminorm (x - z)
    _ ≤ 2 * (variationSeminorm (x - y) + variationSeminorm (y - z)) := by
      nlinarith
    _ = oscillation (x - y) + oscillation (y - z) := by
      rw [oscillation_eq_two_mul_variationSeminorm (x - y),
        oscillation_eq_two_mul_variationSeminorm (y - z)]
      ring

/-- Triangle inequality for finite sums of the variation seminorm. -/
theorem variationSeminorm_finset_sum_le {β : Type*} (s : Finset β) (f : β → ι → ℝ) :
    variationSeminorm (∑ a ∈ s, f a) ≤ ∑ a ∈ s, variationSeminorm (f a) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp [variationSeminorm_zero]
  | insert a s' hmem ih =>
      rw [Finset.sum_insert hmem, Finset.sum_insert hmem]
      have h1 := variationSeminorm_add (f a) (∑ x ∈ s', f x)
      linarith

/-- A constant function has zero variation seminorm (explicit-name alias). -/
theorem variationSeminorm_smul_const (c : ℝ) :
    variationSeminorm (fun _ : ι => c) = 0 := variationSeminorm_const c

/-- The variation seminorm equals half the oscillation. -/
theorem variationSeminorm_eq_half_oscillation (x : ι → ℝ) :
    variationSeminorm x = oscillation x / 2 := rfl

/-! ## Utility lemmas -/

/--
`coordMax` is monotone: if `x ≤ y` pointwise, then `coordMax x ≤ coordMax y`.
-/
lemma coordMax_mono {x y : ι → ℝ} (h : x ≤ y) : coordMax x ≤ coordMax y := by
  unfold coordMax
  apply Finset.sup'_le
  intro i _
  exact le_trans (h i) (Finset.le_sup' y (Finset.mem_univ i))

/--
`coordMin` is monotone: if `x ≤ y` pointwise, then `coordMin x ≤ coordMin y`.
-/
lemma coordMin_mono {x y : ι → ℝ} (h : x ≤ y) : coordMin x ≤ coordMin y := by
  unfold coordMin
  apply Finset.le_inf'
  intro i _
  exact le_trans (Finset.inf'_le x (Finset.mem_univ i)) (h i)

/--
`oscillation` is bounded above by `2 * M` when all coordinates lie in `[-M, M]`.
-/
lemma oscillation_le_two_mul_of_abs_le {x : ι → ℝ} {M : ℝ}
    (h : ∀ i, |x i| ≤ M) : oscillation x ≤ 2 * M := by
  unfold oscillation
  have hmax : coordMax x ≤ M := by
    unfold coordMax
    apply Finset.sup'_le
    intro i _
    exact (abs_le.mp (h i)).2
  have hmin : -M ≤ coordMin x := by
    unfold coordMin
    apply Finset.le_inf'
    intro i _
    exact (abs_le.mp (h i)).1
  linarith

/--
`variationSeminorm` is bounded above by `M` when all coordinates lie in `[-M, M]`.
-/
lemma variationSeminorm_le_of_abs_le {x : ι → ℝ} {M : ℝ}
    (h : ∀ i, |x i| ≤ M) : variationSeminorm x ≤ M := by
  unfold variationSeminorm
  have hosc := oscillation_le_two_mul_of_abs_le h
  linarith

/--
`coordMax` of the zero function is 0.
-/
lemma coordMax_zero : coordMax (fun _ : ι => (0 : ℝ)) = 0 := by
  apply le_antisymm
  · apply Finset.sup'_le Finset.univ_nonempty
    intros i _
    rfl
  · exact le_coordMax (fun _ => 0) (Classical.choice inferInstance)

/--
`coordMin` of the zero function is 0.
-/
lemma coordMin_zero : coordMin (fun _ : ι => (0 : ℝ)) = 0 := by
  apply le_antisymm
  · exact coordMin_le (fun _ => 0) (Classical.choice inferInstance)
  · apply Finset.le_inf' Finset.univ_nonempty
    intros i _
    rfl

/--
`oscillation` of the zero function is 0.
-/
lemma oscillation_zero : oscillation (fun _ : ι => (0 : ℝ)) = 0 := by
  simp [oscillation, coordMax_zero, coordMin_zero]

end KLProjection
end FlowSinkhorn
