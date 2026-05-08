import FlowSinkhorn.KLProjection.Variation
import FlowSinkhorn.KLProjection.UniformBound

/-!
# Topical maps

A map `T : (ι → ℝ) → (ι → ℝ)` is **topical** if it is both:
1. **Monotone**: `x ≤ y` pointwise implies `T x ≤ T y` pointwise.
2. **Translation-equivariant**: `T (x + c) = T x + c` for all `x` and constant `c : ℝ`.

Topical maps appear naturally in iterative algorithms for optimal transport, Wasserstein
distances on graphs, and more generally in KL-projection-based algorithms. This module
provides their abstract theory, independently of any specific application.

Key results:
- `IsTopical`: bundled predicate.
- Closure under identity, composition, and iteration.
- `variationSeminorm_nonexpansive_of_isTopical`: the variation seminorm is nonexpansive
  under topical maps (abstract form of Proposition `prop:topical-nonexpansive`).
- `SeminormNonexpansive_variationSeminormAsSeminorm_of_isTopical`: bridge to the
  `SeminormNonexpansive` predicate used by the generic blueprint.

**Re-use note**: this file can be imported by any module that works with topical maps.
It has no dependency on KL divergence, optimal transport, or graph structure.
-/

namespace FlowSinkhorn
namespace KLProjection

-- Paper-facing theorem names are intentionally verbose for traceability.
set_option linter.style.longLine false

variable {ι : Type*}

/--
A map on finite real-valued functions is **topical** if it is monotone and
translation-equivariant.

This bundles the two hypotheses of Proposition `prop:topical-nonexpansive`:
monotone + translation-equivariant implies variation-seminorm-nonexpansive.
-/
structure IsTopical (T : (ι → ℝ) → (ι → ℝ)) : Prop where
  /-- The map is monotone with respect to the pointwise order. -/
  mono : Monotone T
  /-- The map commutes with adding a constant to all coordinates. -/
  trans : TranslationEquivariant T

/--
The identity map is topical.
-/
theorem isTopical_id : IsTopical (id : (ι → ℝ) → (ι → ℝ)) :=
  ⟨fun _ _ h => h, fun _ _ => rfl⟩

/--
The composition of two topical maps is topical.
-/
theorem isTopical_comp {T₁ T₂ : (ι → ℝ) → (ι → ℝ)}
    (h₁ : IsTopical T₁) (h₂ : IsTopical T₂) :
    IsTopical (T₂ ∘ T₁) :=
  ⟨h₂.mono.comp h₁.mono, fun x c => by
    simp only [Function.comp]
    rw [h₁.trans, h₂.trans]⟩

/--
Every iterate of a topical map is topical.
-/
theorem isTopical_iterate {T : (ι → ℝ) → (ι → ℝ)} (h : IsTopical T) :
    ∀ k : ℕ, IsTopical (T^[k]) := by
  intro k
  induction k with
  | zero =>
      simp only [Function.iterate_zero]
      exact isTopical_id
  | succ k ih =>
      rw [Function.iterate_succ']
      exact isTopical_comp ih h

/--
Successive iterates of a topical map stay topical.
-/
theorem isTopical_iterate_succ {T : (ι → ℝ) → (ι → ℝ)} (h : IsTopical T) (k : ℕ) :
    IsTopical (T^[k + 1]) := by
  simpa [Nat.succ_eq_add_one] using isTopical_iterate h (k + 1)

/--
Composing on the left by an iterate of a topical map preserves topicality.
-/
theorem isTopical_comp_iterate_left
    {S T : (ι → ℝ) → (ι → ℝ)}
    (hS : IsTopical S) (hT : IsTopical T) (k : ℕ) :
    IsTopical ((T^[k]) ∘ S) :=
  isTopical_comp hS (isTopical_iterate hT k)

/--
Composing on the right by an iterate of a topical map preserves topicality.
-/
theorem isTopical_comp_iterate_right
    {S T : (ι → ℝ) → (ι → ℝ)}
    (hS : IsTopical S) (hT : IsTopical T) (k : ℕ) :
    IsTopical (S ∘ (T^[k])) :=
  isTopical_comp (isTopical_iterate hT k) hS

/--
Any nested iterate of a topical map is topical.
-/
theorem isTopical_iterate_iterate
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m k : ℕ) :
    IsTopical (((T^[m])^[k])) := by
  exact isTopical_iterate (isTopical_iterate hT m) k

section FiniteIndex

variable [Fintype ι] [Nonempty ι]

/--
Topical maps are nonexpansive for the variation seminorm.

This is the abstract core of Proposition `prop:topical-nonexpansive` in the paper.
-/
theorem variationSeminorm_nonexpansive_of_isTopical {T : (ι → ℝ) → (ι → ℝ)}
    (h : IsTopical T) (x y : ι → ℝ) :
    variationSeminorm (T x - T y) ≤ variationSeminorm (x - y) :=
  variationSeminorm_nonexpansive_of_topical T h.mono h.trans x y

/--
Topical maps are nonexpansive for oscillation.
-/
theorem oscillation_nonexpansive_of_isTopical {T : (ι → ℝ) → (ι → ℝ)}
    (h : IsTopical T) (x y : ι → ℝ) :
    oscillation (T x - T y) ≤ oscillation (x - y) :=
  oscillation_nonexpansive_of_topical T h.mono h.trans x y

/--
A topical map is `SeminormNonexpansive` for `variationSeminormAsSeminorm`.

This is the bridge between the `IsTopical` predicate and the `SeminormNonexpansive`
predicate used in the generic convergence blueprint (`PrimalDualBounds/Blueprint.lean`).
-/
theorem SeminormNonexpansive_variationSeminormAsSeminorm_of_isTopical
    {T : (ι → ℝ) → (ι → ℝ)} (h : IsTopical T) :
    SeminormNonexpansive variationSeminormAsSeminorm T := by
  intro x y
  change variationSeminorm (T x - T y) ≤ variationSeminorm (x - y)
  exact variationSeminorm_nonexpansive_of_isTopical h x y

/--
Every iterate of a topical map is `SeminormNonexpansive` for `variationSeminormAsSeminorm`.
-/
theorem SeminormNonexpansive_iterate_of_isTopical {T : (ι → ℝ) → (ι → ℝ)}
    (h : IsTopical T) (k : ℕ) :
    SeminormNonexpansive variationSeminormAsSeminorm (T^[k]) :=
  SeminormNonexpansive_variationSeminormAsSeminorm_of_isTopical (isTopical_iterate h k)

/--
Successor-index `SeminormNonexpansive` bridge for topical iterates.
-/
theorem SeminormNonexpansive_iterate_succ_of_isTopical
    {T : (ι → ℝ) → (ι → ℝ)} (h : IsTopical T) (k : ℕ) :
    SeminormNonexpansive variationSeminormAsSeminorm (T^[k + 1]) := by
  simpa [Nat.succ_eq_add_one] using
    SeminormNonexpansive_iterate_of_isTopical h (k + 1)

/--
Iterates of a topical map are nonexpansive for the variation seminorm.
-/
theorem variationSeminorm_nonexpansive_iterate_of_isTopical {T : (ι → ℝ) → (ι → ℝ)}
    (h : IsTopical T) (k : ℕ) (x y : ι → ℝ) :
    variationSeminorm ((T^[k]) x - (T^[k]) y) ≤ variationSeminorm (x - y) := by
  exact (SeminormNonexpansive_iterate_of_isTopical h k) x y

/--
Iterates of a topical map are nonexpansive for oscillation.
-/
theorem oscillation_nonexpansive_iterate_of_isTopical {T : (ι → ℝ) → (ι → ℝ)}
    (h : IsTopical T) (k : ℕ) (x y : ι → ℝ) :
    oscillation ((T^[k]) x - (T^[k]) y) ≤ oscillation (x - y) :=
  oscillation_nonexpansive_of_isTopical (isTopical_iterate h k) x y

/--
The `(k+1)` iterate of a topical map is variation-seminorm-nonexpansive.
-/
theorem variationSeminorm_nonexpansive_iterate_succ_of_isTopical
    {T : (ι → ℝ) → (ι → ℝ)} (h : IsTopical T) (k : ℕ) (x y : ι → ℝ) :
    variationSeminorm ((T^[k + 1]) x - (T^[k + 1]) y) ≤ variationSeminorm (x - y) := by
  simpa [Nat.succ_eq_add_one] using
    variationSeminorm_nonexpansive_iterate_of_isTopical h (k + 1) x y

/--
The `(k+1)` iterate of a topical map is oscillation-nonexpansive.
-/
theorem oscillation_nonexpansive_iterate_succ_of_isTopical
    {T : (ι → ℝ) → (ι → ℝ)} (h : IsTopical T) (k : ℕ) (x y : ι → ℝ) :
    oscillation ((T^[k + 1]) x - (T^[k + 1]) y) ≤ oscillation (x - y) := by
  simpa [Nat.succ_eq_add_one] using
    oscillation_nonexpansive_iterate_of_isTopical h (k + 1) x y

/--
Uniform orbit bound from topicality, fixed-point control, and an arbitrary base iterate.

This is the bundled-`IsTopical` bridge to the generic nonexpansive fixed-point estimate:
if `T` is topical and `variationSeminorm uStar ≤ B`, then
`variationSeminorm ((T^[k]) u0) ≤ variationSeminorm u0 + 2 * B`.
-/
theorem variationSeminorm_orbitBound_with_base_of_isTopical
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B : ℝ} (hbound : variationSeminorm uStar ≤ B)
    (k : ℕ) :
    variationSeminorm ((T^[k]) u0) ≤ variationSeminorm u0 + 2 * B := by
  have hne : SeminormNonexpansive variationSeminormAsSeminorm T :=
    SeminormNonexpansive_variationSeminormAsSeminorm_of_isTopical hT
  simpa using
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
      variationSeminormAsSeminorm T hne (uStar := uStar) (u0 := u0) hfix hbound k

/--
Budget-lifted orbit bound for topical maps.

If the fixed-point control is first established at some base constant `B` and then one has
`B ≤ U`, this upgrades the iterate bound to the larger budget `U`.
-/
theorem variationSeminorm_orbitBound_with_base_of_isTopical_of_bound_le
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (k : ℕ) :
    variationSeminorm ((T^[k]) u0) ≤ variationSeminorm u0 + 2 * U := by
  have hbase :
      variationSeminorm ((T^[k]) u0) ≤ variationSeminorm u0 + 2 * B :=
    variationSeminorm_orbitBound_with_base_of_isTopical hT hfix hbound k
  nlinarith

/--
Zero-seed orbit bound from topicality and fixed-point control.

This is the direct zero-base specialization of
`variationSeminorm_orbitBound_with_base_of_isTopical`.
-/
theorem variationSeminorm_orbitBound_from_zero_of_isTopical
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B : ℝ} (hbound : variationSeminorm uStar ≤ B)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((T^[k]) u0) ≤ 2 * B := by
  have hbase :
      variationSeminorm ((T^[k]) u0) ≤ variationSeminorm u0 + 2 * B :=
    variationSeminorm_orbitBound_with_base_of_isTopical hT hfix hbound k
  rw [hzero, zero_add] at hbase
  exact hbase

/--
Zero-seed orbit bound with budget lifting.

If `variationSeminorm uStar ≤ B` and `B ≤ U`, then
`variationSeminorm ((T^[k]) u0) ≤ 2 * U` for any zero-seed `u0`.
-/
theorem variationSeminorm_orbitBound_from_zero_of_isTopical_of_bound_le
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((T^[k]) u0) ≤ 2 * U := by
  have hbase :
      variationSeminorm ((T^[k]) u0) ≤ variationSeminorm u0 + 2 * U :=
    variationSeminorm_orbitBound_with_base_of_isTopical_of_bound_le hT hfix hbound hBU k
  rw [hzero, zero_add] at hbase
  exact hbase

/--
Successor-index base orbit bound for topical maps.
-/
theorem variationSeminorm_orbitBound_with_base_of_isTopical_succ
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B : ℝ} (hbound : variationSeminorm uStar ≤ B)
    (k : ℕ) :
    variationSeminorm ((T^[k + 1]) u0) ≤ variationSeminorm u0 + 2 * B := by
  simpa [Nat.succ_eq_add_one] using
    variationSeminorm_orbitBound_with_base_of_isTopical hT hfix hbound (k + 1)

/--
Successor-index zero-seed orbit bound for topical maps.
-/
theorem variationSeminorm_orbitBound_from_zero_of_isTopical_succ
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B : ℝ} (hbound : variationSeminorm uStar ≤ B)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((T^[k + 1]) u0) ≤ 2 * B := by
  simpa [Nat.succ_eq_add_one] using
    variationSeminorm_orbitBound_from_zero_of_isTopical hT hfix hbound hzero (k + 1)

/--
Successor-index budget-lifted base orbit bound for topical maps.

This combines the successor-index form with budget lifting `B ≤ U`.
-/
theorem variationSeminorm_orbitBound_with_base_of_isTopical_succ_of_bound_le
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (k : ℕ) :
    variationSeminorm ((T^[k + 1]) u0) ≤ variationSeminorm u0 + 2 * U := by
  simpa [Nat.succ_eq_add_one] using
    variationSeminorm_orbitBound_with_base_of_isTopical_of_bound_le hT hfix hbound hBU (k + 1)

/--
Successor-index budget-lifted zero-seed orbit bound for topical maps.
-/
theorem variationSeminorm_orbitBound_from_zero_of_isTopical_succ_of_bound_le
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((T^[k + 1]) u0) ≤ 2 * U := by
  simpa [Nat.succ_eq_add_one] using
    variationSeminorm_orbitBound_from_zero_of_isTopical_of_bound_le
      hT hfix hbound hBU hzero (k + 1)

/--
If the lifted budget is nonpositive, the topical orbit stays below the base variation seminorm.
-/
theorem variationSeminorm_orbitBound_with_base_of_isTopical_le_base_of_budget_nonpos
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hU_nonpos : U ≤ 0)
    (k : ℕ) :
    variationSeminorm ((T^[k]) u0) ≤ variationSeminorm u0 := by
  have hbase :
      variationSeminorm ((T^[k]) u0) ≤ variationSeminorm u0 + 2 * U :=
    variationSeminorm_orbitBound_with_base_of_isTopical_of_bound_le
      hT hfix hbound hBU k
  calc
    variationSeminorm ((T^[k]) u0)
        ≤ variationSeminorm u0 + 2 * U := hbase
    _ ≤ variationSeminorm u0 := by nlinarith [hU_nonpos]

/--
Successor-index form of
`variationSeminorm_orbitBound_with_base_of_isTopical_le_base_of_budget_nonpos`.
-/
theorem variationSeminorm_orbitBound_with_base_of_isTopical_succ_le_base_of_budget_nonpos
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hU_nonpos : U ≤ 0)
    (k : ℕ) :
    variationSeminorm ((T^[k + 1]) u0) ≤ variationSeminorm u0 := by
  simpa [Nat.succ_eq_add_one] using
    variationSeminorm_orbitBound_with_base_of_isTopical_le_base_of_budget_nonpos
      hT hfix hbound hBU hU_nonpos (k + 1)

/--
Zero-seed orbit collapses to exact zero when the lifted budget is nonpositive.
-/
theorem variationSeminorm_orbitBound_from_zero_of_isTopical_eq_zero_of_bound_le_of_budget_nonpos
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hU_nonpos : U ≤ 0)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((T^[k]) u0) = 0 := by
  have hboundU : variationSeminorm ((T^[k]) u0) ≤ 2 * U :=
    variationSeminorm_orbitBound_from_zero_of_isTopical_of_bound_le hT hfix hbound hBU hzero k
  have hle0 : variationSeminorm ((T^[k]) u0) ≤ 0 := by
    have h2U_nonpos : 2 * U ≤ 0 := by nlinarith
    exact hboundU.trans h2U_nonpos
  exact le_antisymm hle0 (variationSeminorm_nonneg _)

/--
Successor-index zero-seed orbit collapses to exact zero under a nonpositive lifted budget.
-/
theorem
    variationSeminorm_orbitBound_from_zero_of_isTopical_succ_eq_zero_of_bound_le_of_budget_nonpos
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hU_nonpos : U ≤ 0)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((T^[k + 1]) u0) = 0 := by
  simpa [Nat.succ_eq_add_one] using
    variationSeminorm_orbitBound_from_zero_of_isTopical_eq_zero_of_bound_le_of_budget_nonpos
      hT hfix hbound hBU hU_nonpos hzero (k + 1)

/--
Uniform orbit bound for stride-iterates of a topical map.

This applies the same base estimate to the map `T^[m]`, useful when complexity statements
are written per sweep/epoch rather than per atomic update.
-/
theorem variationSeminorm_orbitBound_with_base_of_isTopical_iterate
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : (T^[m]) uStar = uStar)
    {B : ℝ} (hbound : variationSeminorm uStar ≤ B)
    (k : ℕ) :
    variationSeminorm (((T^[m])^[k]) u0) ≤ variationSeminorm u0 + 2 * B := by
  have hne : SeminormNonexpansive variationSeminormAsSeminorm (T^[m]) :=
    SeminormNonexpansive_iterate_of_isTopical hT m
  simpa using
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
      variationSeminormAsSeminorm (T^[m]) hne (uStar := uStar) (u0 := u0) hfix hbound k

/--
Budget-lifted stride-iterate orbit bound for topical maps.
-/
theorem variationSeminorm_orbitBound_with_base_of_isTopical_iterate_of_bound_le
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : (T^[m]) uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (k : ℕ) :
    variationSeminorm (((T^[m])^[k]) u0) ≤ variationSeminorm u0 + 2 * U := by
  have hbase :
      variationSeminorm (((T^[m])^[k]) u0) ≤ variationSeminorm u0 + 2 * B :=
    variationSeminorm_orbitBound_with_base_of_isTopical_iterate hT m hfix hbound k
  nlinarith

/--
Successor-index budget-lifted stride-iterate base orbit bound for topical maps.
-/
theorem variationSeminorm_orbitBound_with_base_of_isTopical_iterate_succ_of_bound_le
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : (T^[m]) uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (k : ℕ) :
    variationSeminorm (((T^[m])^[k + 1]) u0) ≤ variationSeminorm u0 + 2 * U := by
  simpa [Nat.succ_eq_add_one] using
    variationSeminorm_orbitBound_with_base_of_isTopical_iterate_of_bound_le
      hT m hfix hbound hBU (k + 1)

/--
Successor-index + index-threshold convenience form of
`variationSeminorm_orbitBound_with_base_of_isTopical_iterate_succ_of_bound_le`.
-/
theorem
    variationSeminorm_orbitBound_with_base_of_isTopical_iterate_succ_of_bound_le_of_le_index
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : (T^[m]) uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    {k n : ℕ} (hk : k + 1 ≤ n) :
    variationSeminorm (((T^[m])^[k + 1]) u0) ≤ variationSeminorm u0 + 2 * U := by
  have _hidx : n - (k + 1) + (k + 1) = n := Nat.sub_add_cancel hk
  exact variationSeminorm_orbitBound_with_base_of_isTopical_iterate_succ_of_bound_le
    hT m hfix hbound hBU k

/--
If the lifted stride-iterate budget is nonpositive, the topical orbit stays below
the base variation seminorm.
-/
theorem variationSeminorm_orbitBound_with_base_of_isTopical_iterate_le_base_of_budget_nonpos
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : (T^[m]) uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hU_nonpos : U ≤ 0)
    (k : ℕ) :
    variationSeminorm (((T^[m])^[k]) u0) ≤ variationSeminorm u0 := by
  have hbase :
      variationSeminorm (((T^[m])^[k]) u0) ≤ variationSeminorm u0 + 2 * U :=
    variationSeminorm_orbitBound_with_base_of_isTopical_iterate_of_bound_le
      hT m hfix hbound hBU k
  calc
    variationSeminorm (((T^[m])^[k]) u0)
        ≤ variationSeminorm u0 + 2 * U := hbase
    _ ≤ variationSeminorm u0 := by nlinarith [hU_nonpos]

/--
Successor-index form of
`variationSeminorm_orbitBound_with_base_of_isTopical_iterate_le_base_of_budget_nonpos`.
-/
theorem
    variationSeminorm_orbitBound_with_base_of_isTopical_iterate_succ_le_base_of_budget_nonpos
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : (T^[m]) uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hU_nonpos : U ≤ 0)
    (k : ℕ) :
    variationSeminorm (((T^[m])^[k + 1]) u0) ≤ variationSeminorm u0 := by
  simpa [Nat.succ_eq_add_one] using
    variationSeminorm_orbitBound_with_base_of_isTopical_iterate_le_base_of_budget_nonpos
      hT m hfix hbound hBU hU_nonpos (k + 1)

/--
Successor-index + index-threshold convenience form of
`variationSeminorm_orbitBound_with_base_of_isTopical_iterate_succ_le_base_of_budget_nonpos`.
-/
theorem
    variationSeminorm_orbitBound_with_base_of_isTopical_iterate_succ_le_base_of_budget_nonpos_of_le_index
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : (T^[m]) uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hU_nonpos : U ≤ 0)
    {k n : ℕ} (hk : k + 1 ≤ n) :
    variationSeminorm (((T^[m])^[k + 1]) u0) ≤ variationSeminorm u0 := by
  have _hidx : n - (k + 1) + (k + 1) = n := Nat.sub_add_cancel hk
  exact
    variationSeminorm_orbitBound_with_base_of_isTopical_iterate_succ_le_base_of_budget_nonpos
      hT m hfix hbound hBU hU_nonpos k

/--
Zero-seed stride-iterate orbit bound for topical maps.
-/
theorem variationSeminorm_orbitBound_from_zero_of_isTopical_iterate
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : (T^[m]) uStar = uStar)
    {B : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm (((T^[m])^[k]) u0) ≤ 2 * B := by
  have hbase :
      variationSeminorm (((T^[m])^[k]) u0) ≤ variationSeminorm u0 + 2 * B :=
    variationSeminorm_orbitBound_with_base_of_isTopical_iterate hT m hfix hbound k
  rw [hzero, zero_add] at hbase
  exact hbase

/--
Successor-index zero-seed stride-iterate orbit bound for topical maps.
-/
theorem variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : (T^[m]) uStar = uStar)
    {B : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm (((T^[m])^[k + 1]) u0) ≤ 2 * B := by
  simpa [Nat.succ_eq_add_one] using
    variationSeminorm_orbitBound_from_zero_of_isTopical_iterate hT m hfix hbound hzero (k + 1)

/--
Zero-seed stride-iterate budget bound for topical maps.

Combines the stride-iterate orbit estimate with `variationSeminorm u0 = 0` and budget lifting
`B ≤ U` to obtain the explicit form `variationSeminorm (((T^[m])^[k]) u0) ≤ 2 * U`.
-/
theorem variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_bound_le
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : (T^[m]) uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm (((T^[m])^[k]) u0) ≤ 2 * U := by
  have hbase :
      variationSeminorm (((T^[m])^[k]) u0) ≤ variationSeminorm u0 + 2 * B :=
    variationSeminorm_orbitBound_with_base_of_isTopical_iterate hT m hfix hbound k
  rw [hzero, zero_add] at hbase
  nlinarith

/--
Successor-index budget-lifted zero-seed stride-iterate orbit bound for topical maps.
-/
theorem variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_of_bound_le
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : (T^[m]) uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm (((T^[m])^[k + 1]) u0) ≤ 2 * U := by
  simpa [Nat.succ_eq_add_one] using
    variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_bound_le
      hT m hfix hbound hBU hzero (k + 1)

/--
Zero-seed stride-iterate orbit collapses to exact zero under a nonpositive lifted budget.
-/
theorem variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_eq_zero_of_budget_nonpos
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : (T^[m]) uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hU_nonpos : U ≤ 0)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm (((T^[m])^[k]) u0) = 0 := by
  have hboundU : variationSeminorm (((T^[m])^[k]) u0) ≤ 2 * U :=
    variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_bound_le
      hT m hfix hbound hBU hzero k
  have hle0 : variationSeminorm (((T^[m])^[k]) u0) ≤ 0 := by
    have h2U_nonpos : 2 * U ≤ 0 := by nlinarith
    exact hboundU.trans h2U_nonpos
  exact le_antisymm hle0 (variationSeminorm_nonneg _)

/--
Successor-index form of
`variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_eq_zero_of_budget_nonpos`.
-/
theorem variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_eq_zero_of_budget_nonpos
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : (T^[m]) uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hU_nonpos : U ≤ 0)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm (((T^[m])^[k + 1]) u0) = 0 := by
  simpa [Nat.succ_eq_add_one] using
    variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_eq_zero_of_budget_nonpos
      hT m hfix hbound hBU hU_nonpos hzero (k + 1)

/--
Successor-index + index-threshold convenience form of
`variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_eq_zero_of_budget_nonpos`.
-/
theorem
    variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_eq_zero_of_budget_nonpos_of_le_index
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : (T^[m]) uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hU_nonpos : U ≤ 0)
    (hzero : variationSeminorm u0 = 0)
    {k n : ℕ} (hk : k + 1 ≤ n) :
    variationSeminorm (((T^[m])^[k + 1]) u0) = 0 := by
  have _hidx : n - (k + 1) + (k + 1) = n := Nat.sub_add_cancel hk
  exact variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_eq_zero_of_budget_nonpos
    hT m hfix hbound hBU hU_nonpos hzero k

/--
Stride-iterate base orbit bound from a fixed point of `T`.

This convenience wrapper avoids restating the iterate fixed-point hypothesis
`(T^[m]) uStar = uStar` when one already has `T uStar = uStar`.
-/
theorem variationSeminorm_orbitBound_with_base_of_isTopical_iterate_of_fixedPoint
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (k : ℕ) :
    variationSeminorm (((T^[m])^[k]) u0) ≤ variationSeminorm u0 + 2 * B := by
  exact variationSeminorm_orbitBound_with_base_of_isTopical_iterate
    hT m (Function.iterate_fixed hfix m) hbound k

/--
Successor-index stride-iterate base orbit bound from a fixed point of `T`.
-/
theorem variationSeminorm_orbitBound_with_base_of_isTopical_iterate_succ_of_fixedPoint
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (k : ℕ) :
    variationSeminorm (((T^[m])^[k + 1]) u0) ≤ variationSeminorm u0 + 2 * B := by
  simpa [Nat.succ_eq_add_one] using
    variationSeminorm_orbitBound_with_base_of_isTopical_iterate_of_fixedPoint
      hT m hfix hbound (k + 1)

/--
Stride-iterate budget-lifted base orbit bound from a fixed point of `T`.

This is the fixed-point counterpart of
`variationSeminorm_orbitBound_with_base_of_isTopical_iterate_of_bound_le`.
-/
theorem variationSeminorm_orbitBound_with_base_of_isTopical_iterate_of_fixedPoint_of_bound_le
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (k : ℕ) :
    variationSeminorm (((T^[m])^[k]) u0) ≤ variationSeminorm u0 + 2 * U := by
  exact variationSeminorm_orbitBound_with_base_of_isTopical_iterate_of_bound_le
    hT m (Function.iterate_fixed hfix m) hbound hBU k

/--
Successor-index budget-lifted stride-iterate base orbit bound from a fixed point of `T`.
-/
theorem variationSeminorm_orbitBound_with_base_of_isTopical_iterate_succ_of_fixedPoint_of_bound_le
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (k : ℕ) :
    variationSeminorm (((T^[m])^[k + 1]) u0) ≤ variationSeminorm u0 + 2 * U := by
  simpa [Nat.succ_eq_add_one] using
    variationSeminorm_orbitBound_with_base_of_isTopical_iterate_of_fixedPoint_of_bound_le
      hT m hfix hbound hBU (k + 1)

/--
Zero-seed stride-iterate orbit bound from a fixed point of `T`.
-/
theorem variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_fixedPoint
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm (((T^[m])^[k]) u0) ≤ 2 * B := by
  exact variationSeminorm_orbitBound_from_zero_of_isTopical_iterate
    hT m (Function.iterate_fixed hfix m) hbound hzero k

/--
Successor-index zero-seed stride-iterate orbit bound from a fixed point of `T`.
-/
theorem variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_of_fixedPoint
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm (((T^[m])^[k + 1]) u0) ≤ 2 * B := by
  simpa [Nat.succ_eq_add_one] using
    variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_fixedPoint
      hT m hfix hbound hzero (k + 1)

/--
Zero-seed stride-iterate budget bound from a fixed point of `T`.

This combines fixed-point iterate reduction with budget lifting `B ≤ U`.
-/
theorem variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_fixedPoint_of_bound_le
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm (((T^[m])^[k]) u0) ≤ 2 * U := by
  exact variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_bound_le
    hT m (Function.iterate_fixed hfix m) hbound hBU hzero k

/--
Successor-index zero-seed budget-lifted stride-iterate orbit bound from a fixed point of `T`.
-/
theorem variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_of_fixedPoint_of_bound_le
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm (((T^[m])^[k + 1]) u0) ≤ 2 * U := by
  simpa [Nat.succ_eq_add_one] using
    variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_fixedPoint_of_bound_le
      hT m hfix hbound hBU hzero (k + 1)

/--
Zero-seed stride-iterate orbit from a fixed point collapses to exact zero when the lifted budget
is nonpositive.
-/
theorem
    variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_fixedPoint_eq_zero_of_bound_le_of_budget_nonpos
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hU_nonpos : U ≤ 0)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm (((T^[m])^[k]) u0) = 0 := by
  have hboundU : variationSeminorm (((T^[m])^[k]) u0) ≤ 2 * U :=
    variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_fixedPoint_of_bound_le
      hT m hfix hbound hBU hzero k
  have hle0 : variationSeminorm (((T^[m])^[k]) u0) ≤ 0 := by
    have h2U_nonpos : 2 * U ≤ 0 := by nlinarith
    exact hboundU.trans h2U_nonpos
  exact le_antisymm hle0 (variationSeminorm_nonneg _)

/--
Successor-index version of the nonpositive-budget zero-collapse for fixed-point stride iterates.
-/
theorem
    variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_of_fixedPoint_eq_zero_of_bound_le_of_budget_nonpos
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hU_nonpos : U ≤ 0)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm (((T^[m])^[k + 1]) u0) = 0 := by
  simpa [Nat.succ_eq_add_one] using
    variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_fixedPoint_eq_zero_of_bound_le_of_budget_nonpos
      hT m hfix hbound hBU hU_nonpos hzero (k + 1)

/--
Index-threshold convenience wrapper for the fixed-point iterate budget-lifted base orbit bound.
-/
theorem
    variationSeminorm_orbitBound_with_base_of_isTopical_iterate_of_fixedPoint_of_bound_le_of_le_index
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    {k n : ℕ} (hk : k ≤ n) :
    variationSeminorm (((T^[m])^[k]) u0) ≤ variationSeminorm u0 + 2 * U := by
  have _hidx : n - k + k = n := Nat.sub_add_cancel hk
  exact variationSeminorm_orbitBound_with_base_of_isTopical_iterate_of_fixedPoint_of_bound_le
    hT m hfix hbound hBU k

/--
Index-threshold convenience wrapper for the fixed-point iterate budget-lifted zero-seed
orbit bound.
-/
theorem
    variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_fixedPoint_of_bound_le_of_le_index
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    {k n : ℕ} (hk : k ≤ n) :
    variationSeminorm (((T^[m])^[k]) u0) ≤ 2 * U := by
  have _hidx : n - k + k = n := Nat.sub_add_cancel hk
  exact variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_fixedPoint_of_bound_le
    hT m hfix hbound hBU hzero k

/--
Index-threshold convenience wrapper for fixed-point iterate nonpositive-budget zero collapse.
-/
theorem
    variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_fixedPoint_eq_zero_of_bound_le_of_budget_nonpos_of_le_index
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hU_nonpos : U ≤ 0)
    (hzero : variationSeminorm u0 = 0)
    {k n : ℕ} (hk : k ≤ n) :
    variationSeminorm (((T^[m])^[k]) u0) = 0 := by
  have _hidx : n - k + k = n := Nat.sub_add_cancel hk
  exact
    variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_fixedPoint_eq_zero_of_bound_le_of_budget_nonpos
      hT m hfix hbound hBU hU_nonpos hzero k

/--
Successor-index + threshold-index convenience wrapper for the fixed-point iterate
budget-lifted base orbit bound.
-/
theorem
    variationSeminorm_orbitBound_with_base_of_isTopical_iterate_succ_of_fixedPoint_of_bound_le_of_le_index
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    {k n : ℕ} (hk : k + 1 ≤ n) :
    variationSeminorm (((T^[m])^[k + 1]) u0) ≤ variationSeminorm u0 + 2 * U := by
  have _hidx : n - (k + 1) + (k + 1) = n := Nat.sub_add_cancel hk
  exact variationSeminorm_orbitBound_with_base_of_isTopical_iterate_succ_of_fixedPoint_of_bound_le
    hT m hfix hbound hBU k

/--
Successor-index + threshold-index convenience wrapper for the fixed-point iterate
budget-lifted zero-seed orbit bound.
-/
theorem
    variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_of_fixedPoint_of_bound_le_of_le_index
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    {k n : ℕ} (hk : k + 1 ≤ n) :
    variationSeminorm (((T^[m])^[k + 1]) u0) ≤ 2 * U := by
  have _hidx : n - (k + 1) + (k + 1) = n := Nat.sub_add_cancel hk
  exact variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_of_fixedPoint_of_bound_le
    hT m hfix hbound hBU hzero k

/--
Successor-index + threshold-index convenience wrapper for fixed-point iterate
nonpositive-budget zero collapse.
-/
theorem
    variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_of_fixedPoint_eq_zero_of_bound_le_of_budget_nonpos_of_le_index
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hU_nonpos : U ≤ 0)
    (hzero : variationSeminorm u0 = 0)
    {k n : ℕ} (hk : k + 1 ≤ n) :
    variationSeminorm (((T^[m])^[k + 1]) u0) = 0 := by
  have _hidx : n - (k + 1) + (k + 1) = n := Nat.sub_add_cancel hk
  exact
    variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_of_fixedPoint_eq_zero_of_bound_le_of_budget_nonpos
      hT m hfix hbound hBU hU_nonpos hzero k

/--
Natural-bound companion of
`variationSeminorm_orbitBound_with_base_of_isTopical_iterate_of_fixedPoint_of_bound_le_of_le_index`.
-/
theorem variationSeminorm_orbitBound_with_base_of_isTopical_iterate_of_fixedPoint_of_bound_le_of_natBound
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (n N : ℕ) (_hNn : N ≤ n) :
    variationSeminorm (((T^[m])^[N]) u0) ≤ variationSeminorm u0 + 2 * U :=
  variationSeminorm_orbitBound_with_base_of_isTopical_iterate_of_fixedPoint_of_bound_le_of_le_index
    hT m hfix hbound hBU (k := N) (n := n) _hNn

/--
Natural-bound companion of
`variationSeminorm_orbitBound_with_base_of_isTopical_iterate_succ_of_fixedPoint_of_bound_le_of_le_index`.
-/
theorem variationSeminorm_orbitBound_with_base_of_isTopical_iterate_succ_of_fixedPoint_of_bound_le_of_natBound
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (n N : ℕ) (hNn : N + 1 ≤ n) :
    variationSeminorm (((T^[m])^[N + 1]) u0) ≤ variationSeminorm u0 + 2 * U :=
  variationSeminorm_orbitBound_with_base_of_isTopical_iterate_succ_of_fixedPoint_of_bound_le_of_le_index
    hT m hfix hbound hBU (k := N) (n := n) hNn

/--
Natural-bound companion of
`variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_fixedPoint_of_bound_le_of_le_index`.
-/
theorem variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_fixedPoint_of_bound_le_of_natBound
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (n N : ℕ) (hNn : N ≤ n) :
    variationSeminorm (((T^[m])^[N]) u0) ≤ 2 * U :=
  variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_fixedPoint_of_bound_le_of_le_index
    hT m hfix hbound hBU hzero (k := N) (n := n) hNn

/--
Natural-bound companion of
`variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_of_fixedPoint_of_bound_le_of_le_index`.
-/
theorem variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_of_fixedPoint_of_bound_le_of_natBound
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (n N : ℕ) (hNn : N + 1 ≤ n) :
    variationSeminorm (((T^[m])^[N + 1]) u0) ≤ 2 * U :=
  variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_of_fixedPoint_of_bound_le_of_le_index
    hT m hfix hbound hBU hzero (k := N) (n := n) hNn

/--
Natural-bound companion of
`variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_fixedPoint_eq_zero_of_bound_le_of_budget_nonpos_of_le_index`.
-/
theorem
    variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_fixedPoint_eq_zero_of_bound_le_of_budget_nonpos_of_natBound
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hU_nonpos : U ≤ 0)
    (hzero : variationSeminorm u0 = 0)
    (n N : ℕ) (hNn : N ≤ n) :
    variationSeminorm (((T^[m])^[N]) u0) = 0 :=
  variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_fixedPoint_eq_zero_of_bound_le_of_budget_nonpos_of_le_index
    hT m hfix hbound hBU hU_nonpos hzero (k := N) (n := n) hNn

/--
Natural-bound companion of
`variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_of_fixedPoint_eq_zero_of_bound_le_of_budget_nonpos_of_le_index`.
-/
theorem
    variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_of_fixedPoint_eq_zero_of_bound_le_of_budget_nonpos_of_natBound
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) (m : ℕ)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hU_nonpos : U ≤ 0)
    (hzero : variationSeminorm u0 = 0)
    (n N : ℕ) (hNn : N + 1 ≤ n) :
    variationSeminorm (((T^[m])^[N + 1]) u0) = 0 :=
  variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_of_fixedPoint_eq_zero_of_bound_le_of_budget_nonpos_of_le_index
    hT m hfix hbound hBU hU_nonpos hzero (k := N) (n := n) hNn

/--
One-step stride specialization of the fixed-point iterate base-orbit budget-lifted bound.
-/
theorem variationSeminorm_orbitBound_with_base_of_isTopical_iterate_of_fixedPoint_of_bound_le_oneStep
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (k : ℕ) :
    variationSeminorm (((T^[1])^[k]) u0) ≤ variationSeminorm u0 + 2 * U := by
  simpa using
    variationSeminorm_orbitBound_with_base_of_isTopical_iterate_of_fixedPoint_of_bound_le
      hT 1 hfix hbound hBU k

/--
One-step stride successor-index specialization of the fixed-point iterate base-orbit
budget-lifted bound.
-/
theorem variationSeminorm_orbitBound_with_base_of_isTopical_iterate_succ_of_fixedPoint_of_bound_le_oneStep
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (k : ℕ) :
    variationSeminorm (((T^[1])^[k + 1]) u0) ≤ variationSeminorm u0 + 2 * U := by
  simpa using
    variationSeminorm_orbitBound_with_base_of_isTopical_iterate_succ_of_fixedPoint_of_bound_le
      hT 1 hfix hbound hBU k

/--
One-step stride specialization of the fixed-point iterate zero-seed orbit budget-lifted bound.
-/
theorem variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_fixedPoint_of_bound_le_oneStep
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm (((T^[1])^[k]) u0) ≤ 2 * U := by
  simpa using
    variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_fixedPoint_of_bound_le
      hT 1 hfix hbound hBU hzero k

/--
One-step stride successor-index specialization of the fixed-point iterate zero-seed orbit
budget-lifted bound.
-/
theorem variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_of_fixedPoint_of_bound_le_oneStep
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    {uStar u0 : ι → ℝ} (hfix : T uStar = uStar)
    {B U : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm (((T^[1])^[k + 1]) u0) ≤ 2 * U := by
  simpa using
    variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_of_fixedPoint_of_bound_le
      hT 1 hfix hbound hBU hzero k

end FiniteIndex

end KLProjection
end FlowSinkhorn
