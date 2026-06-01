import FlowSinkhorn.KLProjection.BlockQuotientVocabulary
import FlowSinkhorn.KLProjection.Variation
import Mathlib.Analysis.Seminorm

namespace FlowSinkhorn
namespace KLProjection

lemma abs_le_coordSupNorm {ι : Type*} [Fintype ι] [Nonempty ι] (x : ι → ℝ) (i : ι) :
    |x i| ≤ coordSupNorm x := by
  unfold coordSupNorm
  exact Finset.le_sup' (s := Finset.univ) (f := fun i => |x i|) (by simp)

lemma coordSupNorm_nonneg {ι : Type*} [Fintype ι] [Nonempty ι] (x : ι → ℝ) :
    0 ≤ coordSupNorm x := by
  let i : ι := Classical.choice inferInstance
  exact (abs_nonneg (x i)).trans (abs_le_coordSupNorm x i)

/-- `coordSupNorm 0 = 0`. -/
theorem coordSupNorm_zero {ι : Type*} [Fintype ι] [Nonempty ι] :
    coordSupNorm (0 : ι → ℝ) = 0 := by
  apply le_antisymm
  · unfold coordSupNorm
    apply Finset.sup'_le Finset.univ_nonempty
    intro i _
    simp
  · exact coordSupNorm_nonneg 0

/-- `coordSupNorm (fun _ => c) = |c|` for any constant `c`. -/
theorem coordSupNorm_const {ι : Type*} [Fintype ι] [Nonempty ι] (c : ℝ) :
    coordSupNorm (fun _ : ι => c) = |c| := by
  unfold coordSupNorm
  apply le_antisymm
  · apply Finset.sup'_le Finset.univ_nonempty
    intro i _
    simp
  · exact Finset.le_sup' (f := fun _ : ι => |c|) (Finset.mem_univ (Classical.choice inferInstance))

/--
Triangle inequality for the coordinatewise sup-norm.

`coordSupNorm (x + y) ≤ coordSupNorm x + coordSupNorm y`.
-/
theorem coordSupNorm_add_le {ι : Type*} [Fintype ι] [Nonempty ι] (x y : ι → ℝ) :
    coordSupNorm (x + y) ≤ coordSupNorm x + coordSupNorm y := by
  unfold coordSupNorm
  apply Finset.sup'_le Finset.univ_nonempty
  intro i _
  have hxy : |(x + y) i| ≤ |x i| + |y i| := by
    simp only [Pi.add_apply]
    rcases abs_cases (x i + y i) with ⟨h, _⟩ | ⟨h, _⟩ <;>
    rcases abs_cases (x i) with ⟨hx, _⟩ | ⟨hx', _⟩ <;>
    rcases abs_cases (y i) with ⟨hy, _⟩ | ⟨hy', _⟩ <;>
    linarith
  have hxi : |x i| ≤ Finset.univ.sup' Finset.univ_nonempty (fun j => |x j|) :=
    Finset.le_sup' (f := fun j => |x j|) (Finset.mem_univ i)
  have hyi : |y i| ≤ Finset.univ.sup' Finset.univ_nonempty (fun j => |y j|) :=
    Finset.le_sup' (f := fun j => |y j|) (Finset.mem_univ i)
  linarith

/--
The coordinatewise sup-norm of a difference is bounded by the sum of sup-norms.

`coordSupNorm (x - y) ≤ coordSupNorm x + coordSupNorm y`.
-/
theorem coordSupNorm_sub_le {ι : Type*} [Fintype ι] [Nonempty ι] (x y : ι → ℝ) :
    coordSupNorm (x - y) ≤ coordSupNorm x + coordSupNorm y := by
  have heq : x - y = x + (-y) := sub_eq_add_neg x y
  rw [heq]
  have hcn : coordSupNorm (-y) = coordSupNorm y := by
    unfold coordSupNorm; simp [abs_neg]
  linarith [coordSupNorm_add_le x (-y)]


/--
Scaling the coordinatewise sup-norm by a scalar.

`coordSupNorm (a • x) = |a| * coordSupNorm x`.
-/
theorem coordSupNorm_smul {ι : Type*} [Fintype ι] [Nonempty ι] (a : ℝ) (x : ι → ℝ) :
    coordSupNorm (a • x) = |a| * coordSupNorm x := by
  unfold coordSupNorm
  simp only [Pi.smul_apply, smul_eq_mul, abs_mul]
  apply le_antisymm
  · apply Finset.sup'_le Finset.univ_nonempty
    intro i _
    exact mul_le_mul_of_nonneg_left
      (Finset.le_sup' (f := fun j => |x j|) (Finset.mem_univ i))
      (abs_nonneg a)
  · obtain ⟨j, hj_mem, hj_max⟩ :=
      Finset.exists_mem_eq_sup' Finset.univ_nonempty (f := fun i : ι => |x i|)
    rw [hj_max]
    exact Finset.le_sup' (f := fun i => |a| * |x i|) hj_mem

/--
`coordSupNorm` as a Mathlib `Seminorm ℝ (ι → ℝ)`.

This registers `coordSupNorm` as a genuine Mathlib seminorm using `Seminorm.of`,
using the already-certified `coordSupNorm_add_le` (sub-additivity) and
`coordSupNorm_smul` (homogeneity) lemmas.
-/
noncomputable def coordSupNormAsSeminorm
    {ι : Type*} [Fintype ι] [Nonempty ι] : Seminorm ℝ (ι → ℝ) :=
  Seminorm.of coordSupNorm
    (fun x y => coordSupNorm_add_le x y)
    (fun a x => by
      rw [coordSupNorm_smul]
      simp [Real.norm_eq_abs])

noncomputable section BlockPairs

variable {ι₁ ι₂ : Type*}
variable [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]

lemma blockSupNorm_nonneg (u : BlockPair ι₁ ι₂) : 0 ≤ blockSupNorm u := by
  unfold blockSupNorm
  exact (coordSupNorm_nonneg u.1).trans (le_max_left _ _)

omit [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂] in
lemma pairedShift_zero (u : BlockPair ι₁ ι₂) : pairedShift 0 u = u := by
  ext i <;> simp [pairedShift]

omit [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂] in
lemma pairedShift_add (c d : ℝ) (u : BlockPair ι₁ ι₂) :
    pairedShift c (pairedShift d u) = pairedShift (c + d) u := by
  ext i <;> simp [pairedShift, sub_eq_add_neg, add_assoc, add_comm]

theorem pairedVariationLowerBound_pairedShift (u : BlockPair ι₁ ι₂) (c : ℝ) :
    pairedVariationLowerBound (pairedShift c u) = pairedVariationLowerBound u := by
  unfold pairedVariationLowerBound pairedShift
  simp [variationSeminorm_add_const, sub_eq_add_neg]

theorem pairedVariationLowerBound_le_blockSupNorm_pairedShift
    (u : BlockPair ι₁ ι₂) (c : ℝ) :
    pairedVariationLowerBound u ≤ blockSupNorm (pairedShift c u) := by
  let M : ℝ := blockSupNorm (pairedShift c u)
  have h₁abs : ∀ i, |u.1 i + c| ≤ M := by
    intro i
    have hcoord : |(pairedShift c u).1 i| ≤ coordSupNorm (pairedShift c u).1 :=
      abs_le_coordSupNorm (pairedShift c u).1 i
    have hblock : coordSupNorm (pairedShift c u).1 ≤ M := by
      unfold M blockSupNorm
      exact le_max_left _ _
    simpa [pairedShift] using hcoord.trans hblock
  have h₂abs : ∀ j, |u.2 j + (-c)| ≤ M := by
    intro j
    have hcoord : |(pairedShift c u).2 j| ≤ coordSupNorm (pairedShift c u).2 :=
      abs_le_coordSupNorm (pairedShift c u).2 j
    have hblock : coordSupNorm (pairedShift c u).2 ≤ M := by
      unfold M blockSupNorm
      exact le_max_right _ _
    simpa [pairedShift, sub_eq_add_neg] using hcoord.trans hblock
  have h₁ : variationSeminorm u.1 ≤ M := by
    exact variationSeminorm_le_of_forall_abs_add_const_le u.1 (c := c) (M := M) h₁abs
  have h₂ : variationSeminorm u.2 ≤ M := by
    exact variationSeminorm_le_of_forall_abs_add_const_le u.2 (c := -c) (M := M) h₂abs
  unfold pairedVariationLowerBound
  exact max_le h₁ h₂

lemma pairedShiftNormSet_nonempty (u : BlockPair ι₁ ι₂) :
    (pairedShiftNormSet u).Nonempty :=
  Set.range_nonempty _

lemma pairedShiftNormSet_bddBelow (u : BlockPair ι₁ ι₂) :
    BddBelow (pairedShiftNormSet u) := by
  refine ⟨0, ?_⟩
  rintro r ⟨c, rfl⟩
  exact blockSupNorm_nonneg (pairedShift c u)

theorem pairedShiftNormSet_pairedShift (u : BlockPair ι₁ ι₂) (d : ℝ) :
    pairedShiftNormSet (pairedShift d u) = pairedShiftNormSet u := by
  ext r
  constructor
  · rintro ⟨c, rfl⟩
    refine ⟨c + d, ?_⟩
    simp [pairedShift_add]
  · rintro ⟨c, rfl⟩
    refine ⟨c - d, ?_⟩
    simp [pairedShift_add, sub_eq_add_neg, add_comm]

/-- The infimum-based paired quotient seminorm is invariant under the paired shift action. -/
theorem pairedQuotientSupSeminorm_pairedShift (u : BlockPair ι₁ ι₂) (d : ℝ) :
    pairedQuotientSupSeminorm (pairedShift d u) = pairedQuotientSupSeminorm u := by
  simp [pairedQuotientSupSeminorm, pairedShiftNormSet_pairedShift]

/--
The paired variation lower bound is bounded by the infimum-based paired quotient seminorm.

This packages the pointwise paired-shift estimate into a genuine quotient-level corollary.
-/
theorem pairedVariationLowerBound_le_pairedQuotientSupSeminorm
    (u : BlockPair ι₁ ι₂) :
    pairedVariationLowerBound u ≤ pairedQuotientSupSeminorm u := by
  change pairedVariationLowerBound u ≤ sInf (pairedShiftNormSet u)
  exact le_csInf (pairedShiftNormSet_nonempty u) (by
    intro r hr
    rcases hr with ⟨c, rfl⟩
    exact pairedVariationLowerBound_le_blockSupNorm_pairedShift u c)

theorem pairedQuotientSupSeminorm_nonneg (u : BlockPair ι₁ ι₂) :
    0 ≤ pairedQuotientSupSeminorm u := by
  change 0 ≤ sInf (pairedShiftNormSet u)
  exact le_csInf (pairedShiftNormSet_nonempty u) (by
    intro r hr
    rcases hr with ⟨c, rfl⟩
    exact blockSupNorm_nonneg (pairedShift c u))

/--
Direct sup-control of the variation lower bound (choose zero paired shift).
-/
theorem pairedVariationLowerBound_le_blockSupNorm
    (u : BlockPair ι₁ ι₂) :
    pairedVariationLowerBound u ≤ blockSupNorm u := by
  simpa [pairedShift] using pairedVariationLowerBound_le_blockSupNorm_pairedShift u 0

/--
The paired quotient seminorm is controlled by the unshifted block sup norm.

This is the infimum upper bound obtained by evaluating the orbit at shift `c = 0`.
-/
theorem pairedQuotientSupSeminorm_le_blockSupNorm
    (u : BlockPair ι₁ ι₂) :
    pairedQuotientSupSeminorm u ≤ blockSupNorm u := by
  change sInf (pairedShiftNormSet u) ≤ blockSupNorm u
  apply csInf_le (pairedShiftNormSet_bddBelow u)
  exact ⟨0, by simp [pairedShift]⟩

/--
Two-sided squeeze for the paired quotient seminorm:
`pairedVariationLowerBound ≤ pairedQuotientSupSeminorm ≤ blockSupNorm`.
-/
theorem pairedQuotientSupSeminorm_sandwich
    (u : BlockPair ι₁ ι₂) :
    pairedVariationLowerBound u ≤ pairedQuotientSupSeminorm u ∧
      pairedQuotientSupSeminorm u ≤ blockSupNorm u := by
  exact ⟨pairedVariationLowerBound_le_pairedQuotientSupSeminorm u,
    pairedQuotientSupSeminorm_le_blockSupNorm u⟩

/--
Per-block variation control extracted from the paired quotient seminorm lower bound.

This bridge unpacks the `max`-based lower bound into the two coordinatewise variation estimates.
-/
theorem pairedQuotientSupSeminorm_controls_blockVariations
    (u : BlockPair ι₁ ι₂) :
    variationSeminorm u.1 ≤ pairedQuotientSupSeminorm u ∧
      variationSeminorm u.2 ≤ pairedQuotientSupSeminorm u := by
  have hlow : pairedVariationLowerBound u ≤ pairedQuotientSupSeminorm u :=
    pairedVariationLowerBound_le_pairedQuotientSupSeminorm u
  constructor
  · exact (le_max_left (variationSeminorm u.1) (variationSeminorm u.2)).trans hlow
  · exact (le_max_right (variationSeminorm u.1) (variationSeminorm u.2)).trans hlow

/--
First-block projection of `pairedQuotientSupSeminorm_controls_blockVariations`.
-/
theorem pairedQuotientSupSeminorm_controls_blockVariation_left
    (u : BlockPair ι₁ ι₂) :
    variationSeminorm u.1 ≤ pairedQuotientSupSeminorm u :=
  (pairedQuotientSupSeminorm_controls_blockVariations (u := u)).1

/--
Second-block projection of `pairedQuotientSupSeminorm_controls_blockVariations`.
-/
theorem pairedQuotientSupSeminorm_controls_blockVariation_right
    (u : BlockPair ι₁ ι₂) :
    variationSeminorm u.2 ≤ pairedQuotientSupSeminorm u :=
  (pairedQuotientSupSeminorm_controls_blockVariations (u := u)).2

/--
Shifted-input convenience form of the quotient lower bound.
-/
theorem pairedVariationLowerBound_le_pairedQuotientSupSeminorm_pairedShift
    (u : BlockPair ι₁ ι₂) (d : ℝ) :
    pairedVariationLowerBound (pairedShift d u) ≤ pairedQuotientSupSeminorm u := by
  simpa [pairedVariationLowerBound_pairedShift, pairedQuotientSupSeminorm_pairedShift] using
    (pairedVariationLowerBound_le_pairedQuotientSupSeminorm (u := u))

/--
Shifted-input convenience form of per-block variation control by the paired quotient seminorm.
-/
theorem pairedQuotientSupSeminorm_controls_blockVariations_pairedShift
    (u : BlockPair ι₁ ι₂) (d : ℝ) :
    variationSeminorm (pairedShift d u).1 ≤ pairedQuotientSupSeminorm u ∧
      variationSeminorm (pairedShift d u).2 ≤ pairedQuotientSupSeminorm u := by
  have h :=
    pairedQuotientSupSeminorm_controls_blockVariations (u := pairedShift d u)
  constructor
  · simpa [pairedQuotientSupSeminorm_pairedShift] using h.1
  · simpa [pairedQuotientSupSeminorm_pairedShift] using h.2

/--
First-block projection of
`pairedQuotientSupSeminorm_controls_blockVariations_pairedShift`.
-/
theorem pairedQuotientSupSeminorm_controls_blockVariation_left_pairedShift
    (u : BlockPair ι₁ ι₂) (d : ℝ) :
    variationSeminorm (pairedShift d u).1 ≤ pairedQuotientSupSeminorm u :=
  (pairedQuotientSupSeminorm_controls_blockVariations_pairedShift (u := u) (d := d)).1

/--
Second-block projection of
`pairedQuotientSupSeminorm_controls_blockVariations_pairedShift`.
-/
theorem pairedQuotientSupSeminorm_controls_blockVariation_right_pairedShift
    (u : BlockPair ι₁ ι₂) (d : ℝ) :
    variationSeminorm (pairedShift d u).2 ≤ pairedQuotientSupSeminorm u :=
  (pairedQuotientSupSeminorm_controls_blockVariations_pairedShift (u := u) (d := d)).2

omit [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂] in
@[simp] theorem signedPairedShift_minus_eq_pairedShift (c : ℝ) (u : BlockPair ι₁ ι₂) :
    signedPairedShift .minus c u = pairedShift c u := by
  ext i <;> simp [signedPairedShift, pairedShift, PairedSign.toReal, sub_eq_add_neg]

omit [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂] in
lemma signedPairedShift_zero (τ : PairedSign) (u : BlockPair ι₁ ι₂) :
    signedPairedShift τ 0 u = u := by
  ext i <;> simp [signedPairedShift, PairedSign.toReal]

omit [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂] in
lemma signedPairedShift_add (τ : PairedSign) (c d : ℝ) (u : BlockPair ι₁ ι₂) :
    signedPairedShift τ c (signedPairedShift τ d u) = signedPairedShift τ (c + d) u := by
  ext i <;> simp [signedPairedShift, PairedSign.toReal, mul_add, add_assoc, add_left_comm, add_comm]

/--
For `τ = minus`, the signed-shift orbit set coincides with the classical paired-shift orbit set.
-/
theorem signedPairedShiftNormSet_minus_eq_pairedShiftNormSet
    (u : BlockPair ι₁ ι₂) :
    signedPairedShiftNormSet .minus u = pairedShiftNormSet u := by
  ext r
  constructor
  · rintro ⟨c, rfl⟩
    refine ⟨c, ?_⟩
    simp [signedPairedShift_minus_eq_pairedShift]
  · rintro ⟨c, rfl⟩
    refine ⟨c, ?_⟩
    simp [signedPairedShift_minus_eq_pairedShift]

/--
For `τ = minus`, the signed quotient seminorm equals the classical paired quotient seminorm.
-/
theorem signedPairedQuotientSupSeminorm_minus_eq_pairedQuotientSupSeminorm
    (u : BlockPair ι₁ ι₂) :
    signedPairedQuotientSupSeminorm .minus u = pairedQuotientSupSeminorm u := by
  simp [signedPairedQuotientSupSeminorm, pairedQuotientSupSeminorm,
    signedPairedShiftNormSet_minus_eq_pairedShiftNormSet]

theorem pairedVariationLowerBound_signedPairedShift
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) (c : ℝ) :
    pairedVariationLowerBound (signedPairedShift τ c u) = pairedVariationLowerBound u := by
  unfold pairedVariationLowerBound signedPairedShift
  cases τ <;> simp [PairedSign.toReal, variationSeminorm_add_const]

theorem pairedVariationLowerBound_le_blockSupNorm_signedPairedShift
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) (c : ℝ) :
    pairedVariationLowerBound u ≤ blockSupNorm (signedPairedShift τ c u) := by
  let M : ℝ := blockSupNorm (signedPairedShift τ c u)
  have h₁abs : ∀ i, |u.1 i + c| ≤ M := by
    intro i
    have hcoord : |(signedPairedShift τ c u).1 i| ≤ coordSupNorm (signedPairedShift τ c u).1 :=
      abs_le_coordSupNorm (signedPairedShift τ c u).1 i
    have hblock : coordSupNorm (signedPairedShift τ c u).1 ≤ M := by
      unfold M blockSupNorm
      exact le_max_left _ _
    simpa [signedPairedShift] using hcoord.trans hblock
  have h₂abs : ∀ j, |u.2 j + τ.toReal * c| ≤ M := by
    intro j
    have hcoord : |(signedPairedShift τ c u).2 j| ≤ coordSupNorm (signedPairedShift τ c u).2 :=
      abs_le_coordSupNorm (signedPairedShift τ c u).2 j
    have hblock : coordSupNorm (signedPairedShift τ c u).2 ≤ M := by
      unfold M blockSupNorm
      exact le_max_right _ _
    simpa [signedPairedShift] using hcoord.trans hblock
  have h₁ : variationSeminorm u.1 ≤ M := by
    exact variationSeminorm_le_of_forall_abs_add_const_le u.1 (c := c) (M := M) h₁abs
  have h₂ : variationSeminorm u.2 ≤ M := by
    exact variationSeminorm_le_of_forall_abs_add_const_le u.2 (c := τ.toReal * c) (M := M) h₂abs
  unfold pairedVariationLowerBound
  exact max_le h₁ h₂

lemma signedPairedShiftNormSet_nonempty (τ : PairedSign) (u : BlockPair ι₁ ι₂) :
    (signedPairedShiftNormSet τ u).Nonempty :=
  Set.range_nonempty _

theorem signedPairedShiftNormSet_pairedShift
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) (d : ℝ) :
    signedPairedShiftNormSet τ (signedPairedShift τ d u) = signedPairedShiftNormSet τ u := by
  ext r
  constructor
  · rintro ⟨c, rfl⟩
    refine ⟨c + d, ?_⟩
    simp [signedPairedShift_add]
  · rintro ⟨c, rfl⟩
    refine ⟨c - d, ?_⟩
    simp [signedPairedShift_add, sub_eq_add_neg, add_comm]

/-- The signed quotient seminorm is invariant under the signed paired-balance action. -/
theorem signedPairedQuotientSupSeminorm_pairedShift
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) (d : ℝ) :
    signedPairedQuotientSupSeminorm τ (signedPairedShift τ d u) =
      signedPairedQuotientSupSeminorm τ u := by
  simp [signedPairedQuotientSupSeminorm, signedPairedShiftNormSet_pairedShift]

/--
The paired variation lower bound also bounds the signed quotient seminorm from below.
-/
theorem pairedVariationLowerBound_le_signedPairedQuotientSupSeminorm
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) :
    pairedVariationLowerBound u ≤ signedPairedQuotientSupSeminorm τ u := by
  change pairedVariationLowerBound u ≤ sInf (signedPairedShiftNormSet τ u)
  exact le_csInf (signedPairedShiftNormSet_nonempty τ u) (by
    intro r hr
    rcases hr with ⟨c, rfl⟩
    exact pairedVariationLowerBound_le_blockSupNorm_signedPairedShift τ u c)

theorem signedPairedQuotientSupSeminorm_nonneg
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) :
    0 ≤ signedPairedQuotientSupSeminorm τ u := by
  change 0 ≤ sInf (signedPairedShiftNormSet τ u)
  exact le_csInf (signedPairedShiftNormSet_nonempty τ u) (by
    intro r hr
    rcases hr with ⟨c, rfl⟩
    exact blockSupNorm_nonneg (signedPairedShift τ c u))

/--
The signed paired quotient seminorm is controlled by the unshifted block sup norm.

As for the unsigned case, this follows by evaluating the orbit at shift `c = 0`.
-/
theorem signedPairedQuotientSupSeminorm_le_blockSupNorm
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) :
    signedPairedQuotientSupSeminorm τ u ≤ blockSupNorm u := by
  change sInf (signedPairedShiftNormSet τ u) ≤ blockSupNorm u
  have hbdd : BddBelow (signedPairedShiftNormSet τ u) := by
    refine ⟨0, ?_⟩
    rintro r ⟨c, rfl⟩
    exact blockSupNorm_nonneg (signedPairedShift τ c u)
  apply csInf_le hbdd
  exact ⟨0, by simp [signedPairedShift, PairedSign.toReal]⟩

/--
Two-sided squeeze for the signed quotient seminorm:
`pairedVariationLowerBound ≤ signedPairedQuotientSupSeminorm ≤ blockSupNorm`.
-/
theorem signedPairedQuotientSupSeminorm_sandwich
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) :
    pairedVariationLowerBound u ≤ signedPairedQuotientSupSeminorm τ u ∧
      signedPairedQuotientSupSeminorm τ u ≤ blockSupNorm u := by
  exact ⟨pairedVariationLowerBound_le_signedPairedQuotientSupSeminorm τ u,
    signedPairedQuotientSupSeminorm_le_blockSupNorm τ u⟩

/--
Per-block variation control extracted from the signed quotient seminorm lower bound.
-/
theorem signedPairedQuotientSupSeminorm_controls_blockVariations
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) :
    variationSeminorm u.1 ≤ signedPairedQuotientSupSeminorm τ u ∧
      variationSeminorm u.2 ≤ signedPairedQuotientSupSeminorm τ u := by
  have hlow : pairedVariationLowerBound u ≤ signedPairedQuotientSupSeminorm τ u :=
    pairedVariationLowerBound_le_signedPairedQuotientSupSeminorm τ u
  constructor
  · exact (le_max_left (variationSeminorm u.1) (variationSeminorm u.2)).trans hlow
  · exact (le_max_right (variationSeminorm u.1) (variationSeminorm u.2)).trans hlow

/--
First-block projection of `signedPairedQuotientSupSeminorm_controls_blockVariations`.
-/
theorem signedPairedQuotientSupSeminorm_controls_blockVariation_left
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) :
    variationSeminorm u.1 ≤ signedPairedQuotientSupSeminorm τ u :=
  (signedPairedQuotientSupSeminorm_controls_blockVariations (τ := τ) (u := u)).1

/--
Second-block projection of `signedPairedQuotientSupSeminorm_controls_blockVariations`.
-/
theorem signedPairedQuotientSupSeminorm_controls_blockVariation_right
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) :
    variationSeminorm u.2 ≤ signedPairedQuotientSupSeminorm τ u :=
  (signedPairedQuotientSupSeminorm_controls_blockVariations (τ := τ) (u := u)).2

/--
Minus-sign bridge from signed control to the classical paired quotient seminorm.
-/
theorem signedPairedQuotientSupSeminorm_controls_blockVariations_minus
    (u : BlockPair ι₁ ι₂) :
    variationSeminorm u.1 ≤ pairedQuotientSupSeminorm u ∧
      variationSeminorm u.2 ≤ pairedQuotientSupSeminorm u := by
  simpa [signedPairedQuotientSupSeminorm_minus_eq_pairedQuotientSupSeminorm] using
    (signedPairedQuotientSupSeminorm_controls_blockVariations (τ := .minus) (u := u))

/--
Shifted-input convenience form of the signed quotient lower bound.
-/
theorem pairedVariationLowerBound_le_signedPairedQuotientSupSeminorm_signedPairedShift
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) (d : ℝ) :
    pairedVariationLowerBound (signedPairedShift τ d u) ≤
      signedPairedQuotientSupSeminorm τ u := by
  simpa [pairedVariationLowerBound_signedPairedShift,
    signedPairedQuotientSupSeminorm_pairedShift] using
    (pairedVariationLowerBound_le_signedPairedQuotientSupSeminorm (τ := τ) (u := u))

/--
Shifted-input convenience form of per-block variation control by the signed quotient seminorm.
-/
theorem signedPairedQuotientSupSeminorm_controls_blockVariations_signedPairedShift
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) (d : ℝ) :
    variationSeminorm (signedPairedShift τ d u).1 ≤
        signedPairedQuotientSupSeminorm τ u ∧
      variationSeminorm (signedPairedShift τ d u).2 ≤
        signedPairedQuotientSupSeminorm τ u := by
  have h :=
    signedPairedQuotientSupSeminorm_controls_blockVariations
      (τ := τ) (u := signedPairedShift τ d u)
  constructor
  · simpa [signedPairedQuotientSupSeminorm_pairedShift] using h.1
  · simpa [signedPairedQuotientSupSeminorm_pairedShift] using h.2

/--
First-block projection of
`signedPairedQuotientSupSeminorm_controls_blockVariations_signedPairedShift`.
-/
theorem signedPairedQuotientSupSeminorm_controls_blockVariation_left_signedPairedShift
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) (d : ℝ) :
    variationSeminorm (signedPairedShift τ d u).1 ≤
      signedPairedQuotientSupSeminorm τ u :=
  (signedPairedQuotientSupSeminorm_controls_blockVariations_signedPairedShift
    (τ := τ) (u := u) (d := d)).1

/--
Second-block projection of
`signedPairedQuotientSupSeminorm_controls_blockVariations_signedPairedShift`.
-/
theorem signedPairedQuotientSupSeminorm_controls_blockVariation_right_signedPairedShift
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) (d : ℝ) :
    variationSeminorm (signedPairedShift τ d u).2 ≤
      signedPairedQuotientSupSeminorm τ u :=
  (signedPairedQuotientSupSeminorm_controls_blockVariations_signedPairedShift
    (τ := τ) (u := u) (d := d)).2

end BlockPairs
end KLProjection
end FlowSinkhorn
