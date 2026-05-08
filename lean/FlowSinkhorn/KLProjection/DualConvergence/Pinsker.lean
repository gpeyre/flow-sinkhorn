import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Probability.Moments.SubGaussian
import Mathlib.Tactic
import FlowSinkhorn.KLProjection.DualConvergence.Variational

/-!
# Non-normalized Pinsker reduction

This module formalizes the finite sign-test and scaling steps used in Appendix A
to pass from the normalized variational/MGF Pinsker proof to the non-normalized
equal-mass form.

The strongest paper-facing endpoint in this file is
`pinsker_nonnormalized_of_finite_variational_centeredBernoulli_hoeffding_massShell`: it assumes
the finite all-test KL variational inequality and the centered Bernoulli scalar Hoeffding
inequality, then certifies the sign-test specialization, sign-to-`ℓ¹` identity, sign-mass MGF
decomposition, reduction to the centered Bernoulli form, probability-mass normalization,
logarithmic MGF conversion, quadratic optimization, and non-normalized scaling internally.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace DualConvergence

open MeasureTheory
open scoped BigOperators

variable {n : ℕ}

/-- Finite-dimensional KL expression used in the appendix argument. -/
noncomputable def finiteKL (p q : Fin n → ℝ) : ℝ :=
  ∑ i, p i * Real.log (p i / q i)

/-- Finite-dimensional `ℓ¹` norm. -/
noncomputable def l1Norm (v : Fin n → ℝ) : ℝ :=
  ∑ i, |v i|

/-- Deterministic sign selector used by the finite Pinsker test function. -/
noncomputable def finiteSign (v : Fin n → ℝ) : Fin n → ℝ :=
  fun i => if 0 ≤ v i then 1 else -1

/--
Scaling identity for the finite KL expression on a common positive mass shell.
-/
theorem finiteKL_scale
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M) :
    finiteKL p q = M * finiteKL (fun i => p i / M) (fun i => q i / M) := by
  have hM0 : M ≠ 0 := ne_of_gt hMpos
  unfold finiteKL
  calc
    (∑ i, p i * Real.log (p i / q i))
        = ∑ i, (M * (p i / M)) * Real.log (((M * (p i / M)) / (M * (q i / M)))) := by
            refine Finset.sum_congr rfl ?_
            intro i hi
            have hp : M * (p i / M) = p i := by field_simp [hM0]
            have hq : M * (q i / M) = q i := by field_simp [hM0]
            rw [hp, hq]
    _ = ∑ i, (M * (p i / M)) * Real.log ((p i / M) / (q i / M)) := by
          refine Finset.sum_congr rfl ?_
          intro i hi
          congr 1
          field_simp [hM0]
    _ = ∑ i, M * ((p i / M) * Real.log ((p i / M) / (q i / M))) := by
          refine Finset.sum_congr rfl ?_
          intro i hi
          ring
    _ = M * ∑ i, ((p i / M) * Real.log ((p i / M) / (q i / M))) := by
          rw [Finset.mul_sum]
    _ = M * finiteKL (fun i => p i / M) (fun i => q i / M) := by
          rfl

/-- Normalized finite KL is the common-mass finite KL divided by the mass. -/
theorem finiteKL_normalized_eq_div
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M) :
    finiteKL (fun i => p i / M) (fun i => q i / M) = finiteKL p q / M := by
  have hM0 : M ≠ 0 := ne_of_gt hMpos
  have hscale : finiteKL p q = M * finiteKL (fun i => p i / M) (fun i => q i / M) :=
    finiteKL_scale (p := p) (q := q) hMpos
  calc
    finiteKL (fun i => p i / M) (fun i => q i / M)
        = (M * finiteKL (fun i => p i / M) (fun i => q i / M)) / M := by
            field_simp [hM0]
    _ = finiteKL p q / M := by
            rw [← hscale]

/--
Scaling identity for the finite-dimensional `ℓ¹` norm on a common positive mass shell.
-/
theorem l1Norm_sub_scale
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M) :
    l1Norm (fun i => p i - q i) = M * l1Norm (fun i => p i / M - q i / M) := by
  have hMnonneg : 0 ≤ M := le_of_lt hMpos
  unfold l1Norm
  calc
    (∑ i, |p i - q i|)
        = ∑ i, |M * (p i / M - q i / M)| := by
            refine Finset.sum_congr rfl ?_
            intro i hi
            have hscale : p i - q i = M * (p i / M - q i / M) := by
              field_simp [ne_of_gt hMpos]
            rw [hscale]
    _ = ∑ i, M * |p i / M - q i / M| := by
          refine Finset.sum_congr rfl ?_
          intro i hi
          rw [abs_mul, abs_of_nonneg hMnonneg]
    _ = M * ∑ i, |p i / M - q i / M| := by
          rw [Finset.mul_sum]
    _ = M * l1Norm (fun i => p i / M - q i / M) := by
          rfl

/-- Normalized `ℓ¹` distance is the common-mass `ℓ¹` distance divided by the mass. -/
theorem l1Norm_normalized_sub_eq_div
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M) :
    l1Norm (fun i => p i / M - q i / M)
      = l1Norm (fun i => p i - q i) / M := by
  have hM0 : M ≠ 0 := ne_of_gt hMpos
  have hscale : l1Norm (fun i => p i - q i)
      = M * l1Norm (fun i => p i / M - q i / M) :=
    l1Norm_sub_scale (p := p) (q := q) hMpos
  calc
    l1Norm (fun i => p i / M - q i / M)
        = (M * l1Norm (fun i => p i / M - q i / M)) / M := by
            field_simp [hM0]
    _ = l1Norm (fun i => p i - q i) / M := by
            rw [← hscale]

/-- The finite-dimensional `ℓ¹` norm is nonnegative. -/
theorem l1Norm_nonneg (v : Fin n → ℝ) :
    0 ≤ l1Norm v := by
  unfold l1Norm
  exact Finset.sum_nonneg fun i _ => abs_nonneg (v i)

/-- Pointwise sign identity behind the finite-dimensional `ℓ¹` test function. -/
theorem mul_finiteSign_eq_abs (x : ℝ) :
    x * (if 0 ≤ x then 1 else -1) = |x| := by
  by_cases hx : 0 ≤ x
  · simp [hx, abs_of_nonneg hx]
  · have hxlt : x < 0 := lt_of_not_ge hx
    simp [hx, abs_of_neg hxlt]

/-- The finite sign test is bounded by one, the input needed for Hoeffding's lemma. -/
theorem abs_finiteSign_le_one (v : Fin n → ℝ) (i : Fin n) :
    |finiteSign v i| ≤ 1 := by
  unfold finiteSign
  by_cases hx : 0 ≤ v i
  · simp [hx]
  · simp [hx]

/--
The sign test realizes the finite-dimensional `ℓ¹` norm as a linear functional.

This is the discrete step used in the self-contained Pinsker proof: choosing
`s_i = sign(p_i-q_i)` turns the variational inequality into an `ℓ¹` bound.
-/
theorem finiteSign_l1_identity (v : Fin n → ℝ) :
    ∑ i, v i * finiteSign v i = l1Norm v := by
  unfold finiteSign l1Norm
  refine Finset.sum_congr rfl ?_
  intro i hi
  exact mul_finiteSign_eq_abs (v i)

/-- The finite sign selector only takes the two values `1` and `-1`. -/
theorem finiteSign_eq_one_or_neg_one (v : Fin n → ℝ) (i : Fin n) :
    finiteSign v i = 1 ∨ finiteSign v i = -1 := by
  unfold finiteSign
  by_cases h : 0 ≤ v i
  · simp [h]
  · simp [h]

/-- Positive-sign mass for the two-point decomposition of the finite sign test. -/
noncomputable def finiteSignPosMass (q v : Fin n → ℝ) : ℝ :=
  ∑ i, if finiteSign v i = 1 then q i else 0

/-- Negative-sign mass for the two-point decomposition of the finite sign test. -/
noncomputable def finiteSignNegMass (q v : Fin n → ℝ) : ℝ :=
  ∑ i, if finiteSign v i = -1 then q i else 0

/-- Nonnegativity of the positive-sign mass. -/
theorem finiteSignPosMass_nonneg
    {q v : Fin n → ℝ}
    (hq_nonneg : ∀ i, 0 ≤ q i) :
    0 ≤ finiteSignPosMass q v := by
  unfold finiteSignPosMass
  exact Finset.sum_nonneg fun i _ => by
    by_cases h : finiteSign v i = 1
    · simp [h, hq_nonneg i]
    · simp [h]

/-- Nonnegativity of the negative-sign mass. -/
theorem finiteSignNegMass_nonneg
    {q v : Fin n → ℝ}
    (hq_nonneg : ∀ i, 0 ≤ q i) :
    0 ≤ finiteSignNegMass q v := by
  unfold finiteSignNegMass
  exact Finset.sum_nonneg fun i _ => by
    by_cases h : finiteSign v i = -1
    · simp [h, hq_nonneg i]
    · simp [h]

/--
The positive and negative sign masses add to the total mass.

This is the finite partition step used to reduce the sign-test Hoeffding bound
to a scalar two-point inequality.
-/
theorem finiteSignPosMass_add_negMass
    (q v : Fin n → ℝ) :
    finiteSignPosMass q v + finiteSignNegMass q v = ∑ i, q i := by
  unfold finiteSignPosMass finiteSignNegMass
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl ?_
  intro i hi
  rcases finiteSign_eq_one_or_neg_one v i with hsign | hsign
  · have hone : (1 : ℝ) ≠ -1 := by norm_num
    simp [hsign, hone]
  · have hneg : (-1 : ℝ) ≠ 1 := by norm_num
    simp [hsign, hneg]

/-- The sign-test mean is the difference of the two sign masses. -/
theorem finiteSign_mean_eq_posMass_sub_negMass
    (q v : Fin n → ℝ) :
    ∑ i, q i * finiteSign v i =
      finiteSignPosMass q v - finiteSignNegMass q v := by
  unfold finiteSignPosMass finiteSignNegMass
  rw [← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl ?_
  intro i hi
  rcases finiteSign_eq_one_or_neg_one v i with hsign | hsign
  · have hone : (1 : ℝ) ≠ -1 := by norm_num
    simp [hsign, hone]
  · have hneg : (-1 : ℝ) ≠ 1 := by norm_num
    simp [hsign, hneg]

/-- The sign-test exponential moment is a two-point exponential mixture. -/
theorem finiteSign_exp_sum_eq_twoPoint
    (q v : Fin n → ℝ) (t : ℝ) :
    ∑ i, q i * Real.exp (t * finiteSign v i) =
      finiteSignPosMass q v * Real.exp t
        + finiteSignNegMass q v * Real.exp (-t) := by
  unfold finiteSignPosMass finiteSignNegMass
  rw [Finset.sum_mul, Finset.sum_mul, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl ?_
  intro i hi
  rcases finiteSign_eq_one_or_neg_one v i with hsign | hsign
  · have hone : (1 : ℝ) ≠ -1 := by norm_num
    simp [hsign, hone]
  · have hneg : (-1 : ℝ) ≠ 1 := by norm_num
    simp [hsign, hneg]

/--
A positive finite probability vector gives a positive exponential moment for every test.

This is the small positivity side condition needed to move from the standard exponential
Hoeffding bound to its logarithmic cumulant-generating-function form.
-/
theorem weighted_exp_sum_pos
    {q s : Fin n → ℝ}
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = 1)
    (t : ℝ) :
    0 < ∑ i, q i * Real.exp (t * s i) := by
  have hq_sum_pos : 0 < ∑ i, q i := by
    rw [hq_mass]
    norm_num
  have hq_exists : ∃ i ∈ (Finset.univ : Finset (Fin n)), 0 < q i := by
    exact (Finset.sum_pos_iff_of_nonneg
      (s := (Finset.univ : Finset (Fin n))) (f := q)
      (by intro i hi; exact hq_nonneg i)).mp hq_sum_pos
  refine Finset.sum_pos'
    (fun i hi => mul_nonneg (hq_nonneg i) (le_of_lt (Real.exp_pos _))) ?_
  rcases hq_exists with ⟨i, hi, hqi⟩
  exact ⟨i, hi, mul_pos hqi (Real.exp_pos _)⟩

/-- Nonnegativity of normalized finite weights on a positive mass shell. -/
theorem normalizedWeight_nonneg
    {q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ i, 0 ≤ q i) :
    ∀ i, 0 ≤ q i / M := by
  intro i
  exact div_nonneg (hq_nonneg i) (le_of_lt hMpos)

/-- Normalized finite weights have total mass one on a positive mass shell. -/
theorem normalizedWeight_mass
    {q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq_mass : ∑ i, q i = M) :
    ∑ i, q i / M = 1 := by
  have hM0 : M ≠ 0 := ne_of_gt hMpos
  have hsum : (∑ i, q i / M) = (∑ i, q i) / M := by
    simp [div_eq_mul_inv, Finset.sum_mul]
  rw [hsum, hq_mass]
  field_simp [hM0]

/-- Strict positivity of normalized finite weights on a positive mass shell. -/
theorem normalizedWeight_pos
    {q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq_pos : ∀ i, 0 < q i) :
    ∀ i, 0 < q i / M := by
  intro i
  exact div_pos (hq_pos i) hMpos

/-- Nonnegativity of normalized finite weights from strict positivity. -/
theorem normalizedWeight_nonneg_of_pos
    {q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq_pos : ∀ i, 0 < q i) :
    ∀ i, 0 ≤ q i / M := by
  intro i
  exact (normalizedWeight_pos (q := q) hMpos hq_pos i).le

/--
Logarithmic MGF/Hoeffding form from the standard exponential MGF form.

The paper writes the bound in logarithmic form.  Many analytic references state
Hoeffding as an exponential-moment inequality; this lemma certifies the conversion.
-/
theorem log_mgf_bound_of_exp_mgf_bound
    {q s : Fin n → ℝ}
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = 1)
    (hexp :
      ∀ t : ℝ, 0 ≤ t →
        ∑ i, q i * Real.exp (t * s i)
          ≤ Real.exp (t * (∑ i, q i * s i) + t ^ 2 / 2)) :
    ∀ t : ℝ, 0 ≤ t →
      Real.log (∑ i, q i * Real.exp (t * s i))
        ≤ t * (∑ i, q i * s i) + t ^ 2 / 2 := by
  intro t ht
  have hpos : 0 < ∑ i, q i * Real.exp (t * s i) :=
    weighted_exp_sum_pos (q := q) (s := s) hq_nonneg hq_mass t
  have hlog :=
    Real.log_le_log hpos (hexp t ht)
  simpa [Real.log_exp] using hlog

/-- Positivity of normalized common-mass exponential moments. -/
theorem weighted_exp_sum_pos_massShell
    {q s : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = M)
    (t : ℝ) :
    0 < ∑ i, (q i / M) * Real.exp (t * s i) := by
  exact weighted_exp_sum_pos
    (q := fun i => q i / M) (s := s)
    (normalizedWeight_nonneg (q := q) hMpos hq_nonneg)
    (normalizedWeight_mass (q := q) hMpos hq_mass)
    t

/--
Mass-shell logarithmic MGF/Hoeffding conversion with normalization handled internally.
-/
theorem log_mgf_bound_of_exp_mgf_bound_massShell
    {q s : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = M)
    (hexp :
      ∀ t : ℝ, 0 ≤ t →
        ∑ i, (q i / M) * Real.exp (t * s i)
          ≤ Real.exp (t * (∑ i, (q i / M) * s i) + t ^ 2 / 2)) :
    ∀ t : ℝ, 0 ≤ t →
      Real.log (∑ i, (q i / M) * Real.exp (t * s i))
        ≤ t * (∑ i, (q i / M) * s i) + t ^ 2 / 2 := by
  exact log_mgf_bound_of_exp_mgf_bound
    (q := fun i => q i / M) (s := s)
    (normalizedWeight_nonneg (q := q) hMpos hq_nonneg)
    (normalizedWeight_mass (q := q) hMpos hq_mass)
    hexp

/--
Mathlib-backed Hoeffding bridge for centered variables bounded in `[-1,1]`.

The paper-facing finite Pinsker endpoint currently reduces the remaining analytic scalar input to
a centered Bernoulli Hoeffding inequality.  This theorem records, inside this development, the
standard measure-theoretic route supplied by mathlib: a centered random variable bounded in an
interval of width two has sub-Gaussian MGF parameter one, hence the exact exponential-moment
bound used by the finite Hoeffding interfaces below.
-/
theorem bounded_hoeffding_exp_mgf_of_mathlib
    {Ω : Type*} [MeasurableSpace Ω]
    {μ : MeasureTheory.Measure Ω} [MeasureTheory.IsProbabilityMeasure μ]
    {X : Ω → ℝ}
    (hm : AEMeasurable X μ)
    (hb : ∀ᵐ ω ∂μ, X ω ∈ Set.Icc (-1 : ℝ) 1)
    (hc : ∫ ω, X ω ∂μ = 0) :
    ∀ t : ℝ,
      ProbabilityTheory.mgf X μ t ≤ Real.exp (t ^ 2 / 2) := by
  intro t
  have hparam : ((nnnorm ((1 : ℝ) + 1) / 2) ^ 2 : NNReal) = 1 := by
    apply NNReal.coe_injective
    norm_num [nnnorm]
  have hsg0 :=
    (ProbabilityTheory.hasSubgaussianMGF_of_mem_Icc_of_integral_eq_zero
      (X := X) (μ := μ) (a := (-1 : ℝ)) (b := 1) hm hb hc)
  have hsg : ProbabilityTheory.HasSubgaussianMGF X 1 μ := by
    simpa [hparam] using hsg0
  simpa using hsg.mgf_le t

/--
Mathlib-backed Hoeffding bridge for centered variables supported on any interval of width two.

This is the version needed for centered Bernoulli variables: their range is
`[-2a, 2(1-a)]`, whose width is exactly two, not necessarily `[-1,1]`.
-/
theorem bounded_width_two_hoeffding_exp_mgf_of_mathlib
    {Ω : Type*} [MeasurableSpace Ω]
    {μ : MeasureTheory.Measure Ω} [MeasureTheory.IsProbabilityMeasure μ]
    {X : Ω → ℝ} {lo hi : ℝ}
    (hwidth : hi - lo = 2)
    (hm : AEMeasurable X μ)
    (hb : ∀ᵐ ω ∂μ, X ω ∈ Set.Icc lo hi)
    (hc : ∫ ω, X ω ∂μ = 0) :
    ∀ t : ℝ,
      ProbabilityTheory.mgf X μ t ≤ Real.exp (t ^ 2 / 2) := by
  intro t
  have hparam : ((nnnorm (hi - lo) / 2) ^ 2 : NNReal) = 1 := by
    rw [hwidth]
    apply NNReal.coe_injective
    norm_num [nnnorm]
  have hsg0 :=
    (ProbabilityTheory.hasSubgaussianMGF_of_mem_Icc_of_integral_eq_zero
      (X := X) (μ := μ) (a := lo) (b := hi) hm hb hc)
  have hsg : ProbabilityTheory.HasSubgaussianMGF X 1 μ := by
    simpa [hparam] using hsg0
  simpa using hsg.mgf_le t

/-- Two-point Bernoulli probability measure on `Bool`, with mass `p` at `true`. -/
noncomputable def bernoulliBoolMeasure (p : unitInterval) : MeasureTheory.Measure Bool :=
  unitInterval.toNNReal p • MeasureTheory.Measure.dirac true
    + unitInterval.toNNReal (unitInterval.symm p) • MeasureTheory.Measure.dirac false

/-- Centered sign variable associated with `bernoulliBoolMeasure`. -/
noncomputable def bernoulliSignCentered (p : unitInterval) : Bool → ℝ :=
  fun b => if b then 2 * (1 - (p : ℝ)) else -2 * (p : ℝ)

/-- The two-point Bernoulli measure is a probability measure. -/
theorem bernoulliBoolMeasure_isProbability (p : unitInterval) :
    MeasureTheory.IsProbabilityMeasure (bernoulliBoolMeasure p) := by
  dsimp [bernoulliBoolMeasure]
  infer_instance

/-- The centered Bernoulli sign variable is measurable. -/
theorem bernoulliSignCentered_aemeasurable (p : unitInterval) :
    AEMeasurable (bernoulliSignCentered p) (bernoulliBoolMeasure p) := by
  exact (measurable_of_finite (bernoulliSignCentered p)).aemeasurable

/-- The centered Bernoulli sign variable is supported on an interval of width two. -/
theorem bernoulliSignCentered_range (p : unitInterval) :
    ∀ᵐ b ∂bernoulliBoolMeasure p,
      bernoulliSignCentered p b ∈
        Set.Icc (-2 * (p : ℝ)) (2 * (1 - (p : ℝ))) := by
  refine Filter.Eventually.of_forall ?_
  intro b
  cases b <;> dsimp [bernoulliSignCentered] <;>
    constructor <;> nlinarith [p.2.1, p.2.2]

/-- The centered Bernoulli sign variable has mean zero. -/
theorem bernoulliSignCentered_integral_eq_zero (p : unitInterval) :
    ∫ b, bernoulliSignCentered p b ∂bernoulliBoolMeasure p = 0 := by
  dsimp [bernoulliBoolMeasure, bernoulliSignCentered]
  simp [MeasureTheory.integral_add_measure, NNReal.smul_def, smul_eq_mul,
    unitInterval.coe_toNNReal]
  ring_nf

/-- Explicit MGF of the centered Bernoulli sign variable. -/
theorem bernoulliSignCentered_mgf_eq (p : unitInterval) (t : ℝ) :
    ProbabilityTheory.mgf (bernoulliSignCentered p) (bernoulliBoolMeasure p) t =
      (p : ℝ) * Real.exp (t * (2 * (1 - (p : ℝ))))
        + (1 - (p : ℝ)) * Real.exp (t * (-2 * (p : ℝ))) := by
  dsimp [ProbabilityTheory.mgf, bernoulliBoolMeasure, bernoulliSignCentered]
  simp [MeasureTheory.integral_add_measure, NNReal.smul_def, smul_eq_mul,
    unitInterval.coe_toNNReal]

/--
Centered Bernoulli MGF bound obtained by instantiating mathlib's Hoeffding lemma on `Bool`.
-/
theorem bernoulliSignCentered_mgf_bound_of_mathlib (p : unitInterval) :
    ∀ t : ℝ,
      ProbabilityTheory.mgf (bernoulliSignCentered p) (bernoulliBoolMeasure p) t
        ≤ Real.exp (t ^ 2 / 2) := by
  haveI : MeasureTheory.IsProbabilityMeasure (bernoulliBoolMeasure p) :=
    bernoulliBoolMeasure_isProbability p
  exact bounded_width_two_hoeffding_exp_mgf_of_mathlib
    (X := bernoulliSignCentered p) (μ := bernoulliBoolMeasure p)
    (lo := -2 * (p : ℝ)) (hi := 2 * (1 - (p : ℝ)))
    (by ring)
    (bernoulliSignCentered_aemeasurable p)
    (bernoulliSignCentered_range p)
    (bernoulliSignCentered_integral_eq_zero p)

/--
One-parameter centered Bernoulli Hoeffding inequality, fully discharged from mathlib.

This theorem closes the scalar Hoeffding frontier previously left as the A.3 analytic input.
-/
theorem centeredBernoulli_unit_hoeffding_of_mathlib :
    ∀ a t : ℝ, 0 ≤ a → a ≤ 1 → 0 ≤ t →
      a * Real.exp (2 * (1 - a) * t) + (1 - a) * Real.exp (-2 * a * t)
        ≤ Real.exp (t ^ 2 / 2) := by
  intro a t ha ha1 _ht
  let p : unitInterval := ⟨a, ha, ha1⟩
  have hmgf := bernoulliSignCentered_mgf_bound_of_mathlib p t
  have hform :
      ProbabilityTheory.mgf (bernoulliSignCentered p) (bernoulliBoolMeasure p) t =
        a * Real.exp (2 * (1 - a) * t) + (1 - a) * Real.exp (-2 * a * t) := by
    rw [bernoulliSignCentered_mgf_eq]
    dsimp [p]
    ring_nf
  simpa [hform] using hmgf

/--
Specialize a reusable bounded-test Hoeffding theorem to the `finiteSign` test.

This isolates the only classical analytic ingredient still missing for the fully
self-contained finite Pinsker proof: Hoeffding's exponential-moment bound for
any finite test bounded by one.
-/
theorem finiteSign_exp_mgf_bound_of_bounded_hoeffding
    {q v : Fin n → ℝ}
    (hhoeffding :
      ∀ s : Fin n → ℝ,
        (∀ i, |s i| ≤ 1) →
        ∀ t : ℝ, 0 ≤ t →
          ∑ i, q i * Real.exp (t * s i)
            ≤ Real.exp (t * (∑ i, q i * s i) + t ^ 2 / 2)) :
    ∀ t : ℝ, 0 ≤ t →
      ∑ i, q i * Real.exp (t * finiteSign v i)
        ≤ Real.exp (t * (∑ i, q i * finiteSign v i) + t ^ 2 / 2) := by
  exact hhoeffding (finiteSign v) (abs_finiteSign_le_one v)

/--
Build the quadratic Pinsker family from the signed variational inequality and
the corresponding signed MGF/Hoeffding estimate.

This theorem is the middle layer between the general KL variational formula and
the optimization-free Pinsker bridge below.  It no longer assumes the quadratic
family directly: it derives it from the exact finite sign test used in the paper.
-/
theorem quadratic_variational_of_signed_mgf_bound
    {p q : Fin n → ℝ}
    (hvar :
      ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p i * finiteSign (fun j => p j - q j) i)
          - Real.log
              (∑ i, q i * Real.exp (t * finiteSign (fun j => p j - q j) i))
          ≤ finiteKL p q)
    (hmgf :
      ∀ t : ℝ, 0 ≤ t →
        Real.log
            (∑ i, q i * Real.exp (t * finiteSign (fun j => p j - q j) i))
          ≤ t * (∑ i, q i * finiteSign (fun j => p j - q j) i)
            + t ^ 2 / 2) :
    ∀ t : ℝ, 0 ≤ t →
      t * l1Norm (fun i => p i - q i) - t ^ 2 / 2 ≤ finiteKL p q := by
  intro t ht
  have hv := hvar t ht
  have hm := hmgf t ht
  have hlower :
      t * (∑ i, p i * finiteSign (fun j => p j - q j) i)
        - (t * (∑ i, q i * finiteSign (fun j => p j - q j) i) + t ^ 2 / 2)
        ≤ finiteKL p q := by
    linarith
  have hsum :
      (∑ i, p i * finiteSign (fun j => p j - q j) i)
        - (∑ i, q i * finiteSign (fun j => p j - q j) i)
        = ∑ i, (p i - q i) * finiteSign (fun j => p j - q j) i := by
    calc
      (∑ i, p i * finiteSign (fun j => p j - q j) i)
          - (∑ i, q i * finiteSign (fun j => p j - q j) i)
          = ∑ i,
              (p i * finiteSign (fun j => p j - q j) i
                - q i * finiteSign (fun j => p j - q j) i) := by
              rw [Finset.sum_sub_distrib]
      _ = ∑ i, (p i - q i) * finiteSign (fun j => p j - q j) i := by
              refine Finset.sum_congr rfl ?_
              intro i hi
              ring
  have hsign :
      ∑ i, (p i - q i) * finiteSign (fun j => p j - q j) i
        = l1Norm (fun i => p i - q i) :=
    finiteSign_l1_identity (fun i => p i - q i)
  have halg :
      t * (∑ i, p i * finiteSign (fun j => p j - q j) i)
        - (t * (∑ i, q i * finiteSign (fun j => p j - q j) i) + t ^ 2 / 2)
        = t * l1Norm (fun i => p i - q i) - t ^ 2 / 2 := by
    rw [← hsign, ← hsum]
    ring
  simpa [halg] using hlower

/--
Optimization-free Pinsker bridge.

If the variational/Hoeffding part has produced the family
`t * ‖p-q‖₁ - t^2/2 ≤ KL(p‖q)` for all nonnegative `t`, evaluating it at
`t = ‖p-q‖₁` gives the normalized Pinsker constant.
-/
theorem normalizedPinsker_of_variational_quadratic
    {p q : Fin n → ℝ}
    (hquad :
      ∀ t : ℝ, 0 ≤ t →
        t * l1Norm (fun i => p i - q i) - t ^ 2 / 2 ≤ finiteKL p q) :
    (l1Norm (fun i => p i - q i)) ^ 2 / 2 ≤ finiteKL p q := by
  let L := l1Norm (fun i => p i - q i)
  have hL : 0 ≤ L := l1Norm_nonneg (fun i => p i - q i)
  have h := hquad L hL
  have hopt : L * l1Norm (fun i => p i - q i) - L ^ 2 / 2 = L ^ 2 / 2 := by
    simp [L]
    ring
  have h' : L ^ 2 / 2 ≤ finiteKL p q := by
    simpa [hopt] using h
  simpa [L] using h'

/--
Named normalized Pinsker interface used by the non-normalized scaling theorem.

The remaining analytic work is to discharge `hquad` from the finite-dimensional variational
formula plus the bounded-difference/MGF estimate.
-/
theorem normalizedPinsker_probability_form_of_quadratic_variational
    {p q : Fin n → ℝ}
    (hquad :
      ∀ t : ℝ, 0 ≤ t →
        t * l1Norm (fun i => p i - q i) - t ^ 2 / 2 ≤ finiteKL p q) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / 2 :=
  normalizedPinsker_of_variational_quadratic hquad

/--
Non-normalized Pinsker inequality from the classical probability-mass form.

`hprob` is exactly the classical probability Pinsker inequality applied to the
normalized vectors `p/M` and `q/M`.
-/
theorem pinsker_nonnormalized_of_probability_form
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hprob :
      finiteKL (fun i => p i / M) (fun i => q i / M)
        ≥ (l1Norm (fun i => p i / M - q i / M)) ^ 2 / 2) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
  have hKL : finiteKL p q = M * finiteKL (fun i => p i / M) (fun i => q i / M) :=
    finiteKL_scale (p := p) (q := q) hMpos
  have hL1 : l1Norm (fun i => p i - q i) = M * l1Norm (fun i => p i / M - q i / M) :=
    l1Norm_sub_scale (p := p) (q := q) hMpos
  have hMnonneg : 0 ≤ M := le_of_lt hMpos
  have hmul : M * finiteKL (fun i => p i / M) (fun i => q i / M)
      ≥ M * ((l1Norm (fun i => p i / M - q i / M)) ^ 2 / 2) :=
    mul_le_mul_of_nonneg_left hprob hMnonneg
  have htarget :
      M * ((l1Norm (fun i => p i / M - q i / M)) ^ 2 / 2)
        = (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
    have hM0 : M ≠ 0 := ne_of_gt hMpos
    rw [hL1]
    ring_nf
    field_simp [hM0]
  calc
    finiteKL p q
        = M * finiteKL (fun i => p i / M) (fun i => q i / M) := hKL
    _ ≥ M * ((l1Norm (fun i => p i / M - q i / M)) ^ 2 / 2) := hmul
    _ = (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := htarget

/--
Non-normalized Pinsker from the explicit quadratic variational family for normalized vectors.

Compared with `pinsker_nonnormalized_of_probability_form`, this endpoint no longer assumes
normalized Pinsker as a black box.  It assumes the preceding variational/MGF family, then derives
the normalized Pinsker constant internally and applies the certified scaling step.
-/
theorem pinsker_nonnormalized_of_quadratic_variational
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hquad :
      ∀ t : ℝ, 0 ≤ t →
        t * l1Norm (fun i => p i / M - q i / M)
          - t ^ 2 / 2
          ≤ finiteKL (fun i => p i / M) (fun i => q i / M)) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
  have hprob :
      finiteKL (fun i => p i / M) (fun i => q i / M)
        ≥ (l1Norm (fun i => p i / M - q i / M)) ^ 2 / 2 :=
    normalizedPinsker_probability_form_of_quadratic_variational hquad
  exact pinsker_nonnormalized_of_probability_form (p := p) (q := q) hMpos hprob

/--
Paper-facing non-normalized Pinsker endpoint from the signed variational/MGF proof path.

The remaining analytic premises are now explicit and local:
the KL variational inequality for the finite sign test (`hvar`) and the
Hoeffding/MGF bound for that same sign test (`hmgf`).  The sign-to-`ℓ¹`
identity, quadratic optimization, and non-normalized scaling are all certified here.
-/
theorem pinsker_nonnormalized_of_signed_variational_mgf
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hvar :
      ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p i / M) *
            finiteSign (fun j => p j / M - q j / M) i)
          - Real.log
              (∑ i, (q i / M) *
                Real.exp (t * finiteSign (fun j => p j / M - q j / M) i))
          ≤ finiteKL (fun i => p i / M) (fun i => q i / M))
    (hmgf :
      ∀ t : ℝ, 0 ≤ t →
        Real.log
            (∑ i, (q i / M) *
              Real.exp (t * finiteSign (fun j => p j / M - q j / M) i))
          ≤ t * (∑ i, (q i / M) *
              finiteSign (fun j => p j / M - q j / M) i)
            + t ^ 2 / 2) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
  exact pinsker_nonnormalized_of_quadratic_variational (p := p) (q := q) hMpos
    (quadratic_variational_of_signed_mgf_bound
      (p := fun i => p i / M) (q := fun i => q i / M) hvar hmgf)

/--
Paper-facing non-normalized Pinsker endpoint from the signed variational proof and
the standard exponential Hoeffding/MGF form.

Compared with `pinsker_nonnormalized_of_signed_variational_mgf`, this theorem also certifies
the conversion from the usual exponential MGF inequality to the logarithmic form used in the
paper proof.  The remaining analytic frontier is exactly the finite Hoeffding exponential
bound `hexp` for the bounded sign test.
-/
theorem pinsker_nonnormalized_of_signed_variational_exp_mgf
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ i, 0 ≤ q i / M)
    (hq_mass : ∑ i, q i / M = 1)
    (hvar :
      ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p i / M) *
            finiteSign (fun j => p j / M - q j / M) i)
          - Real.log
              (∑ i, (q i / M) *
                Real.exp (t * finiteSign (fun j => p j / M - q j / M) i))
          ≤ finiteKL (fun i => p i / M) (fun i => q i / M))
    (hexp :
      ∀ t : ℝ, 0 ≤ t →
        ∑ i, (q i / M) *
            Real.exp (t * finiteSign (fun j => p j / M - q j / M) i)
          ≤ Real.exp
              (t * (∑ i, (q i / M) *
                finiteSign (fun j => p j / M - q j / M) i) + t ^ 2 / 2)) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
  have hmgf :
      ∀ t : ℝ, 0 ≤ t →
        Real.log
            (∑ i, (q i / M) *
              Real.exp (t * finiteSign (fun j => p j / M - q j / M) i))
          ≤ t * (∑ i, (q i / M) *
              finiteSign (fun j => p j / M - q j / M) i)
            + t ^ 2 / 2 :=
    log_mgf_bound_of_exp_mgf_bound
      (q := fun i => q i / M)
      (s := finiteSign (fun j => p j / M - q j / M))
      hq_nonneg hq_mass hexp
  exact pinsker_nonnormalized_of_signed_variational_mgf
    (p := p) (q := q) hMpos hvar hmgf

/--
Paper-facing non-normalized Pinsker endpoint from signed variational input plus
a reusable bounded-test Hoeffding theorem.

This endpoint is the cleanest current A.3 interface: all algebraic, sign, logarithmic,
optimization, and scaling steps are certified; the only external analytic theorem is the
standard finite Hoeffding exponential-MGF bound for tests with `|s_i| ≤ 1`.
-/
theorem pinsker_nonnormalized_of_signed_variational_bounded_hoeffding
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ i, 0 ≤ q i / M)
    (hq_mass : ∑ i, q i / M = 1)
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
            ≤ Real.exp (t * (∑ i, (q i / M) * s i) + t ^ 2 / 2)) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
  have hexp :
      ∀ t : ℝ, 0 ≤ t →
        ∑ i, (q i / M) *
            Real.exp (t * finiteSign (fun j => p j / M - q j / M) i)
          ≤ Real.exp
              (t * (∑ i, (q i / M) *
                finiteSign (fun j => p j / M - q j / M) i) + t ^ 2 / 2) :=
    finiteSign_exp_mgf_bound_of_bounded_hoeffding
      (q := fun i => q i / M)
      (v := fun j => p j / M - q j / M)
      hhoeffding
  exact pinsker_nonnormalized_of_signed_variational_exp_mgf
    (p := p) (q := q) hMpos hq_nonneg hq_mass hvar hexp

/--
Specialize the finite sign-test MGF bound to a scalar two-point Hoeffding inequality.

For the Pinsker sign test, the random variable only takes values `1` and `-1`.
This lemma proves the finite decomposition into the positive and negative sign masses, so the
remaining analytic input is the scalar inequality for a two-point distribution.
-/
theorem finiteSign_exp_mgf_bound_of_twoPoint_hoeffding
    {q v : Fin n → ℝ}
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = 1)
    (htwoPoint :
      ∀ a b t : ℝ, 0 ≤ a → 0 ≤ b → a + b = 1 → 0 ≤ t →
        a * Real.exp t + b * Real.exp (-t)
          ≤ Real.exp (t * (a - b) + t ^ 2 / 2)) :
    ∀ t : ℝ, 0 ≤ t →
      ∑ i, q i * Real.exp (t * finiteSign v i)
        ≤ Real.exp (t * (∑ i, q i * finiteSign v i) + t ^ 2 / 2) := by
  intro t ht
  let a := finiteSignPosMass q v
  let b := finiteSignNegMass q v
  have ha : 0 ≤ a := finiteSignPosMass_nonneg (q := q) (v := v) hq_nonneg
  have hb : 0 ≤ b := finiteSignNegMass_nonneg (q := q) (v := v) hq_nonneg
  have hmass : a + b = 1 := by
    dsimp [a, b]
    rw [finiteSignPosMass_add_negMass, hq_mass]
  have htwo := htwoPoint a b t ha hb hmass ht
  have hexp :
      ∑ i, q i * Real.exp (t * finiteSign v i)
        = a * Real.exp t + b * Real.exp (-t) := by
    dsimp [a, b]
    exact finiteSign_exp_sum_eq_twoPoint q v t
  have hmean :
      ∑ i, q i * finiteSign v i = a - b := by
    dsimp [a, b]
    exact finiteSign_mean_eq_posMass_sub_negMass q v
  calc
    ∑ i, q i * Real.exp (t * finiteSign v i)
        = a * Real.exp t + b * Real.exp (-t) := hexp
    _ ≤ Real.exp (t * (a - b) + t ^ 2 / 2) := htwo
    _ = Real.exp (t * (∑ i, q i * finiteSign v i) + t ^ 2 / 2) := by
          rw [hmean]

/--
Convert the centered Bernoulli Hoeffding MGF form into the two-point form needed by the
finite sign-test proof.

When `a + b = 1`, the sign variable with masses `a` at `1` and `b` at `-1` has mean `a - b`.
After factoring out this mean, its centered exponential moment is exactly
`a exp (2bt) + b exp (-2at)`.
-/
theorem twoPoint_hoeffding_of_centeredBernoulli_hoeffding
    (hcentered :
      ∀ a b t : ℝ, 0 ≤ a → 0 ≤ b → a + b = 1 → 0 ≤ t →
        a * Real.exp (2 * b * t) + b * Real.exp (-2 * a * t)
          ≤ Real.exp (t ^ 2 / 2)) :
    ∀ a b t : ℝ, 0 ≤ a → 0 ≤ b → a + b = 1 → 0 ≤ t →
      a * Real.exp t + b * Real.exp (-t)
        ≤ Real.exp (t * (a - b) + t ^ 2 / 2) := by
  intro a b t ha hb hmass ht
  have hpos : 0 ≤ Real.exp (t * (a - b)) := le_of_lt (Real.exp_pos _)
  have h1 :
      Real.exp (t * (a - b)) * Real.exp (2 * b * t) = Real.exp t := by
    rw [← Real.exp_add]
    congr 1
    nlinarith [hmass]
  have h2 :
      Real.exp (t * (a - b)) * Real.exp (-2 * a * t) = Real.exp (-t) := by
    rw [← Real.exp_add]
    congr 1
    nlinarith [hmass]
  have hdecomp :
      a * Real.exp t + b * Real.exp (-t)
        =
      Real.exp (t * (a - b)) *
        (a * Real.exp (2 * b * t) + b * Real.exp (-2 * a * t)) := by
    symm
    calc
      Real.exp (t * (a - b)) *
          (a * Real.exp (2 * b * t) + b * Real.exp (-2 * a * t))
          = a * (Real.exp (t * (a - b)) * Real.exp (2 * b * t))
              + b * (Real.exp (t * (a - b)) * Real.exp (-2 * a * t)) := by
              ring
      _ = a * Real.exp t + b * Real.exp (-t) := by
              rw [h1, h2]
  have hmul :
      Real.exp (t * (a - b)) *
          (a * Real.exp (2 * b * t) + b * Real.exp (-2 * a * t))
        ≤ Real.exp (t * (a - b)) * Real.exp (t ^ 2 / 2) :=
    mul_le_mul_of_nonneg_left (hcentered a b t ha hb hmass ht) hpos
  calc
    a * Real.exp t + b * Real.exp (-t)
        = Real.exp (t * (a - b)) *
          (a * Real.exp (2 * b * t) + b * Real.exp (-2 * a * t)) := hdecomp
    _ ≤ Real.exp (t * (a - b)) * Real.exp (t ^ 2 / 2) := hmul
    _ = Real.exp (t * (a - b) + t ^ 2 / 2) := by
          rw [← Real.exp_add]

/--
Sign-test MGF bound from the centered Bernoulli scalar Hoeffding inequality.

This is a sharper interface than `finiteSign_exp_mgf_bound_of_twoPoint_hoeffding`: the only
remaining scalar analytic fact is the centered Bernoulli MGF bound.
-/
theorem finiteSign_exp_mgf_bound_of_centeredBernoulli_hoeffding
    {q v : Fin n → ℝ}
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = 1)
    (hcentered :
      ∀ a b t : ℝ, 0 ≤ a → 0 ≤ b → a + b = 1 → 0 ≤ t →
        a * Real.exp (2 * b * t) + b * Real.exp (-2 * a * t)
          ≤ Real.exp (t ^ 2 / 2)) :
    ∀ t : ℝ, 0 ≤ t →
      ∑ i, q i * Real.exp (t * finiteSign v i)
        ≤ Real.exp (t * (∑ i, q i * finiteSign v i) + t ^ 2 / 2) := by
  exact finiteSign_exp_mgf_bound_of_twoPoint_hoeffding hq_nonneg hq_mass
    (twoPoint_hoeffding_of_centeredBernoulli_hoeffding hcentered)

/-- The centered Bernoulli MGF bound is trivial when the positive mass is zero. -/
theorem centeredBernoulli_hoeffding_of_leftMass_zero
    {a b t : ℝ}
    (ha : a = 0)
    (hmass : a + b = 1)
    (_ht : 0 ≤ t) :
    a * Real.exp (2 * b * t) + b * Real.exp (-2 * a * t)
      ≤ Real.exp (t ^ 2 / 2) := by
  have hb : b = 1 := by linarith
  have ht2 : 0 ≤ t ^ 2 / 2 := by nlinarith [sq_nonneg t]
  calc
    a * Real.exp (2 * b * t) + b * Real.exp (-2 * a * t)
        = 1 := by simp [ha, hb]
    _ ≤ Real.exp (t ^ 2 / 2) := by
          simpa using Real.one_le_exp_iff.mpr ht2

/-- The centered Bernoulli MGF bound is trivial when the negative mass is zero. -/
theorem centeredBernoulli_hoeffding_of_rightMass_zero
    {a b t : ℝ}
    (hb : b = 0)
    (hmass : a + b = 1)
    (_ht : 0 ≤ t) :
    a * Real.exp (2 * b * t) + b * Real.exp (-2 * a * t)
      ≤ Real.exp (t ^ 2 / 2) := by
  have ha : a = 1 := by linarith
  have ht2 : 0 ≤ t ^ 2 / 2 := by nlinarith [sq_nonneg t]
  calc
    a * Real.exp (2 * b * t) + b * Real.exp (-2 * a * t)
        = 1 := by simp [ha, hb]
    _ ≤ Real.exp (t ^ 2 / 2) := by
          simpa using Real.one_le_exp_iff.mpr ht2

/-- The centered Bernoulli MGF bound is exact at time zero. -/
theorem centeredBernoulli_hoeffding_of_time_zero
    {a b t : ℝ}
    (ht0 : t = 0)
    (hmass : a + b = 1) :
    a * Real.exp (2 * b * t) + b * Real.exp (-2 * a * t)
      ≤ Real.exp (t ^ 2 / 2) := by
  calc
    a * Real.exp (2 * b * t) + b * Real.exp (-2 * a * t)
        = 1 := by simp [ht0, hmass]
    _ ≤ Real.exp (t ^ 2 / 2) := by simp [ht0]

/--
Extend centered Bernoulli Hoeffding from the genuinely nondegenerate case.

This removes the degenerate probability/time cases from the remaining scalar analytic premise:
the only external scalar fact still needed is the interior case `0 < a`, `0 < b`, `0 < t`.
-/
theorem centeredBernoulli_hoeffding_of_pos_pos_pos
    (hinterior :
      ∀ a b t : ℝ, 0 < a → 0 < b → a + b = 1 → 0 < t →
        a * Real.exp (2 * b * t) + b * Real.exp (-2 * a * t)
          ≤ Real.exp (t ^ 2 / 2)) :
    ∀ a b t : ℝ, 0 ≤ a → 0 ≤ b → a + b = 1 → 0 ≤ t →
      a * Real.exp (2 * b * t) + b * Real.exp (-2 * a * t)
        ≤ Real.exp (t ^ 2 / 2) := by
  intro a b t ha hb hmass ht
  rcases lt_or_eq_of_le ha with ha_pos | ha_zero
  · rcases lt_or_eq_of_le hb with hb_pos | hb_zero
    · rcases lt_or_eq_of_le ht with ht_pos | ht_zero
      · exact hinterior a b t ha_pos hb_pos hmass ht_pos
      · exact centeredBernoulli_hoeffding_of_time_zero ht_zero.symm hmass
    · exact centeredBernoulli_hoeffding_of_rightMass_zero hb_zero.symm hmass ht
  · exact centeredBernoulli_hoeffding_of_leftMass_zero ha_zero.symm hmass ht

/--
Reduce the strict-interior centered Bernoulli Hoeffding premise to the usual one-parameter form.

Instead of carrying two positive masses `a,b` plus the constraint `a+b=1`, the remaining scalar
analytic input can be stated for one probability `a ∈ (0,1)` with the other mass `1-a`.
-/
theorem centeredBernoulli_interior_hoeffding_of_unitInterval
    (hunit :
      ∀ a t : ℝ, 0 < a → a < 1 → 0 < t →
        a * Real.exp (2 * (1 - a) * t) + (1 - a) * Real.exp (-2 * a * t)
          ≤ Real.exp (t ^ 2 / 2)) :
    ∀ a b t : ℝ, 0 < a → 0 < b → a + b = 1 → 0 < t →
      a * Real.exp (2 * b * t) + b * Real.exp (-2 * a * t)
        ≤ Real.exp (t ^ 2 / 2) := by
  intro a b t ha hb hmass ht
  have hb_eq : b = 1 - a := by linarith
  have ha_lt_one : a < 1 := by linarith
  simpa [hb_eq] using hunit a t ha ha_lt_one ht

/--
Specialize a finite variational inequality for all test functions to the Pinsker sign test.

This is the finite-dimensional counterpart of applying the Donsker--Varadhan/KL variational
lower bound with `f = t * sign(p-q)`.
-/
theorem signed_variational_of_finite_variational
    {p q v : Fin n → ℝ}
    (hvarAll :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p i * s i)
          - Real.log (∑ i, q i * Real.exp (t * s i))
          ≤ finiteKL p q) :
    ∀ t : ℝ, 0 ≤ t →
      t * (∑ i, p i * finiteSign v i)
        - Real.log (∑ i, q i * Real.exp (t * finiteSign v i))
        ≤ finiteKL p q := by
  intro t ht
  exact hvarAll (finiteSign v) t ht

/--
Normalized-vector specialization of `signed_variational_of_finite_variational` for the exact
sign test used by the non-normalized Pinsker scaling argument.
-/
theorem signed_variational_normalized_of_finite_variational
    {p q : Fin n → ℝ} {M : ℝ}
    (hvarAll :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p i / M) * s i)
          - Real.log (∑ i, (q i / M) * Real.exp (t * s i))
          ≤ finiteKL (fun i => p i / M) (fun i => q i / M)) :
    ∀ t : ℝ, 0 ≤ t →
      t * (∑ i, (p i / M) *
          finiteSign (fun j => p j / M - q j / M) i)
        - Real.log
            (∑ i, (q i / M) *
              Real.exp (t * finiteSign (fun j => p j / M - q j / M) i))
        ≤ finiteKL (fun i => p i / M) (fun i => q i / M) := by
  exact signed_variational_of_finite_variational
    (p := fun i => p i / M)
    (q := fun i => q i / M)
    (v := fun j => p j / M - q j / M)
    hvarAll

/--
Normalize a common-mass finite KL variational inequality.

The paper naturally writes the finite variational input on the original common mass shell
`∑ q_i = M`.  The Pinsker proof itself is cleaner after normalizing by `M`; this lemma certifies
that passage using the explicit finite-KL scaling identity.
-/
theorem finite_variational_normalized_of_massShell_variational
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hvarMass :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p i * s i)
          - M * Real.log (∑ i, (q i / M) * Real.exp (t * s i))
          ≤ finiteKL p q) :
    ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
      t * (∑ i, (p i / M) * s i)
        - Real.log (∑ i, (q i / M) * Real.exp (t * s i))
        ≤ finiteKL (fun i => p i / M) (fun i => q i / M) := by
  intro s t ht
  have hM0 : M ≠ 0 := ne_of_gt hMpos
  have hMnonneg : 0 ≤ M := le_of_lt hMpos
  have hscale : finiteKL p q = M * finiteKL (fun i => p i / M) (fun i => q i / M) :=
    finiteKL_scale (p := p) (q := q) hMpos
  have hsum :
      ∑ i, p i * s i = M * ∑ i, (p i / M) * s i := by
    calc
      ∑ i, p i * s i
          = ∑ i, (M * (p i / M)) * s i := by
              refine Finset.sum_congr rfl ?_
              intro i hi
              have hp : M * (p i / M) = p i := by field_simp [hM0]
              rw [hp]
      _ = ∑ i, M * ((p i / M) * s i) := by
              refine Finset.sum_congr rfl ?_
              intro i hi
              ring
      _ = M * ∑ i, (p i / M) * s i := by
              rw [Finset.mul_sum]
  have hmass :=
    hvarMass s t ht
  have hmul :
      M * (t * (∑ i, (p i / M) * s i)
          - Real.log (∑ i, (q i / M) * Real.exp (t * s i)))
        ≤ M * finiteKL (fun i => p i / M) (fun i => q i / M) := by
    calc
      M * (t * (∑ i, (p i / M) * s i)
          - Real.log (∑ i, (q i / M) * Real.exp (t * s i)))
          = t * (∑ i, p i * s i)
              - M * Real.log (∑ i, (q i / M) * Real.exp (t * s i)) := by
                rw [hsum]
                ring
      _ ≤ finiteKL p q := hmass
      _ = M * finiteKL (fun i => p i / M) (fun i => q i / M) := hscale
  exact le_of_mul_le_mul_left hmul hMpos

/--
Integral computation for mathlib's log-likelihood ratio on the concrete finite measures.

Under strict positivity of `q`, the Radon-Nikodym derivative is the atomic ratio `p_i / q_i`
on the `q`-measure a.e. support.  Absolute continuity transfers this to the `p`-integral,
which is exactly the handwritten finite KL sum.
-/
theorem finiteProbabilityMeasure_integral_llr_eq_finiteKL
    {p q : Fin n → ℝ}
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hq_pos : ∀ i, 0 < q i) :
    ∫ i, MeasureTheory.llr (finiteProbabilityMeasure p) (finiteProbabilityMeasure q) i
        ∂finiteProbabilityMeasure p
      = finiteKL p q := by
  let μ := finiteProbabilityMeasure p
  let ν := finiteProbabilityMeasure q
  have hμν : μ ≪ ν := finiteProbabilityMeasure_absolutelyContinuous p q hq_pos
  have hrn_ν :
      (finiteProbabilityMeasure p).rnDeriv (finiteProbabilityMeasure q)
        =ᵐ[finiteProbabilityMeasure q] fun i => ENNReal.ofReal (p i / q i) :=
    finiteProbabilityMeasure_rnDeriv_ae (p := p) (q := q) hq_pos
  have hllr : MeasureTheory.llr μ ν =ᵐ[μ] fun i => Real.log (p i / q i) := by
    filter_upwards [hμν.ae_le hrn_ν] with i hi
    simp only [μ, ν, MeasureTheory.llr]
    rw [hi]
    exact congrArg Real.log
      (ENNReal.toReal_ofReal (div_nonneg (hp_nonneg i) (le_of_lt (hq_pos i))))
  calc
    ∫ i, MeasureTheory.llr μ ν i ∂μ = ∫ i, Real.log (p i / q i) ∂μ :=
      integral_congr_ae hllr
    _ = ∑ i, p i * Real.log (p i / q i) :=
      finiteProbabilityMeasure_integral p (fun i => Real.log (p i / q i)) hp_nonneg
    _ = finiteKL p q := rfl

/--
Support-aware integral computation for mathlib's log-likelihood ratio.

This is the boundary analogue of `finiteProbabilityMeasure_integral_llr_eq_finiteKL`: zero
reference atoms are allowed as long as `p` gives them zero mass.  Absolute continuity follows
from support domination, and the explicit finite KL remains finite because all problematic
zero-reference atoms have zero `p`-weight.
-/
theorem finiteProbabilityMeasure_integral_llr_eq_finiteKL_of_support
    {p q : Fin n → ℝ}
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hsupp : ∀ i, q i = 0 → p i = 0) :
    ∫ i, MeasureTheory.llr (finiteProbabilityMeasure p) (finiteProbabilityMeasure q) i
        ∂finiteProbabilityMeasure p
      = finiteKL p q := by
  let μ := finiteProbabilityMeasure p
  let ν := finiteProbabilityMeasure q
  have hμν : μ ≪ ν :=
    finiteProbabilityMeasure_absolutelyContinuous_of_support p q hq_nonneg hsupp
  have hrn_ν :
    (finiteProbabilityMeasure p).rnDeriv (finiteProbabilityMeasure q)
        =ᵐ[finiteProbabilityMeasure q] fun i => ENNReal.ofReal (p i / q i) :=
    finiteProbabilityMeasure_rnDeriv_ae_of_support
      (p := p) (q := q) hq_nonneg hsupp
  have hllr : MeasureTheory.llr μ ν =ᵐ[μ] fun i => Real.log (p i / q i) := by
    filter_upwards [hμν.ae_le hrn_ν] with i hi
    simp only [μ, ν, MeasureTheory.llr]
    rw [hi]
    exact congrArg Real.log
      (ENNReal.toReal_ofReal (div_nonneg (hp_nonneg i) (hq_nonneg i)))
  calc
    ∫ i, MeasureTheory.llr μ ν i ∂μ = ∫ i, Real.log (p i / q i) ∂μ :=
      integral_congr_ae hllr
    _ = ∑ i, p i * Real.log (p i / q i) :=
      finiteProbabilityMeasure_integral p (fun i => Real.log (p i / q i)) hp_nonneg
    _ = finiteKL p q := rfl

/--
Exact comparison between mathlib's `klDiv` for the concrete finite probability measures
and the local finite KL sum.

This discharges the previous A.3 compatibility premise under the current strict positivity
assumption on the reference vector.
-/
theorem finiteProbabilityMeasure_klDiv_eq_finiteKL
    {p q : Fin n → ℝ}
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = 1)
    (hq_pos : ∀ i, 0 < q i)
    (hq_mass : ∑ i, q i = 1) :
    (InformationTheory.klDiv (finiteProbabilityMeasure p) (finiteProbabilityMeasure q)).toReal
      = finiteKL p q := by
  haveI : MeasureTheory.IsFiniteMeasure (finiteProbabilityMeasure p) :=
    finiteProbabilityMeasure_isFiniteMeasure p
  haveI : MeasureTheory.IsFiniteMeasure (finiteProbabilityMeasure q) :=
    finiteProbabilityMeasure_isFiniteMeasure q
  have hμν : finiteProbabilityMeasure p ≪ finiteProbabilityMeasure q :=
    finiteProbabilityMeasure_absolutelyContinuous p q hq_pos
  have hmass : finiteProbabilityMeasure p Set.univ = finiteProbabilityMeasure q Set.univ := by
    rw [finiteProbabilityMeasure_apply_univ p hp_nonneg]
    rw [finiteProbabilityMeasure_apply_univ q (fun i => (hq_pos i).le)]
    rw [hp_mass, hq_mass]
  rw [InformationTheory.toReal_klDiv_of_measure_eq hμν hmass]
  exact finiteProbabilityMeasure_integral_llr_eq_finiteKL hp_nonneg hq_pos

/--
Support-aware comparison between mathlib's `klDiv` for concrete finite probability measures
and the local finite KL sum.

This removes the strict-interior assumption from the KL computation.  The necessary and natural
replacement is support domination: if `q_i = 0`, then `p_i = 0`.
-/
theorem finiteProbabilityMeasure_klDiv_eq_finiteKL_of_support
    {p q : Fin n → ℝ}
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = 1)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = 1)
    (hsupp : ∀ i, q i = 0 → p i = 0) :
    (InformationTheory.klDiv (finiteProbabilityMeasure p) (finiteProbabilityMeasure q)).toReal
      = finiteKL p q := by
  haveI : MeasureTheory.IsFiniteMeasure (finiteProbabilityMeasure p) :=
    finiteProbabilityMeasure_isFiniteMeasure p
  haveI : MeasureTheory.IsFiniteMeasure (finiteProbabilityMeasure q) :=
    finiteProbabilityMeasure_isFiniteMeasure q
  have hμν : finiteProbabilityMeasure p ≪ finiteProbabilityMeasure q :=
    finiteProbabilityMeasure_absolutelyContinuous_of_support p q hq_nonneg hsupp
  have hmass : finiteProbabilityMeasure p Set.univ = finiteProbabilityMeasure q Set.univ := by
    rw [finiteProbabilityMeasure_apply_univ p hp_nonneg]
    rw [finiteProbabilityMeasure_apply_univ q hq_nonneg]
    rw [hp_mass, hq_mass]
  rw [InformationTheory.toReal_klDiv_of_measure_eq hμν hmass]
  exact finiteProbabilityMeasure_integral_llr_eq_finiteKL_of_support
    hp_nonneg hq_nonneg hsupp

/--
Concrete finite-measure KL computation after normalizing common-mass vectors.

This is the common-mass version most callers need before invoking the normalized variational or
Pinsker endpoints.
-/
theorem finiteProbabilityMeasure_normalized_klDiv_eq_finiteKL_massShell
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = M)
    (hq_pos : ∀ i, 0 < q i)
    (hq_mass : ∑ i, q i = M) :
    (InformationTheory.klDiv
      (finiteProbabilityMeasure (fun i => p i / M))
      (finiteProbabilityMeasure (fun i => q i / M))).toReal
        = finiteKL (fun i => p i / M) (fun i => q i / M) := by
  exact finiteProbabilityMeasure_klDiv_eq_finiteKL
    (p := fun i => p i / M) (q := fun i => q i / M)
    (normalizedWeight_nonneg (q := p) hMpos hp_nonneg)
    (normalizedWeight_mass (q := p) hMpos hp_mass)
    (normalizedWeight_pos (q := q) hMpos hq_pos)
    (normalizedWeight_mass (q := q) hMpos hq_mass)

/--
Support-aware concrete finite-measure KL computation after normalizing common-mass vectors.

This is the boundary common-mass analogue of
`finiteProbabilityMeasure_normalized_klDiv_eq_finiteKL_massShell`.
-/
theorem finiteProbabilityMeasure_normalized_klDiv_eq_finiteKL_massShell_of_support
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = M)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = M)
    (hsupp : ∀ i, q i = 0 → p i = 0) :
    (InformationTheory.klDiv
      (finiteProbabilityMeasure (fun i => p i / M))
      (finiteProbabilityMeasure (fun i => q i / M))).toReal
        = finiteKL (fun i => p i / M) (fun i => q i / M) := by
  have hM0 : M ≠ 0 := ne_of_gt hMpos
  have hsupp_norm : ∀ i, q i / M = 0 → p i / M = 0 := by
    intro i hq_norm
    have hq0 : q i = 0 := by
      field_simp [hM0] at hq_norm
      simpa using hq_norm
    rw [hsupp i hq0]
    simp
  exact finiteProbabilityMeasure_klDiv_eq_finiteKL_of_support
    (p := fun i => p i / M) (q := fun i => q i / M)
    (normalizedWeight_nonneg (q := p) hMpos hp_nonneg)
    (normalizedWeight_mass (q := p) hMpos hp_mass)
    (normalizedWeight_nonneg (q := q) hMpos hq_nonneg)
    (normalizedWeight_mass (q := q) hMpos hq_mass)
    hsupp_norm

/--
Finite all-test variational inequality from concrete finite probability measures,
with mathlib's measure KL compared to the handwritten finite KL.

This is the main A.3 variational bridge: the appendix variational line is now
obtained from `klDiv_variational_lower_bound` applied to the concrete measures
`finiteProbabilityMeasure p` and `finiteProbabilityMeasure q`.  The only
remaining finite-KL compatibility premise is the scalar comparison between
mathlib `klDiv` and this file's explicit sum `finiteKL`.
-/
theorem finite_variational_of_finiteProbabilityMeasure_klDiv_le_finiteKL
    {p q : Fin n → ℝ}
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = 1)
    (hq_pos : ∀ i, 0 < q i)
    (hq_mass : ∑ i, q i = 1)
    (hkl_le :
      (InformationTheory.klDiv
        (finiteProbabilityMeasure p) (finiteProbabilityMeasure q)).toReal
          ≤ finiteKL p q) :
    ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
      t * (∑ i, p i * s i)
        - Real.log (∑ i, q i * Real.exp (t * s i))
        ≤ finiteKL p q := by
  intro s t _ht
  exact
    (finiteProbabilityMeasure_variational_lower_bound
      hp_nonneg hp_mass hq_pos hq_mass s t).trans hkl_le

/--
Equality-shaped version of
`finite_variational_of_finiteProbabilityMeasure_klDiv_le_finiteKL`.

Downstream finite-simplex developments often prove the exact identity between
mathlib `klDiv` and the handwritten `finiteKL`; this wrapper turns that identity
directly into the all-test variational premise consumed by the Pinsker chain.
-/
theorem finite_variational_of_finiteProbabilityMeasure_klDiv_eq_finiteKL
    {p q : Fin n → ℝ}
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = 1)
    (hq_pos : ∀ i, 0 < q i)
    (hq_mass : ∑ i, q i = 1)
    (hkl_eq :
      (InformationTheory.klDiv
        (finiteProbabilityMeasure p) (finiteProbabilityMeasure q)).toReal
          = finiteKL p q) :
    ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
      t * (∑ i, p i * s i)
        - Real.log (∑ i, q i * Real.exp (t * s i))
        ≤ finiteKL p q := by
  exact finite_variational_of_finiteProbabilityMeasure_klDiv_le_finiteKL
    hp_nonneg hp_mass hq_pos hq_mass (le_of_eq hkl_eq)

/--
Finite all-test variational inequality with the concrete finite-measure KL computation
discharged internally.

This is the premise-minimal normalized A.3 variational endpoint under strict positivity of
the reference vector: callers no longer need to mention mathlib's `klDiv` at all.
-/
theorem finite_variational_of_finiteProbabilityMeasure_klDiv_computed
    {p q : Fin n → ℝ}
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = 1)
    (hq_pos : ∀ i, 0 < q i)
    (hq_mass : ∑ i, q i = 1) :
    ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
      t * (∑ i, p i * s i)
        - Real.log (∑ i, q i * Real.exp (t * s i))
        ≤ finiteKL p q := by
  exact finite_variational_of_finiteProbabilityMeasure_klDiv_eq_finiteKL
    hp_nonneg hp_mass hq_pos hq_mass
    (finiteProbabilityMeasure_klDiv_eq_finiteKL hp_nonneg hp_mass hq_pos hq_mass)

/--
Support-aware finite all-test variational inequality with the concrete finite-measure KL
computation discharged internally.

This is the boundary version of `finite_variational_of_finiteProbabilityMeasure_klDiv_computed`:
the strict interior assumption on `q` is replaced by nonnegativity and support domination.
-/
theorem finite_variational_of_finiteProbabilityMeasure_klDiv_computed_of_support
    {p q : Fin n → ℝ}
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = 1)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = 1)
    (hsupp : ∀ i, q i = 0 → p i = 0) :
    ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
      t * (∑ i, p i * s i)
        - Real.log (∑ i, q i * Real.exp (t * s i))
        ≤ finiteKL p q := by
  intro s t _ht
  have hkl_eq :
      (InformationTheory.klDiv
        (finiteProbabilityMeasure p) (finiteProbabilityMeasure q)).toReal
          = finiteKL p q :=
    finiteProbabilityMeasure_klDiv_eq_finiteKL_of_support
      hp_nonneg hp_mass hq_nonneg hq_mass hsupp
  simpa [hkl_eq] using
    (finiteProbabilityMeasure_variational_lower_bound_of_support
      hp_nonneg hp_mass hq_nonneg hq_mass hsupp s t)

/--
Normalized all-test variational inequality for common-mass vectors, computed from concrete
finite probability measures after dividing by the common mass.
-/
theorem finite_variational_normalized_of_finiteProbabilityMeasure_klDiv_computed_massShell
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = M)
    (hq_pos : ∀ i, 0 < q i)
    (hq_mass : ∑ i, q i = M) :
    ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
      t * (∑ i, (p i / M) * s i)
        - Real.log (∑ i, (q i / M) * Real.exp (t * s i))
        ≤ finiteKL (fun i => p i / M) (fun i => q i / M) := by
  exact finite_variational_of_finiteProbabilityMeasure_klDiv_computed
    (p := fun i => p i / M) (q := fun i => q i / M)
    (normalizedWeight_nonneg (q := p) hMpos hp_nonneg)
    (normalizedWeight_mass (q := p) hMpos hp_mass)
    (normalizedWeight_pos (q := q) hMpos hq_pos)
    (normalizedWeight_mass (q := q) hMpos hq_mass)

/--
Support-aware normalized all-test variational inequality for common-mass vectors, computed from
concrete finite probability measures after dividing by the common mass.
-/
theorem
    finite_variational_normalized_of_finiteProbabilityMeasure_klDiv_computed_massShell_of_support
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = M)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = M)
    (hsupp : ∀ i, q i = 0 → p i = 0) :
    ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
      t * (∑ i, (p i / M) * s i)
        - Real.log (∑ i, (q i / M) * Real.exp (t * s i))
        ≤ finiteKL (fun i => p i / M) (fun i => q i / M) := by
  have hM0 : M ≠ 0 := ne_of_gt hMpos
  have hsupp_norm : ∀ i, q i / M = 0 → p i / M = 0 := by
    intro i hq_norm
    have hq0 : q i = 0 := by
      field_simp [hM0] at hq_norm
      simpa using hq_norm
    rw [hsupp i hq0]
    simp
  exact finite_variational_of_finiteProbabilityMeasure_klDiv_computed_of_support
    (p := fun i => p i / M) (q := fun i => q i / M)
    (normalizedWeight_nonneg (q := p) hMpos hp_nonneg)
    (normalizedWeight_mass (q := p) hMpos hp_mass)
    (normalizedWeight_nonneg (q := q) hMpos hq_nonneg)
    (normalizedWeight_mass (q := q) hMpos hq_mass)
    hsupp_norm

/--
Paper-native common-mass finite variational inequality with computed concrete finite-measure KL.

This exposes the appendix form before probability normalization:
`t⟪p,s⟫ - M log E_{q/M} exp(t s) ≤ KL(p‖q)`.
-/
theorem finite_variational_massShell_of_finiteProbabilityMeasure_klDiv_computed
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = M)
    (hq_pos : ∀ i, 0 < q i)
    (hq_mass : ∑ i, q i = M) :
    ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
      t * (∑ i, p i * s i)
        - M * Real.log (∑ i, (q i / M) * Real.exp (t * s i))
        ≤ finiteKL p q := by
  intro s t ht
  have hM0 : M ≠ 0 := ne_of_gt hMpos
  have hMnonneg : 0 ≤ M := le_of_lt hMpos
  have hvar :=
    finite_variational_normalized_of_finiteProbabilityMeasure_klDiv_computed_massShell
      (p := p) (q := q) hMpos hp_nonneg hp_mass hq_pos hq_mass s t ht
  have hscaled :
      M *
        (t * (∑ i, (p i / M) * s i)
          - Real.log (∑ i, (q i / M) * Real.exp (t * s i)))
        ≤ M * finiteKL (fun i => p i / M) (fun i => q i / M) :=
    mul_le_mul_of_nonneg_left hvar hMnonneg
  have hsum :
      ∑ i, p i * s i = M * ∑ i, (p i / M) * s i := by
    calc
      ∑ i, p i * s i
          = ∑ i, (M * (p i / M)) * s i := by
              refine Finset.sum_congr rfl ?_
              intro i _hi
              have hp : M * (p i / M) = p i := by field_simp [hM0]
              rw [hp]
      _ = ∑ i, M * ((p i / M) * s i) := by
              refine Finset.sum_congr rfl ?_
              intro i _hi
              ring
      _ = M * ∑ i, (p i / M) * s i := by
              rw [Finset.mul_sum]
  have hleft :
      M *
        (t * (∑ i, (p i / M) * s i)
          - Real.log (∑ i, (q i / M) * Real.exp (t * s i)))
        =
      t * (∑ i, p i * s i)
        - M * Real.log (∑ i, (q i / M) * Real.exp (t * s i)) := by
    rw [hsum]
    ring
  have hright :
      M * finiteKL (fun i => p i / M) (fun i => q i / M) = finiteKL p q := by
    exact (finiteKL_scale (p := p) (q := q) hMpos).symm
  calc
    t * (∑ i, p i * s i)
        - M * Real.log (∑ i, (q i / M) * Real.exp (t * s i))
        = M *
            (t * (∑ i, (p i / M) * s i)
              - Real.log (∑ i, (q i / M) * Real.exp (t * s i))) := hleft.symm
    _ ≤ M * finiteKL (fun i => p i / M) (fun i => q i / M) := hscaled
    _ = finiteKL p q := hright

/--
Support-aware paper-native common-mass finite variational inequality.

This is the boundary counterpart of
`finite_variational_massShell_of_finiteProbabilityMeasure_klDiv_computed`: the updated/reference
block `q` may have zero coordinates, provided the source block `p` is also zero on those
coordinates.  This is the natural finite absolute-continuity condition needed by the
measure-theoretic KL variational theorem.
-/
theorem finite_variational_massShell_of_finiteProbabilityMeasure_klDiv_computed_of_support
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = M)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = M)
    (hsupp : ∀ i, q i = 0 → p i = 0) :
    ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
      t * (∑ i, p i * s i)
        - M * Real.log (∑ i, (q i / M) * Real.exp (t * s i))
        ≤ finiteKL p q := by
  intro s t ht
  have hM0 : M ≠ 0 := ne_of_gt hMpos
  have hMnonneg : 0 ≤ M := le_of_lt hMpos
  have hvar :=
    finite_variational_normalized_of_finiteProbabilityMeasure_klDiv_computed_massShell_of_support
      (p := p) (q := q) hMpos hp_nonneg hp_mass hq_nonneg hq_mass hsupp s t ht
  have hscaled :
      M *
        (t * (∑ i, (p i / M) * s i)
          - Real.log (∑ i, (q i / M) * Real.exp (t * s i)))
        ≤ M * finiteKL (fun i => p i / M) (fun i => q i / M) :=
    mul_le_mul_of_nonneg_left hvar hMnonneg
  have hsum :
      ∑ i, p i * s i = M * ∑ i, (p i / M) * s i := by
    calc
      ∑ i, p i * s i
          = ∑ i, (M * (p i / M)) * s i := by
              refine Finset.sum_congr rfl ?_
              intro i _hi
              have hp : M * (p i / M) = p i := by field_simp [hM0]
              rw [hp]
      _ = ∑ i, M * ((p i / M) * s i) := by
              refine Finset.sum_congr rfl ?_
              intro i _hi
              ring
      _ = M * ∑ i, (p i / M) * s i := by
              rw [Finset.mul_sum]
  have hleft :
      M *
        (t * (∑ i, (p i / M) * s i)
          - Real.log (∑ i, (q i / M) * Real.exp (t * s i)))
        =
      t * (∑ i, p i * s i)
        - M * Real.log (∑ i, (q i / M) * Real.exp (t * s i)) := by
    rw [hsum]
    ring
  have hright :
      M * finiteKL (fun i => p i / M) (fun i => q i / M) = finiteKL p q := by
    exact (finiteKL_scale (p := p) (q := q) hMpos).symm
  calc
    t * (∑ i, p i * s i)
        - M * Real.log (∑ i, (q i / M) * Real.exp (t * s i))
        = M *
            (t * (∑ i, (p i / M) * s i)
              - Real.log (∑ i, (q i / M) * Real.exp (t * s i))) := hleft.symm
    _ ≤ M * finiteKL (fun i => p i / M) (fun i => q i / M) := hscaled
    _ = finiteKL p q := hright

/--
Non-normalized Pinsker from signed variational input plus scalar two-point Hoeffding.

This internalizes the only part of the finite Hoeffding theorem needed by the sign-test proof:
the finite distribution is decomposed into its positive/negative sign masses, then the scalar
two-point inequality is applied.
-/
theorem pinsker_nonnormalized_of_signed_variational_twoPoint_hoeffding
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ i, 0 ≤ q i / M)
    (hq_mass : ∑ i, q i / M = 1)
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
          ≤ Real.exp (t * (a - b) + t ^ 2 / 2)) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
  have hexp :
      ∀ t : ℝ, 0 ≤ t →
        ∑ i, (q i / M) *
            Real.exp (t * finiteSign (fun j => p j / M - q j / M) i)
          ≤ Real.exp
              (t * (∑ i, (q i / M) *
                finiteSign (fun j => p j / M - q j / M) i) + t ^ 2 / 2) :=
    finiteSign_exp_mgf_bound_of_twoPoint_hoeffding
      (q := fun i => q i / M)
      (v := fun j => p j / M - q j / M)
      hq_nonneg hq_mass htwoPoint
  exact pinsker_nonnormalized_of_signed_variational_exp_mgf
    (p := p) (q := q) hMpos hq_nonneg hq_mass hvar hexp

/--
Non-normalized Pinsker from a finite all-test variational inequality plus scalar two-point
Hoeffding.

Compared with `pinsker_nonnormalized_of_signed_variational_twoPoint_hoeffding`, this theorem no
longer assumes the variational inequality already specialized to the sign test.
-/
theorem pinsker_nonnormalized_of_finite_variational_twoPoint_hoeffding
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ i, 0 ≤ q i / M)
    (hq_mass : ∑ i, q i / M = 1)
    (hvarAll :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p i / M) * s i)
          - Real.log (∑ i, (q i / M) * Real.exp (t * s i))
          ≤ finiteKL (fun i => p i / M) (fun i => q i / M))
    (htwoPoint :
      ∀ a b t : ℝ, 0 ≤ a → 0 ≤ b → a + b = 1 → 0 ≤ t →
        a * Real.exp t + b * Real.exp (-t)
          ≤ Real.exp (t * (a - b) + t ^ 2 / 2)) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
  exact pinsker_nonnormalized_of_signed_variational_twoPoint_hoeffding
    (p := p) (q := q) hMpos hq_nonneg hq_mass
    (signed_variational_normalized_of_finite_variational
      (p := p) (q := q) (M := M) hvarAll)
    htwoPoint

/--
Non-normalized Pinsker from a finite all-test variational inequality plus the centered Bernoulli
scalar Hoeffding inequality.

This is stronger than the two-point interface: Lean now proves the algebraic conversion from the
centered Bernoulli MGF form to the sign-variable two-point MGF form internally.
-/
theorem pinsker_nonnormalized_of_finite_variational_centeredBernoulli_hoeffding
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ i, 0 ≤ q i / M)
    (hq_mass : ∑ i, q i / M = 1)
    (hvarAll :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p i / M) * s i)
          - Real.log (∑ i, (q i / M) * Real.exp (t * s i))
          ≤ finiteKL (fun i => p i / M) (fun i => q i / M))
    (hcentered :
      ∀ a b t : ℝ, 0 ≤ a → 0 ≤ b → a + b = 1 → 0 ≤ t →
        a * Real.exp (2 * b * t) + b * Real.exp (-2 * a * t)
          ≤ Real.exp (t ^ 2 / 2)) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
  exact pinsker_nonnormalized_of_finite_variational_twoPoint_hoeffding
    (p := p) (q := q) hMpos hq_nonneg hq_mass hvarAll
    (twoPoint_hoeffding_of_centeredBernoulli_hoeffding hcentered)

/--
Mass-shell version of the two-point-Hoeffding A.3 endpoint.

Compared with the bounded-test endpoint, this no longer assumes a general Hoeffding theorem for
all bounded finite tests.  It derives the exact sign-test MGF from the two sign masses and only
uses the scalar two-point inequality.
-/
theorem pinsker_nonnormalized_of_signed_variational_twoPoint_hoeffding_massShell
    {p q : Fin n → ℝ} {M : ℝ}
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
          ≤ Real.exp (t * (a - b) + t ^ 2 / 2)) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
  exact pinsker_nonnormalized_of_signed_variational_twoPoint_hoeffding
    (p := p) (q := q) hMpos
    (normalizedWeight_nonneg (q := q) hMpos hq_nonneg)
    (normalizedWeight_mass (q := q) hMpos hq_mass)
    hvar htwoPoint

/--
Mass-shell non-normalized Pinsker from a finite all-test variational inequality plus scalar
two-point Hoeffding.

This is the strongest current finite A.3 endpoint: the sign-test variational premise is derived
internally from a reusable all-test variational inequality, and the sign-test MGF is reduced
internally to two scalar masses.
-/
theorem pinsker_nonnormalized_of_finite_variational_twoPoint_hoeffding_massShell
    {p q : Fin n → ℝ} {M : ℝ}
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
          ≤ Real.exp (t * (a - b) + t ^ 2 / 2)) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
  exact pinsker_nonnormalized_of_finite_variational_twoPoint_hoeffding
    (p := p) (q := q) hMpos
    (normalizedWeight_nonneg (q := q) hMpos hq_nonneg)
    (normalizedWeight_mass (q := q) hMpos hq_mass)
    hvarAll htwoPoint

/--
Mass-shell non-normalized Pinsker from all-test finite variational input plus centered Bernoulli
Hoeffding.

This is the strongest current finite A.3 endpoint: the sign-test variational premise, the
two-point sign-mass decomposition, and the recentering algebra from Bernoulli Hoeffding are all
certified internally.
-/
theorem pinsker_nonnormalized_of_finite_variational_centeredBernoulli_hoeffding_massShell
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = M)
    (hvarAll :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p i / M) * s i)
          - Real.log (∑ i, (q i / M) * Real.exp (t * s i))
          ≤ finiteKL (fun i => p i / M) (fun i => q i / M))
    (hcentered :
      ∀ a b t : ℝ, 0 ≤ a → 0 ≤ b → a + b = 1 → 0 ≤ t →
        a * Real.exp (2 * b * t) + b * Real.exp (-2 * a * t)
          ≤ Real.exp (t ^ 2 / 2)) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
  exact pinsker_nonnormalized_of_finite_variational_centeredBernoulli_hoeffding
    (p := p) (q := q) hMpos
    (normalizedWeight_nonneg (q := q) hMpos hq_nonneg)
    (normalizedWeight_mass (q := q) hMpos hq_mass)
    hvarAll hcentered

/--
Mass-shell non-normalized Pinsker from all-test finite variational input plus the strict-interior
centered Bernoulli Hoeffding inequality.

This is a slightly sharper A.3 interface: the endpoint proves internally that the Bernoulli
Hoeffding bound is automatic when one sign mass is zero or when `t = 0`; the remaining scalar
analytic input is only the genuinely interior case `0 < a`, `0 < b`, `0 < t`.
-/
theorem pinsker_nonnormalized_of_finite_variational_centeredBernoulliInterior_hoeffding_massShell
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = M)
    (hvarAll :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p i / M) * s i)
          - Real.log (∑ i, (q i / M) * Real.exp (t * s i))
          ≤ finiteKL (fun i => p i / M) (fun i => q i / M))
    (hcenteredInterior :
      ∀ a b t : ℝ, 0 < a → 0 < b → a + b = 1 → 0 < t →
        a * Real.exp (2 * b * t) + b * Real.exp (-2 * a * t)
          ≤ Real.exp (t ^ 2 / 2)) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
  exact pinsker_nonnormalized_of_finite_variational_centeredBernoulli_hoeffding_massShell
    (p := p) (q := q) hMpos hq_nonneg hq_mass hvarAll
    (centeredBernoulli_hoeffding_of_pos_pos_pos hcenteredInterior)

/--
Mass-shell non-normalized Pinsker from all-test finite variational input plus the one-parameter
strict-interior centered Bernoulli Hoeffding inequality.

This is the sharpest current A.3 interface: all degenerate Bernoulli cases and the reduction from
two masses `a,b` with `a+b=1` to the one-parameter form `b=1-a` are certified internally.
-/
theorem pinsker_nonnormalized_of_finite_variational_centeredBernoulliUnit_hoeffding_massShell
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = M)
    (hvarAll :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p i / M) * s i)
          - Real.log (∑ i, (q i / M) * Real.exp (t * s i))
          ≤ finiteKL (fun i => p i / M) (fun i => q i / M))
    (hcenteredUnit :
      ∀ a t : ℝ, 0 < a → a < 1 → 0 < t →
        a * Real.exp (2 * (1 - a) * t) + (1 - a) * Real.exp (-2 * a * t)
          ≤ Real.exp (t ^ 2 / 2)) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
  exact pinsker_nonnormalized_of_finite_variational_centeredBernoulliInterior_hoeffding_massShell
    (p := p) (q := q) hMpos hq_nonneg hq_mass hvarAll
    (centeredBernoulli_interior_hoeffding_of_unitInterval hcenteredUnit)

/--
Mass-shell non-normalized Pinsker from the paper-native common-mass variational inequality and
the one-parameter strict-interior centered Bernoulli Hoeffding inequality.

Compared with the normalized-variational endpoint, this theorem no longer asks the caller to
pre-normalize the all-test variational inequality: the normalized finite-KL input is derived
internally from `finiteKL_scale`.
-/
theorem pinsker_nonnormalized_of_massShell_variational_centeredBernoulliUnit_hoeffding
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = M)
    (hvarMass :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p i * s i)
          - M * Real.log (∑ i, (q i / M) * Real.exp (t * s i))
          ≤ finiteKL p q)
    (hcenteredUnit :
      ∀ a t : ℝ, 0 < a → a < 1 → 0 < t →
        a * Real.exp (2 * (1 - a) * t) + (1 - a) * Real.exp (-2 * a * t)
          ≤ Real.exp (t ^ 2 / 2)) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
  exact pinsker_nonnormalized_of_finite_variational_centeredBernoulliUnit_hoeffding_massShell
    (p := p) (q := q) hMpos hq_nonneg hq_mass
    (finite_variational_normalized_of_massShell_variational (p := p) (q := q) hMpos hvarMass)
    hcenteredUnit

/--
Mass-shell non-normalized Pinsker from the paper-native common-mass variational inequality.

Compared with `pinsker_nonnormalized_of_massShell_variational_centeredBernoulliUnit_hoeffding`,
this endpoint no longer assumes the scalar centered Bernoulli Hoeffding inequality: it is derived
internally from mathlib's measure-theoretic Hoeffding lemma instantiated on the two-point
Bernoulli probability space.
-/
theorem pinsker_nonnormalized_of_massShell_variational_mathlib_hoeffding
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = M)
    (hvarMass :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p i * s i)
          - M * Real.log (∑ i, (q i / M) * Real.exp (t * s i))
          ≤ finiteKL p q) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
  exact pinsker_nonnormalized_of_massShell_variational_centeredBernoulliUnit_hoeffding
    (p := p) (q := q) hMpos hq_nonneg hq_mass hvarMass
    (fun a t ha ha_lt ht =>
      centeredBernoulli_unit_hoeffding_of_mathlib a t ha.le ha_lt.le ht.le)

/--
Non-normalized Pinsker from a normalized all-test finite variational inequality, with the scalar
Hoeffding step discharged by mathlib.

This endpoint is useful when the caller has already normalized the common-mass vectors.  It avoids
rerouting through the paper-native mass-shell variational statement while keeping the same certified
Pinsker/Hoefdding proof chain.
-/
theorem pinsker_nonnormalized_of_finite_variational_mathlib_hoeffding_massShell
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = M)
    (hvarAll :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p i / M) * s i)
          - Real.log (∑ i, (q i / M) * Real.exp (t * s i))
          ≤ finiteKL (fun i => p i / M) (fun i => q i / M)) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
  exact pinsker_nonnormalized_of_finite_variational_centeredBernoulliUnit_hoeffding_massShell
    (p := p) (q := q) hMpos hq_nonneg hq_mass hvarAll
    (fun a t ha ha_lt ht =>
      centeredBernoulli_unit_hoeffding_of_mathlib a t ha.le ha_lt.le ht.le)

/--
Mass-shell A.3 endpoint using the concrete finite-measure variational bridge.

Compared with `pinsker_nonnormalized_of_massShell_variational_mathlib_hoeffding`,
this theorem no longer assumes the common-mass all-test variational inequality.
It derives the normalized all-test inequality from the probability vectors
`p/M`, `q/M` via `finiteProbabilityMeasure_variational_lower_bound`; the only
remaining compatibility premise is that mathlib's KL divergence for these
concrete finite measures is bounded by the handwritten normalized `finiteKL`.
-/
theorem pinsker_nonnormalized_of_finiteProbabilityMeasure_klDiv_mathlib_hoeffding_massShell
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = M)
    (hq_pos : ∀ i, 0 < q i)
    (hq_mass : ∑ i, q i = M)
    (hkl_le :
      (InformationTheory.klDiv
        (finiteProbabilityMeasure (fun i => p i / M))
        (finiteProbabilityMeasure (fun i => q i / M))).toReal
          ≤ finiteKL (fun i => p i / M) (fun i => q i / M)) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
  have hp_norm_nonneg : ∀ i, 0 ≤ p i / M :=
    normalizedWeight_nonneg (q := p) hMpos hp_nonneg
  have hp_norm_mass : ∑ i, p i / M = 1 :=
    normalizedWeight_mass (q := p) hMpos hp_mass
  have hq_norm_pos : ∀ i, 0 < q i / M := by
    intro i
    exact div_pos (hq_pos i) hMpos
  have hq_norm_mass : ∑ i, q i / M = 1 :=
    normalizedWeight_mass (q := q) hMpos hq_mass
  have hvarAll :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, (p i / M) * s i)
          - Real.log (∑ i, (q i / M) * Real.exp (t * s i))
          ≤ finiteKL (fun i => p i / M) (fun i => q i / M) :=
    finite_variational_of_finiteProbabilityMeasure_klDiv_le_finiteKL
      hp_norm_nonneg hp_norm_mass hq_norm_pos hq_norm_mass hkl_le
  exact pinsker_nonnormalized_of_finite_variational_mathlib_hoeffding_massShell
    (p := p) (q := q) hMpos (fun i => (hq_pos i).le) hq_mass hvarAll

/--
Equality-shaped mass-shell endpoint for callers that establish the exact finite
KL identity for the concrete probability measures.
-/
theorem pinsker_nonnormalized_of_finiteProbabilityMeasure_klDiv_eq_mathlib_hoeffding_massShell
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = M)
    (hq_pos : ∀ i, 0 < q i)
    (hq_mass : ∑ i, q i = M)
    (hkl_eq :
      (InformationTheory.klDiv
        (finiteProbabilityMeasure (fun i => p i / M))
        (finiteProbabilityMeasure (fun i => q i / M))).toReal
          = finiteKL (fun i => p i / M) (fun i => q i / M)) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
  exact pinsker_nonnormalized_of_finiteProbabilityMeasure_klDiv_mathlib_hoeffding_massShell
    (p := p) (q := q) hMpos hp_nonneg hp_mass hq_pos hq_mass (le_of_eq hkl_eq)

/--
Mass-shell A.3 endpoint with the concrete finite-measure KL computation discharged internally.

This is the premise-free version of
`pinsker_nonnormalized_of_finiteProbabilityMeasure_klDiv_eq_mathlib_hoeffding_massShell` under the
same strict positivity assumptions: the normalized mathlib `klDiv = finiteKL` identity is now
proved by `finiteProbabilityMeasure_klDiv_eq_finiteKL`.
-/
theorem pinsker_nonnormalized_of_finiteProbabilityMeasure_klDiv_computed_mathlib_hoeffding_massShell
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = M)
    (hq_pos : ∀ i, 0 < q i)
    (hq_mass : ∑ i, q i = M) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
  have hp_norm_nonneg : ∀ i, 0 ≤ p i / M :=
    normalizedWeight_nonneg (q := p) hMpos hp_nonneg
  have hp_norm_mass : ∑ i, p i / M = 1 :=
    normalizedWeight_mass (q := p) hMpos hp_mass
  have hq_norm_pos : ∀ i, 0 < q i / M := by
    intro i
    exact div_pos (hq_pos i) hMpos
  have hq_norm_mass : ∑ i, q i / M = 1 :=
    normalizedWeight_mass (q := q) hMpos hq_mass
  have hkl_eq :
      (InformationTheory.klDiv
        (finiteProbabilityMeasure (fun i => p i / M))
        (finiteProbabilityMeasure (fun i => q i / M))).toReal
          = finiteKL (fun i => p i / M) (fun i => q i / M) :=
    finiteProbabilityMeasure_klDiv_eq_finiteKL
      hp_norm_nonneg hp_norm_mass hq_norm_pos hq_norm_mass
  exact pinsker_nonnormalized_of_finiteProbabilityMeasure_klDiv_eq_mathlib_hoeffding_massShell
    (p := p) (q := q) hMpos hp_nonneg hp_mass hq_pos hq_mass hkl_eq

/--
Probability-mass Pinsker from a normalized all-test finite variational inequality, with Hoeffding
discharged by mathlib.

This is the non-normalized theorem specialized to mass `M = 1`; it is intentionally separate from
`normalizedPinsker_of_finite_variational_mathlib_hoeffding` because downstream code sometimes wants
the denominator to remain visibly in the common-mass format before simplification.
-/
theorem pinsker_probabilityMass_of_finite_variational_mathlib_hoeffding
    {p q : Fin n → ℝ}
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = 1)
    (hvarAll :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p i * s i)
          - Real.log (∑ i, q i * Real.exp (t * s i))
          ≤ finiteKL p q) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * 1) := by
  simpa using
    (pinsker_nonnormalized_of_finite_variational_mathlib_hoeffding_massShell
      (p := p) (q := q) (M := 1)
      (by norm_num) hq_nonneg hq_mass (by simpa using hvarAll))

/--
Normalized finite Pinsker from the all-test finite variational inequality and mathlib Hoeffding.

This is the paper-facing endpoint for Proposition `app-prop:pinsker-normalized`.  The remaining
finite variational premise is the finite-dimensional version of the KL variational representation;
the Hoeffding/sign-test part is fully internalized through
`centeredBernoulli_unit_hoeffding_of_mathlib`.
-/
theorem normalizedPinsker_of_finite_variational_mathlib_hoeffding
    {p q : Fin n → ℝ}
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = 1)
    (hvarAll :
      ∀ s : Fin n → ℝ, ∀ t : ℝ, 0 ≤ t →
        t * (∑ i, p i * s i)
          - Real.log (∑ i, q i * Real.exp (t * s i))
          ≤ finiteKL p q) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / 2 := by
  have hpin :=
    pinsker_probabilityMass_of_finite_variational_mathlib_hoeffding
      (p := p) (q := q) hq_nonneg hq_mass hvarAll
  simpa using hpin

/--
Normalized finite Pinsker from the concrete finite-measure variational bridge and a caller-supplied
upper comparison from mathlib `klDiv` to the explicit finite KL.
-/
theorem normalizedPinsker_of_finiteProbabilityMeasure_klDiv_le_mathlib_hoeffding
    {p q : Fin n → ℝ}
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = 1)
    (hq_pos : ∀ i, 0 < q i)
    (hq_mass : ∑ i, q i = 1)
    (hkl_le :
      (InformationTheory.klDiv
        (finiteProbabilityMeasure p) (finiteProbabilityMeasure q)).toReal
          ≤ finiteKL p q) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / 2 := by
  exact normalizedPinsker_of_finite_variational_mathlib_hoeffding
    (p := p) (q := q) (fun i => (hq_pos i).le) hq_mass
    (finite_variational_of_finiteProbabilityMeasure_klDiv_le_finiteKL
      hp_nonneg hp_mass hq_pos hq_mass hkl_le)

/--
Normalized finite Pinsker from the concrete finite-measure variational bridge and an exact
mathlib-`klDiv`/finite-KL identity.
-/
theorem normalizedPinsker_of_finiteProbabilityMeasure_klDiv_eq_mathlib_hoeffding
    {p q : Fin n → ℝ}
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = 1)
    (hq_pos : ∀ i, 0 < q i)
    (hq_mass : ∑ i, q i = 1)
    (hkl_eq :
      (InformationTheory.klDiv
        (finiteProbabilityMeasure p) (finiteProbabilityMeasure q)).toReal
          = finiteKL p q) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / 2 := by
  exact normalizedPinsker_of_finiteProbabilityMeasure_klDiv_le_mathlib_hoeffding
    hp_nonneg hp_mass hq_pos hq_mass (le_of_eq hkl_eq)

/--
Premise-minimal normalized finite Pinsker endpoint using computed mathlib KL and mathlib
Hoeffding internally.
-/
theorem normalizedPinsker_of_finiteProbabilityMeasure_klDiv_computed_mathlib_hoeffding
    {p q : Fin n → ℝ}
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = 1)
    (hq_pos : ∀ i, 0 < q i)
    (hq_mass : ∑ i, q i = 1) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / 2 := by
  exact normalizedPinsker_of_finiteProbabilityMeasure_klDiv_eq_mathlib_hoeffding
    hp_nonneg hp_mass hq_pos hq_mass
    (finiteProbabilityMeasure_klDiv_eq_finiteKL hp_nonneg hp_mass hq_pos hq_mass)

/--
Premise-minimal normalized finite Pinsker endpoint under support domination.

This boundary endpoint keeps the computed concrete-measure KL and mathlib Hoeffding route, but
allows zero coordinates in `q` whenever `p` is also zero there.
-/
theorem normalizedPinsker_of_finiteProbabilityMeasure_klDiv_computed_mathlib_hoeffding_of_support
    {p q : Fin n → ℝ}
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = 1)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = 1)
    (hsupp : ∀ i, q i = 0 → p i = 0) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / 2 := by
  exact normalizedPinsker_of_finite_variational_mathlib_hoeffding
    (p := p) (q := q) hq_nonneg hq_mass
    (finite_variational_of_finiteProbabilityMeasure_klDiv_computed_of_support
      hp_nonneg hp_mass hq_nonneg hq_mass hsupp)

/--
Probability-mass form of the computed finite-measure Pinsker endpoint.

The conclusion keeps the denominator as `2 * 1`, matching the common-mass statement specialized
to probability mass.
-/
theorem pinsker_probabilityMass_of_finiteProbabilityMeasure_klDiv_computed_mathlib_hoeffding
    {p q : Fin n → ℝ}
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = 1)
    (hq_pos : ∀ i, 0 < q i)
    (hq_mass : ∑ i, q i = 1) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * 1) := by
  have hpin :=
    normalizedPinsker_of_finiteProbabilityMeasure_klDiv_computed_mathlib_hoeffding
      (p := p) (q := q) hp_nonneg hp_mass hq_pos hq_mass
  simpa using hpin

/--
Common-mass Pinsker endpoint routed through the computed paper-native mass-shell variational
inequality and mathlib Hoeffding.
-/
theorem pinsker_nonnormalized_of_massShell_finiteProbabilityMeasure_klDiv_computed_mathlib_hoeffding
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = M)
    (hq_pos : ∀ i, 0 < q i)
    (hq_mass : ∑ i, q i = M) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
  exact pinsker_nonnormalized_of_massShell_variational_mathlib_hoeffding
    (p := p) (q := q) hMpos (fun i => (hq_pos i).le) hq_mass
    (finite_variational_massShell_of_finiteProbabilityMeasure_klDiv_computed
      (p := p) (q := q) hMpos hp_nonneg hp_mass hq_pos hq_mass)

/--
Common-mass Pinsker endpoint under support domination, with concrete finite-measure KL and
mathlib Hoeffding discharged internally.

This is the boundary counterpart of
`pinsker_nonnormalized_of_massShell_finiteProbabilityMeasure_klDiv_computed_mathlib_hoeffding`:
`q` may vanish on coordinates where `p` also vanishes.
-/
theorem
    pinsker_nonnormalized_of_finiteProbabilityMeasure_klDiv_computed_support
    {p q : Fin n → ℝ} {M : ℝ}
    (hMpos : 0 < M)
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = M)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = M)
    (hsupp : ∀ i, q i = 0 → p i = 0) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
  exact pinsker_nonnormalized_of_finite_variational_mathlib_hoeffding_massShell
    (p := p) (q := q) hMpos hq_nonneg hq_mass
    (finite_variational_normalized_of_finiteProbabilityMeasure_klDiv_computed_massShell_of_support
      (p := p) (q := q) hMpos hp_nonneg hp_mass hq_nonneg hq_mass hsupp)

/--
Mass-shell version of the bounded-Hoeffding A.3 endpoint.

This is the closest current interface to the paper statement: nonnegativity and unit mass for
the normalized `q/M` weights are derived internally from `q_i ≥ 0`, `∑ q_i = M`, and `M > 0`.
-/
theorem pinsker_nonnormalized_of_signed_variational_bounded_hoeffding_massShell
    {p q : Fin n → ℝ} {M : ℝ}
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
            ≤ Real.exp (t * (∑ i, (q i / M) * s i) + t ^ 2 / 2)) :
    finiteKL p q ≥ (l1Norm (fun i => p i - q i)) ^ 2 / (2 * M) := by
  exact pinsker_nonnormalized_of_signed_variational_bounded_hoeffding
    (p := p) (q := q) hMpos
    (normalizedWeight_nonneg (q := q) hMpos hq_nonneg)
    (normalizedWeight_mass (q := q) hMpos hq_mass)
    hvar hhoeffding

end DualConvergence
end KLProjection
end FlowSinkhorn
