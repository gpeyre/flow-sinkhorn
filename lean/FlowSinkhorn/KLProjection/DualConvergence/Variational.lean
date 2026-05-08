import Mathlib.InformationTheory.KullbackLeibler.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.MeasureTheory.Measure.LogLikelihoodRatio
import Mathlib.MeasureTheory.Measure.Tilted

/-!
# Variational KL lower bounds

This module provides a reusable variational lower bound for KL divergence in the
probability-measure setting:

`E_μ[f] - log E_ν[e^f] ≤ KL(μ‖ν)`.

It is a key technical ingredient for internalizing normalized Pinsker-type bounds.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace DualConvergence

open MeasureTheory
open scoped BigOperators ENNReal

variable {α : Type*} [MeasurableSpace α]
variable {μ ν : Measure α} {f : α → ℝ}

theorem klDiv_variational_lower_bound
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν] [SigmaFinite μ] [SigmaFinite ν]
    (hμν : μ ≪ ν)
    (hf : Integrable f μ)
    (hfν : Integrable (fun x ↦ Real.exp (f x)) ν)
    (hllr : Integrable (llr μ ν) μ) :
    (∫ x, f x ∂μ) - Real.log (∫ x, Real.exp (f x) ∂ν) ≤ (InformationTheory.klDiv μ ν).toReal := by
  have hμν_tilt : μ ≪ ν.tilted f :=
    hμν.trans (absolutelyContinuous_tilted (μ := ν) (f := f) hfν)
  have hllr_tilt : Integrable (llr μ (ν.tilted f)) μ :=
    integrable_llr_tilted_right (μ := μ) (ν := ν) hμν hf hllr hfν
  have hnonneg_tilt : 0 ≤ ∫ x, llr μ (ν.tilted f) x ∂μ := by
    have hnonneg_toReal : 0 ≤ (InformationTheory.klDiv μ (ν.tilted f)).toReal :=
      ENNReal.toReal_nonneg
    have hkl_eq :
        (InformationTheory.klDiv μ (ν.tilted f)).toReal = ∫ x, llr μ (ν.tilted f) x ∂μ := by
      have hmass : μ Set.univ = (ν.tilted f) Set.univ := by
        simp [isProbabilityMeasure_tilted (μ := ν) hfν]
      simpa [hmass] using
        (InformationTheory.toReal_klDiv_of_measure_eq
          (μ := μ) (ν := ν.tilted f) hμν_tilt hmass)
    exact hkl_eq ▸ hnonneg_toReal
  have htilt :
      ∫ x, llr μ (ν.tilted f) x ∂μ
        = ∫ x, llr μ ν x ∂μ - ∫ x, f x ∂μ + Real.log (∫ x, Real.exp (f x) ∂ν) :=
    integral_llr_tilted_right (μ := μ) (ν := ν) hμν hf hfν hllr
  have hkl :
      (InformationTheory.klDiv μ ν).toReal = ∫ x, llr μ ν x ∂μ := by
    have hmass : μ Set.univ = ν Set.univ := by simp
    simpa [hmass] using
      (InformationTheory.toReal_klDiv_of_measure_eq (μ := μ) (ν := ν) hμν hmass)
  have haux :
      (∫ x, f x ∂μ) - Real.log (∫ x, Real.exp (f x) ∂ν) ≤ ∫ x, llr μ ν x ∂μ := by
    linarith [hnonneg_tilt, htilt]
  exact haux.trans (by simp [hkl])

section Finite

variable {n : ℕ}

/--
Concrete finite measure associated with a real weight vector.

For nonnegative weights summing to one, this is the probability measure used to
turn the measure-theoretic KL variational formula into the finite all-test
inequality in Appendix A.
-/
noncomputable def finiteProbabilityMeasure (p : Fin n → ℝ) : Measure (Fin n) :=
  ∑ i, ENNReal.ofReal (p i) • Measure.dirac i

/-- Integrals against `finiteProbabilityMeasure` are the expected finite weighted sums. -/
theorem finiteProbabilityMeasure_integral
    (p f : Fin n → ℝ)
    (hp : ∀ i, 0 ≤ p i) :
    ∫ i, f i ∂finiteProbabilityMeasure p = ∑ i, p i * f i := by
  dsimp [finiteProbabilityMeasure]
  rw [MeasureTheory.integral_finset_sum_measure]
  · simp only [MeasureTheory.integral_smul_measure, MeasureTheory.integral_dirac, smul_eq_mul]
    refine Finset.sum_congr rfl ?_
    intro i _hi
    rw [ENNReal.toReal_ofReal (hp i)]
  · intro i _hi
    exact
      (MeasureTheory.integrable_dirac (by simp) :
        MeasureTheory.Integrable f (MeasureTheory.Measure.dirac i)).smul_measure
          ENNReal.ofReal_ne_top

/-- Total mass of the concrete finite measure. -/
theorem finiteProbabilityMeasure_apply_univ
    (p : Fin n → ℝ)
    (hp : ∀ i, 0 ≤ p i) :
    finiteProbabilityMeasure p Set.univ = ENNReal.ofReal (∑ i, p i) := by
  dsimp [finiteProbabilityMeasure]
  calc
    (∑ i, ENNReal.ofReal (p i) • Measure.dirac i) Set.univ
        = ∑ i, ENNReal.ofReal (p i) := by simp
    _ = ENNReal.ofReal (∑ i, p i) := by
      rw [ENNReal.ofReal_sum_of_nonneg]
      intro i _hi
      exact hp i

/-- The concrete finite measure has finite total mass for every real weight vector. -/
theorem finiteProbabilityMeasure_isFiniteMeasure (p : Fin n → ℝ) :
    IsFiniteMeasure (finiteProbabilityMeasure p) := by
  refine ⟨?_⟩
  dsimp [finiteProbabilityMeasure]
  simp

/-- A nonnegative finite vector of mass one defines a probability measure. -/
theorem finiteProbabilityMeasure_isProbability
    (p : Fin n → ℝ)
    (hp : ∀ i, 0 ≤ p i)
    (hmass : ∑ i, p i = 1) :
    IsProbabilityMeasure (finiteProbabilityMeasure p) := by
  refine ⟨?_⟩
  rw [finiteProbabilityMeasure_apply_univ p hp, hmass]
  norm_num

/-- Singleton masses of the concrete finite measure. -/
theorem finiteProbabilityMeasure_apply_singleton
    (p : Fin n → ℝ) (i : Fin n) :
    finiteProbabilityMeasure p {i} = ENNReal.ofReal (p i) := by
  dsimp [finiteProbabilityMeasure]
  simp only [Measure.coe_finset_sum, Finset.sum_apply, Measure.smul_apply, smul_eq_mul]
  rw [Finset.sum_eq_single i]
  · simp
  · intro j _hj hji
    simp [hji]
  · intro hi
    simp at hi

/--
Atomic density representation of the concrete finite measures.

Strict positivity of `q` removes support pathologies: the measure generated by `p`
is obtained from the measure generated by `q` by weighting atom `i` with `p i / q i`
(with `ENNReal.ofReal` clipping, matching the measure construction).
-/
theorem finiteProbabilityMeasure_eq_withDensity
    {p q : Fin n → ℝ}
    (hq_pos : ∀ i, 0 < q i) :
    finiteProbabilityMeasure p =
      (finiteProbabilityMeasure q).withDensity (fun i => ENNReal.ofReal (p i / q i)) := by
  refine Measure.ext_of_singleton ?_
  intro i
  rw [finiteProbabilityMeasure_apply_singleton]
  rw [withDensity_apply _ (MeasurableSet.singleton i)]
  rw [lintegral_singleton]
  rw [finiteProbabilityMeasure_apply_singleton]
  rw [ENNReal.ofReal_div_of_pos (hq_pos i)]
  exact (ENNReal.div_mul_cancel
    (by exact (ENNReal.ofReal_pos.mpr (hq_pos i)).ne') (by simp)).symm

/--
Atomic density representation under the natural support condition.

This is the boundary version of `finiteProbabilityMeasure_eq_withDensity`: zero atoms of
the reference are allowed, provided `p` gives them zero mass too.  On zero reference atoms the
chosen density value is irrelevant; Lean's real division sets `0 / 0 = 0`, matching the
with-density measure because the reference atom has mass zero.
-/
theorem finiteProbabilityMeasure_eq_withDensity_of_support
    {p q : Fin n → ℝ}
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hsupp : ∀ i, q i = 0 → p i = 0) :
    finiteProbabilityMeasure p =
      (finiteProbabilityMeasure q).withDensity (fun i => ENNReal.ofReal (p i / q i)) := by
  refine Measure.ext_of_singleton ?_
  intro i
  rw [finiteProbabilityMeasure_apply_singleton]
  rw [withDensity_apply _ (MeasurableSet.singleton i)]
  rw [lintegral_singleton]
  rw [finiteProbabilityMeasure_apply_singleton]
  by_cases hq0 : q i = 0
  · have hp0 : p i = 0 := hsupp i hq0
    simp [hp0, hq0]
  · have hq_pos : 0 < q i := lt_of_le_of_ne (hq_nonneg i) (Ne.symm hq0)
    rw [ENNReal.ofReal_div_of_pos hq_pos]
    exact (ENNReal.div_mul_cancel
      (by exact (ENNReal.ofReal_pos.mpr hq_pos).ne') (by simp)).symm

/-- Radon-Nikodym derivative of the concrete finite measures under strict reference positivity. -/
theorem finiteProbabilityMeasure_rnDeriv_ae
    {p q : Fin n → ℝ}
    (hq_pos : ∀ i, 0 < q i) :
    (finiteProbabilityMeasure p).rnDeriv (finiteProbabilityMeasure q)
      =ᵐ[finiteProbabilityMeasure q] fun i => ENNReal.ofReal (p i / q i) := by
  haveI : IsFiniteMeasure (finiteProbabilityMeasure q) := finiteProbabilityMeasure_isFiniteMeasure q
  rw [finiteProbabilityMeasure_eq_withDensity (p := p) (q := q) hq_pos]
  exact Measure.rnDeriv_withDensity (finiteProbabilityMeasure q) (by fun_prop)

/-- Radon-Nikodym derivative of concrete finite measures under support domination. -/
theorem finiteProbabilityMeasure_rnDeriv_ae_of_support
    {p q : Fin n → ℝ}
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hsupp : ∀ i, q i = 0 → p i = 0) :
    (finiteProbabilityMeasure p).rnDeriv (finiteProbabilityMeasure q)
      =ᵐ[finiteProbabilityMeasure q] fun i => ENNReal.ofReal (p i / q i) := by
  haveI : IsFiniteMeasure (finiteProbabilityMeasure q) := finiteProbabilityMeasure_isFiniteMeasure q
  rw [finiteProbabilityMeasure_eq_withDensity_of_support
    (p := p) (q := q) hq_nonneg hsupp]
  exact Measure.rnDeriv_withDensity (finiteProbabilityMeasure q) (by fun_prop)

/--
If the reference finite vector is strictly positive, every finite measure is absolutely
continuous with respect to its concrete measure.
-/
theorem finiteProbabilityMeasure_absolutelyContinuous
    (p q : Fin n → ℝ)
    (hq : ∀ i, 0 < q i) :
    finiteProbabilityMeasure p ≪ finiteProbabilityMeasure q := by
  intro s hs
  by_cases hs_empty : s = ∅
  · simp [hs_empty]
  · have hex : ∃ i, i ∈ s := Set.nonempty_iff_ne_empty.mpr hs_empty
    rcases hex with ⟨i, hi⟩
    have hmono : finiteProbabilityMeasure q {i} ≤ finiteProbabilityMeasure q s := by
      exact measure_mono (by
        intro x hx
        rw [Set.mem_singleton_iff] at hx
        simpa [hx] using hi)
    have hzero : finiteProbabilityMeasure q {i} = 0 := by
      exact le_antisymm (by simpa [hs] using hmono) bot_le
    have hpos : 0 < finiteProbabilityMeasure q {i} := by
      rw [finiteProbabilityMeasure_apply_singleton q i]
      exact ENNReal.ofReal_pos.mpr (hq i)
    exact False.elim ((ne_of_gt hpos) hzero)

/--
Support-aware absolute continuity for concrete finite measures.

This is the finite discrete condition `p_i = 0` whenever `q_i = 0`, expressed as
absolute continuity of the associated concrete measures.
-/
theorem finiteProbabilityMeasure_absolutelyContinuous_of_support
    (p q : Fin n → ℝ)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hsupp : ∀ i, q i = 0 → p i = 0) :
    finiteProbabilityMeasure p ≪ finiteProbabilityMeasure q := by
  rw [finiteProbabilityMeasure_eq_withDensity_of_support
    (p := p) (q := q) hq_nonneg hsupp]
  exact withDensity_absolutelyContinuous _ _

/--
Finite all-test variational inequality from the concrete finite probability measures.

This is the finite bridge corresponding to the appendix line
`E_p f - log E_q e^f ≤ KL(p‖q)`, obtained by applying the reusable
measure-theoretic theorem to the measures generated by the two probability
vectors.  The right-hand side is still mathlib's `klDiv`; `Pinsker.lean`
contains the small bridge that compares it to the handwritten `finiteKL`.
-/
theorem finiteProbabilityMeasure_variational_lower_bound
    {p q : Fin n → ℝ}
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = 1)
    (hq_pos : ∀ i, 0 < q i)
    (hq_mass : ∑ i, q i = 1) :
    ∀ s : Fin n → ℝ, ∀ t : ℝ,
      t * (∑ i, p i * s i)
        - Real.log (∑ i, q i * Real.exp (t * s i))
        ≤ (InformationTheory.klDiv
            (finiteProbabilityMeasure p) (finiteProbabilityMeasure q)).toReal := by
  intro s t
  let μ := finiteProbabilityMeasure p
  let ν := finiteProbabilityMeasure q
  haveI hμprob : IsProbabilityMeasure μ :=
    finiteProbabilityMeasure_isProbability p hp_nonneg hp_mass
  haveI hνprob : IsProbabilityMeasure ν :=
    finiteProbabilityMeasure_isProbability q (fun i => (hq_pos i).le) hq_mass
  have hμν : μ ≪ ν :=
    finiteProbabilityMeasure_absolutelyContinuous p q hq_pos
  have hvar := klDiv_variational_lower_bound
    (μ := μ) (ν := ν) (f := fun i : Fin n => t * s i)
    hμν
    (MeasureTheory.Integrable.of_finite)
    (MeasureTheory.Integrable.of_finite)
    (MeasureTheory.Integrable.of_finite)
  have hp_int :
      (∫ i, t * s i ∂μ) = t * (∑ i, p i * s i) := by
    dsimp [μ]
    rw [finiteProbabilityMeasure_integral p (fun i => t * s i) hp_nonneg]
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl ?_
    intro i _hi
    ring
  have hq_int :
      (∫ i, Real.exp (t * s i) ∂ν) =
        ∑ i, q i * Real.exp (t * s i) := by
    dsimp [ν]
    exact finiteProbabilityMeasure_integral q
      (fun i => Real.exp (t * s i)) (fun i => (hq_pos i).le)
  simpa [hp_int, hq_int] using hvar

/--
Support-aware finite all-test variational inequality from concrete finite probability measures.

This boundary version allows zero coordinates in the reference probability vector, provided
`p` is supported on `q`.  It is the correct finite hypothesis for the variational bridge:
absolute continuity replaces the stronger interior assumption `∀ i, 0 < q i`.
-/
theorem finiteProbabilityMeasure_variational_lower_bound_of_support
    {p q : Fin n → ℝ}
    (hp_nonneg : ∀ i, 0 ≤ p i)
    (hp_mass : ∑ i, p i = 1)
    (hq_nonneg : ∀ i, 0 ≤ q i)
    (hq_mass : ∑ i, q i = 1)
    (hsupp : ∀ i, q i = 0 → p i = 0) :
    ∀ s : Fin n → ℝ, ∀ t : ℝ,
      t * (∑ i, p i * s i)
        - Real.log (∑ i, q i * Real.exp (t * s i))
        ≤ (InformationTheory.klDiv
            (finiteProbabilityMeasure p) (finiteProbabilityMeasure q)).toReal := by
  intro s t
  let μ := finiteProbabilityMeasure p
  let ν := finiteProbabilityMeasure q
  haveI hμprob : IsProbabilityMeasure μ :=
    finiteProbabilityMeasure_isProbability p hp_nonneg hp_mass
  haveI hνprob : IsProbabilityMeasure ν :=
    finiteProbabilityMeasure_isProbability q hq_nonneg hq_mass
  have hμν : μ ≪ ν :=
    finiteProbabilityMeasure_absolutelyContinuous_of_support p q hq_nonneg hsupp
  have hvar := klDiv_variational_lower_bound
    (μ := μ) (ν := ν) (f := fun i : Fin n => t * s i)
    hμν
    (MeasureTheory.Integrable.of_finite)
    (MeasureTheory.Integrable.of_finite)
    (MeasureTheory.Integrable.of_finite)
  have hp_int :
      (∫ i, t * s i ∂μ) = t * (∑ i, p i * s i) := by
    dsimp [μ]
    rw [finiteProbabilityMeasure_integral p (fun i => t * s i) hp_nonneg]
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl ?_
    intro i _hi
    ring
  have hq_int :
      (∫ i, Real.exp (t * s i) ∂ν) =
        ∑ i, q i * Real.exp (t * s i) := by
    dsimp [ν]
    exact finiteProbabilityMeasure_integral q
      (fun i => Real.exp (t * s i)) hq_nonneg
  simpa [hp_int, hq_int] using hvar

end Finite

end DualConvergence
end KLProjection
end FlowSinkhorn
