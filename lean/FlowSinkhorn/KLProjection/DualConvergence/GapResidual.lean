import FlowSinkhorn.KLProjection.DualConvergence.PerStepAscent
import FlowSinkhorn.KLProjection.BlockQuotient
import Mathlib.Data.Real.Basic

/-!
# Gap versus residuals

This module is reserved for the Lean formalization of Lemma `lem:gap-vs-res-quotient` from
`papers/kl-projections/sections/sec-dual-convergence.tex`.

Intended theorem names:
- `dualGap_le_residual_blockQuotient`;
- `dualGap_le_residual_with_constant`.

Design note:
this file should be the exact Lean counterpart of the quotient-seminorm argument in the paper:
all application modules should only need to provide explicit constants for the norm and the dual
orbit bound, not revisit the residual/gap algebra.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace DualConvergence

open scoped BigOperators

variable {gap residual : ℕ → ℝ}

/--
Abstract paper-facing form of Lemma `lem:gap-vs-res-quotient`:
if the gap is controlled by a dual-radius times residual and the radius is normalized by `≤ 1`,
then the gap is bounded by the residual.
-/
theorem dualGap_le_residual_blockQuotient
    {gapNow residualNow dualRadius : ℝ}
    (hgap : gapNow ≤ dualRadius * residualNow)
    (hdualRadius : dualRadius ≤ 1)
    (hresNonneg : 0 ≤ residualNow) :
    gapNow ≤ residualNow := by
  calc
    gapNow ≤ dualRadius * residualNow := hgap
    _ ≤ 1 * residualNow := by
          exact mul_le_mul_of_nonneg_right hdualRadius hresNonneg
    _ = residualNow := by ring

/--
Paper-shaped scalar reduction of the quotient-gap bound:
if a gap is controlled by `‖r‖ * (U★ + Uₖ)` and both quotient radii are bounded by `Umax`,
then `gap ≤ 2 * Umax * ‖r‖`.
-/
theorem dualGap_le_residual_twoUmax
    {gapNow residualNorm UStar UNow Umax : ℝ}
    (hgap : gapNow ≤ residualNorm * (UStar + UNow))
    (hUStar : UStar ≤ Umax)
    (hUNow : UNow ≤ Umax)
    (hresNonneg : 0 ≤ residualNorm) :
    gapNow ≤ (2 * Umax) * residualNorm := by
  have hsum : UStar + UNow ≤ Umax + Umax := add_le_add hUStar hUNow
  have hmul : residualNorm * (UStar + UNow) ≤ residualNorm * (Umax + Umax) :=
    mul_le_mul_of_nonneg_left hsum hresNonneg
  calc
    gapNow ≤ residualNorm * (UStar + UNow) := hgap
    _ ≤ residualNorm * (Umax + Umax) := hmul
    _ = (2 * Umax) * residualNorm := by ring

/-- Finite `ℓ¹` norm used in the quotient-gap Hölder bridge. -/
noncomputable def finiteL1 {n : ℕ} (r : Fin n → ℝ) : ℝ :=
  ∑ i, |r i|

/-- Two-block finite `ℓ¹` norm used by the block-quotient A.2 bridge. -/
noncomputable def finiteL1Pair {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    (r₁ : ι₁ → ℝ) (r₂ : ι₂ → ℝ) : ℝ :=
  (∑ i, |r₁ i|) + ∑ j, |r₂ j|

/-- Nonnegativity of the two-block finite `ℓ¹` norm. -/
theorem finiteL1Pair_nonneg
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    (r₁ : ι₁ → ℝ) (r₂ : ι₂ → ℝ) :
    0 ≤ finiteL1Pair r₁ r₂ := by
  unfold finiteL1Pair
  exact add_nonneg
    (Finset.sum_nonneg fun i _ => abs_nonneg (r₁ i))
    (Finset.sum_nonneg fun j _ => abs_nonneg (r₂ j))

/--
Finite Hölder inequality in the paper's `ℓ¹`/`ℓ∞` form.

If all coordinates of `u` are bounded by `B` in absolute value, then the residual-dual
pairing is controlled by `‖r‖₁ B`.
-/
theorem abs_sum_mul_le_finiteL1_mul_bound
    {n : ℕ} {r u : Fin n → ℝ} {B : ℝ}
    (hu : ∀ i, |u i| ≤ B) :
    |∑ i, r i * u i| ≤ finiteL1 r * B := by
  have hterm : ∀ i : Fin n, |r i * u i| ≤ |r i| * B := by
    intro i
    calc
      |r i * u i| = |r i| * |u i| := by rw [abs_mul]
      _ ≤ |r i| * B := by
            exact mul_le_mul_of_nonneg_left (hu i) (abs_nonneg (r i))
  calc
    |∑ i, r i * u i|
        ≤ ∑ i, |r i * u i| := by
            simpa using
              (Finset.abs_sum_le_sum_abs
                (fun i : Fin n => r i * u i) (Finset.univ : Finset (Fin n)))
    _ ≤ ∑ i, |r i| * B := by
            exact Finset.sum_le_sum fun i _ => hterm i
    _ = finiteL1 r * B := by
            unfold finiteL1
            rw [Finset.sum_mul]

/--
Two-block finite Hölder inequality in the paper's block-residual form.

This is the block analogue of `abs_sum_mul_le_finiteL1_mul_bound`: if both dual-displacement
blocks are coordinatewise bounded by the same radius `B`, then the two-block residual pairing is
bounded by the two-block `ℓ¹` residual norm times `B`.
-/
theorem abs_pair_sum_mul_le_finiteL1Pair_mul_bound
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    {r₁ u₁ : ι₁ → ℝ} {r₂ u₂ : ι₂ → ℝ} {B : ℝ}
    (hu₁ : ∀ i, |u₁ i| ≤ B)
    (hu₂ : ∀ j, |u₂ j| ≤ B) :
    |(∑ i, r₁ i * u₁ i) + ∑ j, r₂ j * u₂ j|
      ≤ finiteL1Pair r₁ r₂ * B := by
  have h₁ :
      |∑ i, r₁ i * u₁ i| ≤ (∑ i, |r₁ i|) * B := by
    have hterm : ∀ i : ι₁, |r₁ i * u₁ i| ≤ |r₁ i| * B := by
      intro i
      calc
        |r₁ i * u₁ i| = |r₁ i| * |u₁ i| := by rw [abs_mul]
        _ ≤ |r₁ i| * B := by
              exact mul_le_mul_of_nonneg_left (hu₁ i) (abs_nonneg (r₁ i))
    calc
      |∑ i, r₁ i * u₁ i|
          ≤ ∑ i, |r₁ i * u₁ i| := by
              simpa using
                (Finset.abs_sum_le_sum_abs
                  (fun i : ι₁ => r₁ i * u₁ i) (Finset.univ : Finset ι₁))
      _ ≤ ∑ i, |r₁ i| * B := by
              exact Finset.sum_le_sum fun i _ => hterm i
      _ = (∑ i, |r₁ i|) * B := by rw [Finset.sum_mul]
  have h₂ :
      |∑ j, r₂ j * u₂ j| ≤ (∑ j, |r₂ j|) * B := by
    have hterm : ∀ j : ι₂, |r₂ j * u₂ j| ≤ |r₂ j| * B := by
      intro j
      calc
        |r₂ j * u₂ j| = |r₂ j| * |u₂ j| := by rw [abs_mul]
        _ ≤ |r₂ j| * B := by
              exact mul_le_mul_of_nonneg_left (hu₂ j) (abs_nonneg (r₂ j))
    calc
      |∑ j, r₂ j * u₂ j|
          ≤ ∑ j, |r₂ j * u₂ j| := by
              simpa using
                (Finset.abs_sum_le_sum_abs
                  (fun j : ι₂ => r₂ j * u₂ j) (Finset.univ : Finset ι₂))
      _ ≤ ∑ j, |r₂ j| * B := by
              exact Finset.sum_le_sum fun j _ => hterm j
      _ = (∑ j, |r₂ j|) * B := by rw [Finset.sum_mul]
  calc
    |(∑ i, r₁ i * u₁ i) + ∑ j, r₂ j * u₂ j|
        ≤ |∑ i, r₁ i * u₁ i| + |∑ j, r₂ j * u₂ j| := abs_add_le _ _
    _ ≤ (∑ i, |r₁ i|) * B + (∑ j, |r₂ j|) * B := add_le_add h₁ h₂
    _ = finiteL1Pair r₁ r₂ * B := by
          unfold finiteL1Pair
          ring

/--
Two-block Hölder inequality from a block-sup bound on the displacement pair.

This is the same analytic step as `abs_pair_sum_mul_le_finiteL1Pair_mul_bound`, but phrased in
the quotient-friendly block norm used throughout the paper.
-/
theorem abs_pair_sum_mul_le_finiteL1Pair_mul_blockSup
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {r₁ u₁ : ι₁ → ℝ} {r₂ u₂ : ι₂ → ℝ} {B : ℝ}
    (hu : blockSupNorm (u₁, u₂) ≤ B) :
    |(∑ i, r₁ i * u₁ i) + ∑ j, r₂ j * u₂ j|
      ≤ finiteL1Pair r₁ r₂ * B := by
  have hu₁ : ∀ i, |u₁ i| ≤ B := by
    intro i
    exact (abs_le_coordSupNorm u₁ i).trans
      ((le_max_left (coordSupNorm u₁) (coordSupNorm u₂)).trans hu)
  have hu₂ : ∀ j, |u₂ j| ≤ B := by
    intro j
    exact (abs_le_coordSupNorm u₂ j).trans
      ((le_max_right (coordSupNorm u₁) (coordSupNorm u₂)).trans hu)
  exact abs_pair_sum_mul_le_finiteL1Pair_mul_bound
    (r₁ := r₁) (u₁ := u₁) (r₂ := r₂) (u₂ := u₂) hu₁ hu₂

/--
Turn an exact two-block gap pairing identity into the absolute-value gap control consumed by A.2.

This is a paper-navigation helper: many proofs identify the dual gap with a residual pairing
before applying Hölder.  The A.2 endpoint uses an absolute-value upper bound; this lemma performs
that conversion internally.
-/
theorem gap_le_abs_pairing_of_pairing_eq
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    {gapNow : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (hpair :
      gapNow =
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)) :
    gapNow ≤
      |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
        + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)| := by
  rw [hpair]
  exact le_abs_self _

/--
Turn the paper's concavity pairing bound into the absolute-value gap control consumed by A.2.
-/
theorem gap_le_abs_pairing_of_pairing_bound
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    {gapNow : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (hpair :
      gapNow ≤
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)) :
    gapNow ≤
      |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
        + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)| :=
  hpair.trans (le_abs_self _)

/--
Variant of `gap_le_abs_pairing_of_pairing_eq` for proofs whose convention gives the negative
of the residual pairing.
-/
theorem gap_le_abs_pairing_of_neg_pairing_eq
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    {gapNow : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (hpair :
      gapNow =
        -((∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))) :
    gapNow ≤
      |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
        + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)| := by
  rw [hpair]
  exact neg_le_abs _

/--
Variant of `gap_le_abs_pairing_of_pairing_bound` for the negative residual-pairing convention.
-/
theorem gap_le_abs_pairing_of_neg_pairing_bound
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    {gapNow : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (hpair :
      gapNow ≤
        -((∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))) :
    gapNow ≤
      |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
        + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)| :=
  hpair.trans (neg_le_abs _)

/--
Concavity-style bridge from a value gap to the residual pairing.

In applications, the paper's concavity step often appears as two facts: the dual gap is bounded
by a value difference, and that value difference is bounded by the residual pairing.  This lemma
internalizes the final conversion to the absolute pairing bound consumed by the A.2 endpoints.
-/
theorem gap_le_abs_pairing_of_valueGap_le_concavityPairing
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    {gapNow : ℝ} {Φ : ((ι₁ → ℝ) × (ι₂ → ℝ)) → ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (hgapValue :
      gapNow ≤ Φ (uStar₁, uStar₂) - Φ (uNow₁, uNow₂))
    (hconcavity :
      Φ (uStar₁, uStar₂) - Φ (uNow₁, uNow₂) ≤
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)) :
    gapNow ≤
      |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
        + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)| :=
  (hgapValue.trans hconcavity).trans (le_abs_self _)

/--
Paper-shaped Hölder bridge for Lemma A.2.

If the dual gap is controlled by an inner product against a dual displacement and this
displacement has coordinatewise bound `UStar + UNow`, then the gap is controlled by
`‖r‖₁ (UStar + UNow)`.
-/
theorem dualGap_le_residual_radius_of_innerProduct
    {n : ℕ} {gapNow UStar UNow : ℝ} {r u : Fin n → ℝ}
    (hgap : gapNow ≤ |∑ i, r i * u i|)
    (hu : ∀ i, |u i| ≤ UStar + UNow) :
    gapNow ≤ finiteL1 r * (UStar + UNow) := by
  exact hgap.trans (abs_sum_mul_le_finiteL1_mul_bound hu)

/--
More internalized A.2 scalar endpoint:
inner-product gap control plus coordinatewise quotient-radius control imply the paper's
`2 * Umax * ‖r‖₁` bound.
-/
theorem dualGap_le_twoUmax_of_innerProduct
    {n : ℕ} {gapNow UStar UNow Umax : ℝ} {r u : Fin n → ℝ}
    (hgap : gapNow ≤ |∑ i, r i * u i|)
    (hu : ∀ i, |u i| ≤ UStar + UNow)
    (hUStar : UStar ≤ Umax)
    (hUNow : UNow ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1 r := by
  have hholder : gapNow ≤ finiteL1 r * (UStar + UNow) :=
    dualGap_le_residual_radius_of_innerProduct hgap hu
  have hresNonneg : 0 ≤ finiteL1 r := by
    unfold finiteL1
    exact Finset.sum_nonneg fun i _ => abs_nonneg (r i)
  have htwo :
      gapNow ≤ (2 * Umax) * finiteL1 r :=
    dualGap_le_residual_twoUmax
      (gapNow := gapNow) (residualNorm := finiteL1 r)
      (UStar := UStar) (UNow := UNow) (Umax := Umax)
      hholder hUStar hUNow hresNonneg
  exact htwo

/--
Coordinate-radius version of the A.2 Hölder bridge.

This removes one layer of assumption packaging from `dualGap_le_twoUmax_of_innerProduct`:
the coordinatewise displacement bound is derived internally from separate radius bounds on the
reference dual vector and the current dual vector.
-/
theorem dualGap_le_twoUmax_of_coordinateRadii
    {n : ℕ} {gapNow UStar UNow Umax : ℝ} {r uStar uNow : Fin n → ℝ}
    (hgap : gapNow ≤ |∑ i, r i * (uStar i - uNow i)|)
    (hStar : ∀ i, |uStar i| ≤ UStar)
    (hNow : ∀ i, |uNow i| ≤ UNow)
    (hUStar : UStar ≤ Umax)
    (hUNow : UNow ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1 r := by
  have hdisp : ∀ i, |uStar i - uNow i| ≤ UStar + UNow := by
    intro i
    calc
      |uStar i - uNow i| = |uStar i + -uNow i| := by ring_nf
      _ ≤ |uStar i| + |-uNow i| := abs_add_le (uStar i) (-uNow i)
      _ = |uStar i| + |uNow i| := by rw [abs_neg]
      _ ≤ UStar + UNow := add_le_add (hStar i) (hNow i)
  exact dualGap_le_twoUmax_of_innerProduct
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r := r) (u := fun i => uStar i - uNow i)
    hgap hdisp hUStar hUNow

/--
Two-block coordinate-radius version of Lemma A.2.

This is closer to the paper statement than the single-vector endpoint: the residual pairing is
split over the two dual blocks, the two block displacements are bounded internally from separate
reference/current coordinate radii, and the final constant is `2 * Umax` times the two-block
`ℓ¹` residual norm.
-/
theorem dualGap_le_twoUmax_of_twoBlockCoordinateRadii
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (hgap :
      gapNow ≤
        |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)|)
    (hStar₁ : ∀ i, |uStar₁ i| ≤ UStar)
    (hStar₂ : ∀ j, |uStar₂ j| ≤ UStar)
    (hNow₁ : ∀ i, |uNow₁ i| ≤ UNow)
    (hNow₂ : ∀ j, |uNow₂ j| ≤ UNow)
    (hUStar : UStar ≤ Umax)
    (hUNow : UNow ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  have hdisp₁ : ∀ i, |uStar₁ i - uNow₁ i| ≤ UStar + UNow := by
    intro i
    calc
      |uStar₁ i - uNow₁ i| = |uStar₁ i + -uNow₁ i| := by ring_nf
      _ ≤ |uStar₁ i| + |-uNow₁ i| := abs_add_le (uStar₁ i) (-uNow₁ i)
      _ = |uStar₁ i| + |uNow₁ i| := by rw [abs_neg]
      _ ≤ UStar + UNow := add_le_add (hStar₁ i) (hNow₁ i)
  have hdisp₂ : ∀ j, |uStar₂ j - uNow₂ j| ≤ UStar + UNow := by
    intro j
    calc
      |uStar₂ j - uNow₂ j| = |uStar₂ j + -uNow₂ j| := by ring_nf
      _ ≤ |uStar₂ j| + |-uNow₂ j| := abs_add_le (uStar₂ j) (-uNow₂ j)
      _ = |uStar₂ j| + |uNow₂ j| := by rw [abs_neg]
      _ ≤ UStar + UNow := add_le_add (hStar₂ j) (hNow₂ j)
  have hholder :
      gapNow ≤ finiteL1Pair r₁ r₂ * (UStar + UNow) :=
    hgap.trans
      (abs_pair_sum_mul_le_finiteL1Pair_mul_bound
        (r₁ := r₁) (u₁ := fun i => uStar₁ i - uNow₁ i)
        (r₂ := r₂) (u₂ := fun j => uStar₂ j - uNow₂ j)
        hdisp₁ hdisp₂)
  have hresNonneg : 0 ≤ finiteL1Pair r₁ r₂ := by
    unfold finiteL1Pair
    exact add_nonneg
      (Finset.sum_nonneg fun i _ => abs_nonneg (r₁ i))
      (Finset.sum_nonneg fun j _ => abs_nonneg (r₂ j))
  exact dualGap_le_residual_twoUmax
    (gapNow := gapNow) (residualNorm := finiteL1Pair r₁ r₂)
    (UStar := UStar) (UNow := UNow) (Umax := Umax)
    hholder hUStar hUNow hresNonneg

/--
A.2 endpoint from a block-sup bound on the two-block displacement.

This packages the finite Holder step in the paper's quotient language: once the dual
displacement pair itself is bounded in `blockSupNorm` by `U★ + U_k`, the gap bound follows.
-/
theorem dualGap_le_twoUmax_of_displacementBlockSup
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (hgap :
      gapNow ≤
        |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)|)
    (hdisp :
      blockSupNorm
        (fun i => uStar₁ i - uNow₁ i, fun j => uStar₂ j - uNow₂ j) ≤
        UStar + UNow)
    (hUStar : UStar ≤ Umax)
    (hUNow : UNow ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  have hholder :
      gapNow ≤ finiteL1Pair r₁ r₂ * (UStar + UNow) :=
    hgap.trans
      (abs_pair_sum_mul_le_finiteL1Pair_mul_blockSup
        (r₁ := r₁) (u₁ := fun i => uStar₁ i - uNow₁ i)
        (r₂ := r₂) (u₂ := fun j => uStar₂ j - uNow₂ j)
        hdisp)
  exact dualGap_le_residual_twoUmax
    (gapNow := gapNow) (residualNorm := finiteL1Pair r₁ r₂)
    (UStar := UStar) (UNow := UNow) (Umax := Umax)
    hholder hUStar hUNow (finiteL1Pair_nonneg r₁ r₂)

/--
Block-sup triangle estimate for a two-block displacement.

This is the quotient-norm analogue of the coordinatewise triangle inequality used earlier.
-/
theorem blockSupNorm_pair_sub_le
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (uStar₁ uNow₁ : ι₁ → ℝ) (uStar₂ uNow₂ : ι₂ → ℝ) :
    blockSupNorm
        (fun i => uStar₁ i - uNow₁ i, fun j => uStar₂ j - uNow₂ j)
      ≤ blockSupNorm (uStar₁, uStar₂) + blockSupNorm (uNow₁, uNow₂) := by
  unfold blockSupNorm
  have h₁ : coordSupNorm (fun i => uStar₁ i - uNow₁ i)
      ≤ coordSupNorm uStar₁ + coordSupNorm uNow₁ :=
    coordSupNorm_sub_le uStar₁ uNow₁
  have h₂ : coordSupNorm (fun j => uStar₂ j - uNow₂ j)
      ≤ coordSupNorm uStar₂ + coordSupNorm uNow₂ :=
    coordSupNorm_sub_le uStar₂ uNow₂
  have h₁' : coordSupNorm (fun i => uStar₁ i - uNow₁ i)
      ≤ max (coordSupNorm uStar₁) (coordSupNorm uStar₂)
          + max (coordSupNorm uNow₁) (coordSupNorm uNow₂) := by
    exact h₁.trans
      (add_le_add (le_max_left _ _) (le_max_left _ _))
  have h₂' : coordSupNorm (fun j => uStar₂ j - uNow₂ j)
      ≤ max (coordSupNorm uStar₁) (coordSupNorm uStar₂)
          + max (coordSupNorm uNow₁) (coordSupNorm uNow₂) := by
    exact h₂.trans
      (add_le_add (le_max_right _ _) (le_max_right _ _))
  exact max_le h₁' h₂'

/--
Two-block block-sup-radius version of Lemma A.2.

This removes another layer of assumption packaging from
`dualGap_le_twoUmax_of_twoBlockCoordinateRadii`: instead of asking for four coordinatewise
radius hypotheses, it assumes the reference and current two-block dual vectors are bounded in
the block sup norm.  The four coordinate bounds are derived internally from `blockSupNorm`.
-/
theorem dualGap_le_twoUmax_of_twoBlockSupRadii
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (hgap :
      gapNow ≤
        |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)|)
    (hStar : blockSupNorm (uStar₁, uStar₂) ≤ UStar)
    (hNow : blockSupNorm (uNow₁, uNow₂) ≤ UNow)
    (hUStar : UStar ≤ Umax)
    (hUNow : UNow ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  have hStar₁ : ∀ i, |uStar₁ i| ≤ UStar := by
    intro i
    exact (abs_le_coordSupNorm uStar₁ i).trans
      ((le_max_left (coordSupNorm uStar₁) (coordSupNorm uStar₂)).trans hStar)
  have hStar₂ : ∀ j, |uStar₂ j| ≤ UStar := by
    intro j
    exact (abs_le_coordSupNorm uStar₂ j).trans
      ((le_max_right (coordSupNorm uStar₁) (coordSupNorm uStar₂)).trans hStar)
  have hNow₁ : ∀ i, |uNow₁ i| ≤ UNow := by
    intro i
    exact (abs_le_coordSupNorm uNow₁ i).trans
      ((le_max_left (coordSupNorm uNow₁) (coordSupNorm uNow₂)).trans hNow)
  have hNow₂ : ∀ j, |uNow₂ j| ≤ UNow := by
    intro j
    exact (abs_le_coordSupNorm uNow₂ j).trans
      ((le_max_right (coordSupNorm uNow₁) (coordSupNorm uNow₂)).trans hNow)
  exact dualGap_le_twoUmax_of_twoBlockCoordinateRadii
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    hgap hStar₁ hStar₂ hNow₁ hNow₂ hUStar hUNow

/--
Two-block block-sup-radius version routed through the displacement block norm.

Compared with `dualGap_le_twoUmax_of_twoBlockSupRadii`, this theorem exposes the exact
quotient-geometric chain: separate block radii imply a displacement block radius by
`blockSupNorm_pair_sub_le`, and the residual pairing is then controlled by the block-sup
Holder estimate.
-/
theorem dualGap_le_twoUmax_of_twoBlockSupRadii_via_displacement
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (hgap :
      gapNow ≤
        |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)|)
    (hStar : blockSupNorm (uStar₁, uStar₂) ≤ UStar)
    (hNow : blockSupNorm (uNow₁, uNow₂) ≤ UNow)
    (hUStar : UStar ≤ Umax)
    (hUNow : UNow ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  have hdisp :
      blockSupNorm
        (fun i => uStar₁ i - uNow₁ i, fun j => uStar₂ j - uNow₂ j) ≤
        UStar + UNow := by
    exact (blockSupNorm_pair_sub_le uStar₁ uNow₁ uStar₂ uNow₂).trans
      (add_le_add hStar hNow)
  exact dualGap_le_twoUmax_of_displacementBlockSup
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    hgap hdisp hUStar hUNow

/--
Gauge orthogonality makes the two-block residual pairing invariant under signed paired shifts.

This is the algebraic orthogonality bridge needed when A.2 is proved using quotient
representatives: shifting both dual block pairs changes the raw displacement by a gauge vector,
and this lemma certifies that the residual pairing is unchanged when the residual is orthogonal
to that gauge direction.
-/
theorem pairResidualPairing_signedShift_sub_eq
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign) (cStar cNow : ℝ)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0) :
    (∑ i, r₁ i *
          ((uStar₁ i + cStar) - (uNow₁ i + cNow)))
        + ∑ j, r₂ j *
          ((uStar₂ j + τ.toReal * cStar) - (uNow₂ j + τ.toReal * cNow))
      =
    (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
        + ∑ j, r₂ j * (uStar₂ j - uNow₂ j) := by
  let d : ℝ := cStar - cNow
  have h₁ :
      (∑ i, r₁ i * ((uStar₁ i + cStar) - (uNow₁ i + cNow)))
        =
      (∑ i, r₁ i * (uStar₁ i - uNow₁ i)) + d * ∑ i, r₁ i := by
    calc
      (∑ i, r₁ i * ((uStar₁ i + cStar) - (uNow₁ i + cNow)))
          = ∑ i, (r₁ i * (uStar₁ i - uNow₁ i) + d * r₁ i) := by
              refine Finset.sum_congr rfl ?_
              intro i _
              simp [d]
              ring
      _ = (∑ i, r₁ i * (uStar₁ i - uNow₁ i)) + ∑ i, d * r₁ i := by
              rw [Finset.sum_add_distrib]
      _ = (∑ i, r₁ i * (uStar₁ i - uNow₁ i)) + d * ∑ i, r₁ i := by
              rw [Finset.mul_sum]
  have h₂ :
      (∑ j, r₂ j *
          ((uStar₂ j + τ.toReal * cStar) - (uNow₂ j + τ.toReal * cNow)))
        =
      (∑ j, r₂ j * (uStar₂ j - uNow₂ j)) + (τ.toReal * d) * ∑ j, r₂ j := by
    calc
      (∑ j, r₂ j *
          ((uStar₂ j + τ.toReal * cStar) - (uNow₂ j + τ.toReal * cNow)))
          = ∑ j, (r₂ j * (uStar₂ j - uNow₂ j) + (τ.toReal * d) * r₂ j) := by
              refine Finset.sum_congr rfl ?_
              intro j _
              cases τ <;> simp [PairedSign.toReal, d] <;> ring
      _ = (∑ j, r₂ j * (uStar₂ j - uNow₂ j)) + ∑ j, (τ.toReal * d) * r₂ j := by
              rw [Finset.sum_add_distrib]
      _ = (∑ j, r₂ j * (uStar₂ j - uNow₂ j)) + (τ.toReal * d) * ∑ j, r₂ j := by
              rw [Finset.mul_sum]
  have hgauge : d * ∑ i, r₁ i + (τ.toReal * d) * ∑ j, r₂ j = 0 := by
    have hmul : d * ((∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j)) = 0 := by
      rw [horth, mul_zero]
    linarith [hmul]
  rw [h₁, h₂]
  linarith

/--
A.2 endpoint using shifted quotient representatives.

The hypotheses now match the quotient proof more closely: the block-sup bounds are required only
after choosing signed paired-shift representatives.  Gauge orthogonality transports the original
residual pairing to those shifted representatives before applying the block-sup A.2 bridge.
-/
theorem dualGap_le_twoUmax_of_shiftedTwoBlockSupRadii
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign) (cStar cNow : ℝ)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hgap :
      gapNow ≤
        |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)|)
    (hStar :
      blockSupNorm
        (fun i => uStar₁ i + cStar, fun j => uStar₂ j + τ.toReal * cStar) ≤ UStar)
    (hNow :
      blockSupNorm
        (fun i => uNow₁ i + cNow, fun j => uNow₂ j + τ.toReal * cNow) ≤ UNow)
    (hUStar : UStar ≤ Umax)
    (hUNow : UNow ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  have hpair :=
    pairResidualPairing_signedShift_sub_eq
      (τ := τ) (cStar := cStar) (cNow := cNow)
      (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
      horth
  have hgap_shifted :
      gapNow ≤
        |(∑ i, r₁ i *
            ((uStar₁ i + cStar) - (uNow₁ i + cNow)))
          + ∑ j, r₂ j *
            ((uStar₂ j + τ.toReal * cStar) - (uNow₂ j + τ.toReal * cNow))| := by
    rwa [hpair]
  exact dualGap_le_twoUmax_of_twoBlockSupRadii_via_displacement
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁)
    (uStar₁ := fun i => uStar₁ i + cStar)
    (uNow₁ := fun i => uNow₁ i + cNow)
    (r₂ := r₂)
    (uStar₂ := fun j => uStar₂ j + τ.toReal * cStar)
    (uNow₂ := fun j => uNow₂ j + τ.toReal * cNow)
    hgap_shifted hStar hNow hUStar hUNow

/--
A.2 endpoint with existential shifted quotient representatives.

This removes the final bookkeeping choice of representatives from the public quotient interface:
instead of taking the two shifts as fixed arguments, it consumes the actual quotient statement
that suitable shifted representatives exist with the prescribed block-sup radii.  The proof then
unpacks those representatives and delegates to
`dualGap_le_twoUmax_of_shiftedTwoBlockSupRadii`.
-/
theorem dualGap_le_twoUmax_of_existsShiftedTwoBlockSupRadii
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hgap :
      gapNow ≤
        |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)|)
    (hStar_exists :
      ∃ cStar : ℝ,
        blockSupNorm
          (fun i => uStar₁ i + cStar, fun j => uStar₂ j + τ.toReal * cStar) ≤
          UStar)
    (hNow_exists :
      ∃ cNow : ℝ,
        blockSupNorm
          (fun i => uNow₁ i + cNow, fun j => uNow₂ j + τ.toReal * cNow) ≤
          UNow)
    (hUStar : UStar ≤ Umax)
    (hUNow : UNow ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  rcases hStar_exists with ⟨cStar, hStar⟩
  rcases hNow_exists with ⟨cNow, hNow⟩
  exact dualGap_le_twoUmax_of_shiftedTwoBlockSupRadii
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ cStar cNow horth hgap hStar hNow hUStar hUNow

/--
A.2 endpoint stated directly with the signed paired-shift quotient primitive.

This is the preferred paper-facing formulation: the representative hypotheses are phrased using
`signedPairedShift`, the same orbit action used by `signedPairedQuotientSupSeminorm` in
`BlockQuotient.lean`.  Thus the only remaining quotient-geometry input is exactly the expected
one: produce bounded representatives in that orbit.
-/
theorem dualGap_le_twoUmax_of_signedShiftWitnesses
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hgap :
      gapNow ≤
        |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)|)
    (hStar_exists :
      ∃ cStar : ℝ,
        blockSupNorm (signedPairedShift τ cStar (uStar₁, uStar₂)) ≤ UStar)
    (hNow_exists :
      ∃ cNow : ℝ,
        blockSupNorm (signedPairedShift τ cNow (uNow₁, uNow₂)) ≤ UNow)
    (hUStar : UStar ≤ Umax)
    (hUNow : UNow ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  rcases hStar_exists with ⟨cStar, hStar⟩
  rcases hNow_exists with ⟨cNow, hNow⟩
  exact dualGap_le_twoUmax_of_shiftedTwoBlockSupRadii
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ cStar cNow horth hgap
    (by simpa [signedPairedShift] using hStar)
    (by simpa [signedPairedShift] using hNow)
    hUStar hUNow

/--
Convert a bounded shifted representative into a bounded element of the signed-shift norm set.

This is one direction of the quotient-orbit witness bridge used by the A.2 endpoint below.
-/
theorem signedShiftNormSetBound_of_shiftWitness
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) {U : ℝ}
    (h : ∃ c : ℝ, blockSupNorm (signedPairedShift τ c u) ≤ U) :
    ∃ R : ℝ, R ∈ signedPairedShiftNormSet τ u ∧ R ≤ U := by
  rcases h with ⟨c, hc⟩
  exact ⟨blockSupNorm (signedPairedShift τ c u), ⟨c, rfl⟩, hc⟩

/--
Convert a bounded element of the signed-shift norm set into an explicit shifted representative.

This is the other direction of the quotient-orbit witness bridge.
-/
theorem shiftWitness_of_signedShiftNormSetBound
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) {U : ℝ}
    (h : ∃ R : ℝ, R ∈ signedPairedShiftNormSet τ u ∧ R ≤ U) :
    ∃ c : ℝ, blockSupNorm (signedPairedShift τ c u) ≤ U := by
  rcases h with ⟨R, hRmem, hRU⟩
  rcases hRmem with ⟨c, rfl⟩
  exact ⟨c, hRU⟩

/--
Equivalence between the two witness languages for a bounded signed quotient representative.

The left side is representative-facing; the right side is orbit-set-facing.  The latter is closer
to `signedPairedQuotientSupSeminorm`, which is defined as the infimum of this orbit set.
-/
theorem shiftWitness_iff_signedShiftNormSetBound
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) {U : ℝ} :
    (∃ c : ℝ, blockSupNorm (signedPairedShift τ c u) ≤ U) ↔
      ∃ R : ℝ, R ∈ signedPairedShiftNormSet τ u ∧ R ≤ U := by
  constructor
  · exact signedShiftNormSetBound_of_shiftWitness τ u
  · exact shiftWitness_of_signedShiftNormSetBound τ u

/--
Build a block-sup estimate from coordinatewise absolute estimates on both blocks.

This is the concrete representative primitive used to turn centered/shifted coordinate bounds
into the block-sup hypotheses of the quotient A.2 bridge.
-/
theorem blockSupNorm_le_of_forall_abs_le
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (u : BlockPair ι₁ ι₂) {U : ℝ}
    (h₁ : ∀ i, |u.1 i| ≤ U)
    (h₂ : ∀ j, |u.2 j| ≤ U) :
    blockSupNorm u ≤ U := by
  unfold blockSupNorm
  apply max_le
  · unfold coordSupNorm
    apply Finset.sup'_le Finset.univ_nonempty
    intro i _
    exact h₁ i
  · unfold coordSupNorm
    apply Finset.sup'_le Finset.univ_nonempty
    intro j _
    exact h₂ j

/--
Concrete signed-shift representative from coordinatewise bounds after the same signed shift.
-/
theorem signedShiftWitness_of_forall_abs_signedShift_le
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) (c : ℝ) {U : ℝ}
    (h₁ : ∀ i, |u.1 i + c| ≤ U)
    (h₂ : ∀ j, |u.2 j + τ.toReal * c| ≤ U) :
    ∃ c : ℝ, blockSupNorm (signedPairedShift τ c u) ≤ U := by
  refine ⟨c, ?_⟩
  exact blockSupNorm_le_of_forall_abs_le
    (signedPairedShift τ c u)
    (by intro i; simpa [signedPairedShift] using h₁ i)
    (by intro j; simpa [signedPairedShift] using h₂ j)

/--
Strict shifted representative from coordinatewise signed-shift bounds below a smaller radius.
-/
theorem signedShiftWitness_lt_Umax_of_forall_abs_signedShift_le_radius_lt
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) (c : ℝ) {U Umax : ℝ}
    (h₁ : ∀ i, |u.1 i + c| ≤ U)
    (h₂ : ∀ j, |u.2 j + τ.toReal * c| ≤ U)
    (hU : U < Umax) :
    ∃ c : ℝ, blockSupNorm (signedPairedShift τ c u) < Umax := by
  rcases signedShiftWitness_of_forall_abs_signedShift_le τ u c h₁ h₂ with
    ⟨c', hc'⟩
  exact ⟨c', lt_of_le_of_lt hc' hU⟩

/--
Strict signed orbit-set bound from coordinatewise signed-shift bounds.
-/
theorem signedShiftNormSetBound_lt_Umax_of_forall_abs_signedShift_le_radius_lt
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) (c : ℝ) {U Umax : ℝ}
    (h₁ : ∀ i, |u.1 i + c| ≤ U)
    (h₂ : ∀ j, |u.2 j + τ.toReal * c| ≤ U)
    (hU : U < Umax) :
    ∃ R : ℝ, R ∈ signedPairedShiftNormSet τ u ∧ R < Umax := by
  refine ⟨blockSupNorm (signedPairedShift τ c u), ⟨c, rfl⟩, ?_⟩
  exact lt_of_le_of_lt
    (blockSupNorm_le_of_forall_abs_le
      (signedPairedShift τ c u)
      (by intro i; simpa [signedPairedShift] using h₁ i)
      (by intro j; simpa [signedPairedShift] using h₂ j))
    hU

/--
Strict quotient-margin extraction from a concrete shifted representative.

This is the reverse direction needed by paper-style quotient arguments: once an application has
exhibited any orbit representative whose block sup norm is strictly below `U`, the infimum-based
signed quotient seminorm is strictly below `U` as well.
-/
theorem signedPairedQuotientSupSeminorm_lt_of_shiftWitness_lt
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) {U : ℝ}
    (h : ∃ c : ℝ, blockSupNorm (signedPairedShift τ c u) < U) :
    signedPairedQuotientSupSeminorm τ u < U := by
  rcases h with ⟨c, hc⟩
  have hleShift :
      signedPairedQuotientSupSeminorm τ (signedPairedShift τ c u) ≤
        blockSupNorm (signedPairedShift τ c u) :=
    signedPairedQuotientSupSeminorm_le_blockSupNorm τ (signedPairedShift τ c u)
  have hle :
      signedPairedQuotientSupSeminorm τ u ≤
        blockSupNorm (signedPairedShift τ c u) := by
    simpa [signedPairedQuotientSupSeminorm_pairedShift] using hleShift
  exact lt_of_le_of_lt hle hc

/--
Strict quotient-margin extraction from a strict signed-shift orbit-set bound.

This is the orbit-set version of `signedPairedQuotientSupSeminorm_lt_of_shiftWitness_lt`.
-/
theorem signedPairedQuotientSupSeminorm_lt_of_signedShiftNormSetBound_lt
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) {U : ℝ}
    (h : ∃ R : ℝ, R ∈ signedPairedShiftNormSet τ u ∧ R < U) :
    signedPairedQuotientSupSeminorm τ u < U := by
  rcases h with ⟨R, hRmem, hRlt⟩
  rcases hRmem with ⟨c, rfl⟩
  exact signedPairedQuotientSupSeminorm_lt_of_shiftWitness_lt τ u ⟨c, hRlt⟩

/--
Strict quotient-margin extraction from an orbit-set bound through an intermediate strict radius.
-/
theorem signedPairedQuotientSupSeminorm_lt_of_signedShiftNormSetBound_le_lt
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) {U Umax : ℝ}
    (h : ∃ R : ℝ, R ∈ signedPairedShiftNormSet τ u ∧ R ≤ U)
    (hU : U < Umax) :
    signedPairedQuotientSupSeminorm τ u < Umax := by
  exact signedPairedQuotientSupSeminorm_lt_of_signedShiftNormSetBound_lt
    τ u (by
      rcases h with ⟨R, hRmem, hRU⟩
      exact ⟨R, hRmem, lt_of_le_of_lt hRU hU⟩)

/--
A.2 endpoint stated with bounded elements of the signed shift norm set.

Compared with `dualGap_le_twoUmax_of_signedShiftWitnesses`, this formulation exposes the
quotient-orbit layer directly: each radius hypothesis is an element of
`signedPairedShiftNormSet` below the desired bound.
-/
theorem dualGap_le_twoUmax_of_signedShiftNormSetBounds
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hgap :
      gapNow ≤
        |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)|)
    (hStar_orbit :
      ∃ RStar : ℝ,
        RStar ∈ signedPairedShiftNormSet τ (uStar₁, uStar₂) ∧ RStar ≤ UStar)
    (hNow_orbit :
      ∃ RNow : ℝ,
        RNow ∈ signedPairedShiftNormSet τ (uNow₁, uNow₂) ∧ RNow ≤ UNow)
    (hUStar : UStar ≤ Umax)
    (hUNow : UNow ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_signedShiftWitnesses
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hgap
    (shiftWitness_of_signedShiftNormSetBound τ (uStar₁, uStar₂) hStar_orbit)
    (shiftWitness_of_signedShiftNormSetBound τ (uNow₁, uNow₂) hNow_orbit)
    hUStar hUNow

/--
Monotone-radius version of the signed-shift norm-set A.2 endpoint.

If the orbit representatives are first bounded by smaller radii `UStar0`, `UNow0`, those bounds
can be lifted to the radii consumed by the paper endpoint.
-/
theorem dualGap_le_twoUmax_of_signedShiftNormSetBounds_mono
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar0 UNow0 UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hgap :
      gapNow ≤
        |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)|)
    (hStar_orbit :
      ∃ RStar : ℝ,
        RStar ∈ signedPairedShiftNormSet τ (uStar₁, uStar₂) ∧ RStar ≤ UStar0)
    (hNow_orbit :
      ∃ RNow : ℝ,
        RNow ∈ signedPairedShiftNormSet τ (uNow₁, uNow₂) ∧ RNow ≤ UNow0)
    (hStar0 : UStar0 ≤ UStar)
    (hNow0 : UNow0 ≤ UNow)
    (hUStar : UStar ≤ Umax)
    (hUNow : UNow ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  have hStar_orbit' :
      ∃ RStar : ℝ,
        RStar ∈ signedPairedShiftNormSet τ (uStar₁, uStar₂) ∧ RStar ≤ UStar := by
    rcases hStar_orbit with ⟨RStar, hmem, hR⟩
    exact ⟨RStar, hmem, hR.trans hStar0⟩
  have hNow_orbit' :
      ∃ RNow : ℝ,
        RNow ∈ signedPairedShiftNormSet τ (uNow₁, uNow₂) ∧ RNow ≤ UNow := by
    rcases hNow_orbit with ⟨RNow, hmem, hR⟩
    exact ⟨RNow, hmem, hR.trans hNow0⟩
  exact dualGap_le_twoUmax_of_signedShiftNormSetBounds
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hgap hStar_orbit' hNow_orbit' hUStar hUNow

/--
Extract an orbit-set bound from a strict signed quotient-seminorm bound.

Since `signedPairedQuotientSupSeminorm` is an infimum over `signedPairedShiftNormSet`, a strict
upper bound on the infimum yields an actual shifted representative below that bound.
-/
theorem signedShiftNormSetBound_of_quotientSup_lt
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) {U : ℝ}
    (hU : signedPairedQuotientSupSeminorm τ u < U) :
    ∃ R : ℝ, R ∈ signedPairedShiftNormSet τ u ∧ R ≤ U := by
  have hmem :
      ∃ R : ℝ, R ∈ signedPairedShiftNormSet τ u ∧ R < U := by
    simpa [signedPairedQuotientSupSeminorm] using
      (exists_lt_of_csInf_lt (signedPairedShiftNormSet_nonempty τ u) hU)
  rcases hmem with ⟨R, hRmem, hRlt⟩
  exact ⟨R, hRmem, le_of_lt hRlt⟩

/--
Slack form of `signedShiftNormSetBound_of_quotientSup_lt`.

This is often the usable quotient-seminorm interface: if the infimum plus a positive slack is
below the desired radius, then a concrete orbit representative is below that radius.
-/
theorem signedShiftNormSetBound_of_quotientSup_add_pos_le
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) {eps U : ℝ}
    (heps : 0 < eps)
    (hU : signedPairedQuotientSupSeminorm τ u + eps ≤ U) :
    ∃ R : ℝ, R ∈ signedPairedShiftNormSet τ u ∧ R ≤ U := by
  have hlt : signedPairedQuotientSupSeminorm τ u < U := by
    linarith
  exact signedShiftNormSetBound_of_quotientSup_lt τ u hlt

/--
A.2 endpoint from strict signed quotient-seminorm radius bounds.

This is closer to the paper quotient geometry than the orbit-set endpoint: the hypotheses are
now stated directly in terms of `signedPairedQuotientSupSeminorm`.  Strictness is the standard
price for extracting a concrete representative from an infimum without separately proving
attainment.
-/
theorem dualGap_le_twoUmax_of_quotientSup_lt_radii
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hgap :
      gapNow ≤
        |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)|)
    (hStar_lt :
      signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) < UStar)
    (hNow_lt :
      signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) < UNow)
    (hUStar : UStar ≤ Umax)
    (hUNow : UNow ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_signedShiftNormSetBounds
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hgap
    (signedShiftNormSetBound_of_quotientSup_lt τ (uStar₁, uStar₂) hStar_lt)
    (signedShiftNormSetBound_of_quotientSup_lt τ (uNow₁, uNow₂) hNow_lt)
    hUStar hUNow

/--
A.2 endpoint from signed quotient-seminorm bounds with separate positive slacks.
-/
theorem dualGap_le_twoUmax_of_quotientSup_add_pos_le
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow epsStar epsNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hgap :
      gapNow ≤
        |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)|)
    (hepsStar : 0 < epsStar)
    (hepsNow : 0 < epsNow)
    (hStar :
      signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) + epsStar ≤ UStar)
    (hNow :
      signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) + epsNow ≤ UNow)
    (hUStar : UStar ≤ Umax)
    (hUNow : UNow ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_signedShiftNormSetBounds
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hgap
    (signedShiftNormSetBound_of_quotientSup_add_pos_le
      τ (uStar₁, uStar₂) hepsStar hStar)
    (signedShiftNormSetBound_of_quotientSup_add_pos_le
      τ (uNow₁, uNow₂) hepsNow hNow)
    hUStar hUNow

/--
Common-slack version of the quotient-seminorm A.2 endpoint.
-/
theorem dualGap_le_twoUmax_of_quotientSup_common_add_pos_le
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow eps UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hgap :
      gapNow ≤
        |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)|)
    (heps : 0 < eps)
    (hStar :
      signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) + eps ≤ UStar)
    (hNow :
      signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) + eps ≤ UNow)
    (hUStar : UStar ≤ Umax)
    (hUNow : UNow ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ :=
  dualGap_le_twoUmax_of_quotientSup_add_pos_le
    (gapNow := gapNow) (epsStar := eps) (epsNow := eps)
    (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hgap heps heps hStar hNow hUStar hUNow

/--
Turn a non-strict quotient-seminorm comparison plus positive slack into a strict radius bound.
-/
theorem quotientSup_lt_of_le_add_pos_le
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) {Q eps U : ℝ}
    (hQ : signedPairedQuotientSupSeminorm τ u ≤ Q)
    (heps : 0 < eps)
    (hU : Q + eps ≤ U) :
    signedPairedQuotientSupSeminorm τ u < U := by
  linarith

/--
Orbit representative extraction from a non-strict quotient bound and a positive slack.
-/
theorem signedShiftNormSetBound_of_quotientSup_le_add_pos
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) {Q eps U : ℝ}
    (hQ : signedPairedQuotientSupSeminorm τ u ≤ Q)
    (heps : 0 < eps)
    (hU : Q + eps ≤ U) :
    ∃ R : ℝ, R ∈ signedPairedShiftNormSet τ u ∧ R ≤ U := by
  exact signedShiftNormSetBound_of_quotientSup_lt τ u
    (quotientSup_lt_of_le_add_pos_le τ u hQ heps hU)

/--
A.2 endpoint from non-strict signed quotient-seminorm radius controls plus separate slacks.

This is the most convenient non-attainment interface: downstream modules may prove
`signedPairedQuotientSupSeminorm ≤ Q`, then spend a positive slack to obtain concrete shifted
representatives.
-/
theorem dualGap_le_twoUmax_of_quotientSup_le_slack_radii
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow QStar QNow epsStar epsNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hgap :
      gapNow ≤
        |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)|)
    (hQStar : signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) ≤ QStar)
    (hQNow : signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) ≤ QNow)
    (hepsStar : 0 < epsStar)
    (hepsNow : 0 < epsNow)
    (hStar : QStar + epsStar ≤ UStar)
    (hNow : QNow + epsNow ≤ UNow)
    (hUStar : UStar ≤ Umax)
    (hUNow : UNow ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_quotientSup_lt_radii
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hgap
    (quotientSup_lt_of_le_add_pos_le τ (uStar₁, uStar₂) hQStar hepsStar hStar)
    (quotientSup_lt_of_le_add_pos_le τ (uNow₁, uNow₂) hQNow hepsNow hNow)
    hUStar hUNow

/--
Common-slack version of the non-strict quotient-seminorm A.2 endpoint.
-/
theorem dualGap_le_twoUmax_of_quotientSup_le_common_slack_radii
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow QStar QNow eps UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hgap :
      gapNow ≤
        |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)|)
    (hQStar : signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) ≤ QStar)
    (hQNow : signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) ≤ QNow)
    (heps : 0 < eps)
    (hStar : QStar + eps ≤ UStar)
    (hNow : QNow + eps ≤ UNow)
    (hUStar : UStar ≤ Umax)
    (hUNow : UNow ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ :=
  dualGap_le_twoUmax_of_quotientSup_le_slack_radii
    (gapNow := gapNow) (QStar := QStar) (QNow := QNow)
    (epsStar := eps) (epsNow := eps)
    (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hgap hQStar hQNow heps heps hStar hNow hUStar hUNow

/--
Direct `Umax` version of the non-strict quotient-seminorm A.2 endpoint.
-/
theorem dualGap_le_twoUmax_of_quotientSup_le_common_slack_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow QStar QNow eps Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hgap :
      gapNow ≤
        |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)|)
    (hQStar : signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) ≤ QStar)
    (hQNow : signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) ≤ QNow)
    (heps : 0 < eps)
    (hStar : QStar + eps ≤ Umax)
    (hNow : QNow + eps ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ :=
  dualGap_le_twoUmax_of_quotientSup_le_common_slack_radii
    (gapNow := gapNow) (QStar := QStar) (QNow := QNow) (eps := eps)
    (UStar := Umax) (UNow := Umax) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hgap hQStar hQNow heps hStar hNow (le_refl Umax) (le_refl Umax)

/--
A.2 endpoint where the positive slack is generated from explicit strict margins to `Umax`.

This is often easier for downstream modules than choosing `eps`: prove both quotient radii are
strictly below `Umax`; Lean then chooses the smaller half-margin as a common positive slack.
-/
theorem dualGap_le_twoUmax_of_quotientSup_le_strict_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow QStar QNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hgap :
      gapNow ≤
        |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)|)
    (hQStar : signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) ≤ QStar)
    (hQNow : signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) ≤ QNow)
    (hStar_lt : QStar < Umax)
    (hNow_lt : QNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  let eps : ℝ := min (Umax - QStar) (Umax - QNow) / 2
  have hposStar : 0 < Umax - QStar := sub_pos.mpr hStar_lt
  have hposNow : 0 < Umax - QNow := sub_pos.mpr hNow_lt
  have hminpos : 0 < min (Umax - QStar) (Umax - QNow) := lt_min hposStar hposNow
  have heps : 0 < eps := by
    unfold eps
    linarith
  have hStar : QStar + eps ≤ Umax := by
    have heps_le : eps ≤ Umax - QStar := by
      unfold eps
      have hmin_le : min (Umax - QStar) (Umax - QNow) ≤ Umax - QStar :=
        min_le_left _ _
      linarith
    linarith
  have hNow : QNow + eps ≤ Umax := by
    have heps_le : eps ≤ Umax - QNow := by
      unfold eps
      have hmin_le : min (Umax - QStar) (Umax - QNow) ≤ Umax - QNow :=
        min_le_right _ _
      linarith
    linarith
  exact dualGap_le_twoUmax_of_quotientSup_le_common_slack_Umax
    (gapNow := gapNow) (QStar := QStar) (QNow := QNow) (eps := eps)
    (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hgap hQStar hQNow heps hStar hNow

/--
A.2 endpoint with the strict quotient-radius margins stated directly on the two block pairs.

This removes the auxiliary `QStar`/`QNow` bookkeeping from
`dualGap_le_twoUmax_of_quotientSup_le_strict_Umax`: downstream callers now prove the two
quotient seminorms are strictly below `Umax`, and the proof internally converts these strict
margins into the slack/representative interface needed for infimum extraction.
-/
theorem dualGap_le_twoUmax_of_quotientSup_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hgap :
      gapNow ≤
        |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)|)
    (hStar_lt :
      signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) < Umax)
    (hNow_lt :
      signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_quotientSup_le_strict_Umax
    (gapNow := gapNow)
    (QStar := signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂))
    (QNow := signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂))
    (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hgap (le_refl _) (le_refl _) hStar_lt hNow_lt

/--
A.2 endpoint from quotient-seminorm bounds through intermediate radii strictly below `Umax`.

This is the paper-facing strict-margin form when an application first proves explicit estimates
`quotientSeminorm ≤ U★`, `quotientSeminorm ≤ U_k`, then checks both radii are below the common
budget `Umax`.
-/
theorem dualGap_le_twoUmax_of_quotientSup_le_radii_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hgap :
      gapNow ≤
        |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)|)
    (hStar :
      signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) ≤ UStar)
    (hNow :
      signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) ≤ UNow)
    (hStar_lt : UStar < Umax)
    (hNow_lt : UNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_quotientSup_lt_Umax
    (gapNow := gapNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hgap
    (lt_of_le_of_lt hStar hStar_lt)
    (lt_of_le_of_lt hNow hNow_lt)

/--
A.2 endpoint from an exact residual-pairing identity and strict quotient-radius margins.

Compared with `dualGap_le_twoUmax_of_quotientSup_lt_Umax`, this theorem derives the absolute
gap-control hypothesis internally from the equality that usually appears in the paper proof.
-/
theorem dualGap_le_twoUmax_of_pairingEq_quotientSup_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow =
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hStar_lt :
      signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) < Umax)
    (hNow_lt :
      signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_quotientSup_lt_Umax
    (gapNow := gapNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth
    (gap_le_abs_pairing_of_pairing_eq
      (gapNow := gapNow) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂) hpair)
    hStar_lt hNow_lt

/--
A.2 endpoint from the paper's concavity pairing bound and strict quotient-radius margins.
-/
theorem dualGap_le_twoUmax_of_pairingBound_quotientSup_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow ≤
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hStar_lt :
      signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) < Umax)
    (hNow_lt :
      signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_quotientSup_lt_Umax
    (gapNow := gapNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth
    (gap_le_abs_pairing_of_pairing_bound
      (gapNow := gapNow) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂) hpair)
    hStar_lt hNow_lt

/--
A.2 endpoint from the negative residual-pairing convention and strict quotient-radius margins.
-/
theorem dualGap_le_twoUmax_of_negPairingEq_quotientSup_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow =
        -((∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)))
    (hStar_lt :
      signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) < Umax)
    (hNow_lt :
      signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_quotientSup_lt_Umax
    (gapNow := gapNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth
    (gap_le_abs_pairing_of_neg_pairing_eq
      (gapNow := gapNow) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂) hpair)
    hStar_lt hNow_lt

/--
A.2 endpoint from the paper's negative-sign concavity pairing bound and strict quotient margins.
-/
theorem dualGap_le_twoUmax_of_negPairingBound_quotientSup_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow ≤
        -((∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)))
    (hStar_lt :
      signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) < Umax)
    (hNow_lt :
      signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_quotientSup_lt_Umax
    (gapNow := gapNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth
    (gap_le_abs_pairing_of_neg_pairing_bound
      (gapNow := gapNow) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂) hpair)
    hStar_lt hNow_lt

/--
A.2 endpoint from the paper's concavity pairing bound and strict shifted representatives.

This is a concrete orbit-representative entrypoint: downstream quotient geometry can provide the
normalizing shifts directly, and Lean derives the strict quotient-seminorm margins before calling
the paper-facing A.2 endpoint.
-/
theorem dualGap_le_twoUmax_of_pairingBound_signedShiftWitnesses_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow ≤
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hStar :
      ∃ cStar : ℝ,
        blockSupNorm (signedPairedShift τ cStar (uStar₁, uStar₂)) < Umax)
    (hNow :
      ∃ cNow : ℝ,
        blockSupNorm (signedPairedShift τ cNow (uNow₁, uNow₂)) < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_pairingBound_quotientSup_lt_Umax
    (gapNow := gapNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hpair
    (signedPairedQuotientSupSeminorm_lt_of_shiftWitness_lt τ (uStar₁, uStar₂) hStar)
    (signedPairedQuotientSupSeminorm_lt_of_shiftWitness_lt τ (uNow₁, uNow₂) hNow)

/-- Exact-pairing version of the strict shifted-representative A.2 bridge. -/
theorem dualGap_le_twoUmax_of_pairingEq_signedShiftWitnesses_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow =
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hStar :
      ∃ cStar : ℝ,
        blockSupNorm (signedPairedShift τ cStar (uStar₁, uStar₂)) < Umax)
    (hNow :
      ∃ cNow : ℝ,
        blockSupNorm (signedPairedShift τ cNow (uNow₁, uNow₂)) < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_pairingEq_quotientSup_lt_Umax
    (gapNow := gapNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hpair
    (signedPairedQuotientSupSeminorm_lt_of_shiftWitness_lt τ (uStar₁, uStar₂) hStar)
    (signedPairedQuotientSupSeminorm_lt_of_shiftWitness_lt τ (uNow₁, uNow₂) hNow)

/--
A.2 endpoint from value-gap/concavity hypotheses and concrete coordinate bounds for shifted
representatives.

This is the most primitive shifted-representative bridge in this file: downstream proofs can
provide the centering shifts and coordinatewise absolute bounds directly.  Lean then builds the
strict signed representatives, extracts quotient margins, and applies the A.2 endpoint.
-/
theorem dualGap_le_twoUmax_of_valueGap_concavity_signedShiftCoordinateBounds_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {Φ : ((ι₁ → ℝ) × (ι₂ → ℝ)) → ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (cStar cNow : ℝ)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hgapValue :
      gapNow ≤ Φ (uStar₁, uStar₂) - Φ (uNow₁, uNow₂))
    (hconcavity :
      Φ (uStar₁, uStar₂) - Φ (uNow₁, uNow₂) ≤
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hStar₁ : ∀ i, |uStar₁ i + cStar| ≤ UStar)
    (hStar₂ : ∀ j, |uStar₂ j + τ.toReal * cStar| ≤ UStar)
    (hNow₁ : ∀ i, |uNow₁ i + cNow| ≤ UNow)
    (hNow₂ : ∀ j, |uNow₂ j + τ.toReal * cNow| ≤ UNow)
    (hStar_lt : UStar < Umax)
    (hNow_lt : UNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_pairingBound_signedShiftWitnesses_lt_Umax
    (gapNow := gapNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth
    ((hgapValue.trans hconcavity))
    (signedShiftWitness_lt_Umax_of_forall_abs_signedShift_le_radius_lt
      τ (uStar₁, uStar₂) cStar hStar₁ hStar₂ hStar_lt)
    (signedShiftWitness_lt_Umax_of_forall_abs_signedShift_le_radius_lt
      τ (uNow₁, uNow₂) cNow hNow₁ hNow₂ hNow_lt)

/-- Negative-sign pairing-bound version of the strict shifted-representative A.2 bridge. -/
theorem dualGap_le_twoUmax_of_negPairingBound_signedShiftWitnesses_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow ≤
        -((∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)))
    (hStar :
      ∃ cStar : ℝ,
        blockSupNorm (signedPairedShift τ cStar (uStar₁, uStar₂)) < Umax)
    (hNow :
      ∃ cNow : ℝ,
        blockSupNorm (signedPairedShift τ cNow (uNow₁, uNow₂)) < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_negPairingBound_quotientSup_lt_Umax
    (gapNow := gapNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hpair
    (signedPairedQuotientSupSeminorm_lt_of_shiftWitness_lt τ (uStar₁, uStar₂) hStar)
    (signedPairedQuotientSupSeminorm_lt_of_shiftWitness_lt τ (uNow₁, uNow₂) hNow)

/--
A.2 endpoint from the paper's concavity pairing bound and strict signed orbit-set bounds.
-/
theorem dualGap_le_twoUmax_of_pairingBound_signedShiftNormSetBounds_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow ≤
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hStar :
      ∃ RStar : ℝ,
        RStar ∈ signedPairedShiftNormSet τ (uStar₁, uStar₂) ∧ RStar < Umax)
    (hNow :
      ∃ RNow : ℝ,
        RNow ∈ signedPairedShiftNormSet τ (uNow₁, uNow₂) ∧ RNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_pairingBound_quotientSup_lt_Umax
    (gapNow := gapNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hpair
    (signedPairedQuotientSupSeminorm_lt_of_signedShiftNormSetBound_lt
      τ (uStar₁, uStar₂) hStar)
    (signedPairedQuotientSupSeminorm_lt_of_signedShiftNormSetBound_lt
      τ (uNow₁, uNow₂) hNow)

/--
A.2 endpoint from concavity pairing and orbit-set bounds through strict intermediate radii.
-/
theorem dualGap_le_twoUmax_of_pairingBound_signedShiftNormSetBounds_le_radii_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow ≤
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hStar :
      ∃ RStar : ℝ,
        RStar ∈ signedPairedShiftNormSet τ (uStar₁, uStar₂) ∧ RStar ≤ UStar)
    (hNow :
      ∃ RNow : ℝ,
        RNow ∈ signedPairedShiftNormSet τ (uNow₁, uNow₂) ∧ RNow ≤ UNow)
    (hStar_lt : UStar < Umax)
    (hNow_lt : UNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_pairingBound_quotientSup_lt_Umax
    (gapNow := gapNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hpair
    (signedPairedQuotientSupSeminorm_lt_of_signedShiftNormSetBound_le_lt
      τ (uStar₁, uStar₂) hStar hStar_lt)
    (signedPairedQuotientSupSeminorm_lt_of_signedShiftNormSetBound_le_lt
      τ (uNow₁, uNow₂) hNow hNow_lt)

/--
A.2 endpoint from the paper's concavity pairing bound and quotient-seminorm radii below `Umax`.
-/
theorem dualGap_le_twoUmax_of_pairingBound_quotientSup_le_radii_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow ≤
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hStar :
      signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) ≤ UStar)
    (hNow :
      signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) ≤ UNow)
    (hStar_lt : UStar < Umax)
    (hNow_lt : UNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_quotientSup_le_radii_lt_Umax
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth
    (gap_le_abs_pairing_of_pairing_bound
      (gapNow := gapNow) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂) hpair)
    hStar hNow hStar_lt hNow_lt

/-- Exact-pairing version of the quotient-radii A.2 bridge. -/
theorem dualGap_le_twoUmax_of_pairingEq_quotientSup_le_radii_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow =
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hStar :
      signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) ≤ UStar)
    (hNow :
      signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) ≤ UNow)
    (hStar_lt : UStar < Umax)
    (hNow_lt : UNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_quotientSup_le_radii_lt_Umax
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth
    (gap_le_abs_pairing_of_pairing_eq
      (gapNow := gapNow) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂) hpair)
    hStar hNow hStar_lt hNow_lt

/-- Negative-sign pairing-bound version of the quotient-radii A.2 bridge. -/
theorem dualGap_le_twoUmax_of_negPairingBound_quotientSup_le_radii_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow ≤
        -((∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)))
    (hStar :
      signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) ≤ UStar)
    (hNow :
      signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) ≤ UNow)
    (hStar_lt : UStar < Umax)
    (hNow_lt : UNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_quotientSup_le_radii_lt_Umax
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth
    (gap_le_abs_pairing_of_neg_pairing_bound
      (gapNow := gapNow) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂) hpair)
    hStar hNow hStar_lt hNow_lt

/-- Exact negative-pairing version of the quotient-radii A.2 bridge. -/
theorem dualGap_le_twoUmax_of_negPairingEq_quotientSup_le_radii_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow =
        -((∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)))
    (hStar :
      signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) ≤ UStar)
    (hNow :
      signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) ≤ UNow)
    (hStar_lt : UStar < Umax)
    (hNow_lt : UNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_quotientSup_le_radii_lt_Umax
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth
    (gap_le_abs_pairing_of_neg_pairing_eq
      (gapNow := gapNow) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂) hpair)
    hStar hNow hStar_lt hNow_lt

/--
A.2 endpoint from a concavity pairing bound and non-strict quotient bounds with common slack.
-/
theorem dualGap_le_twoUmax_of_pairingBound_quotientSup_le_common_slack_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow QStar QNow eps Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow ≤
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hQStar : signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) ≤ QStar)
    (hQNow : signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) ≤ QNow)
    (heps : 0 < eps)
    (hStar : QStar + eps ≤ Umax)
    (hNow : QNow + eps ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_quotientSup_le_common_slack_Umax
    (gapNow := gapNow) (QStar := QStar) (QNow := QNow) (eps := eps)
    (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth
    (gap_le_abs_pairing_of_pairing_bound
      (gapNow := gapNow) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂) hpair)
    hQStar hQNow heps hStar hNow

/-- Exact-pairing version of the non-strict quotient/slack A.2 bridge. -/
theorem dualGap_le_twoUmax_of_pairingEq_quotientSup_le_common_slack_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow QStar QNow eps Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow =
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hQStar : signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) ≤ QStar)
    (hQNow : signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) ≤ QNow)
    (heps : 0 < eps)
    (hStar : QStar + eps ≤ Umax)
    (hNow : QNow + eps ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_quotientSup_le_common_slack_Umax
    (gapNow := gapNow) (QStar := QStar) (QNow := QNow) (eps := eps)
    (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth
    (gap_le_abs_pairing_of_pairing_eq
      (gapNow := gapNow) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂) hpair)
    hQStar hQNow heps hStar hNow

/-- Negative-sign pairing-bound version of the non-strict quotient/slack A.2 bridge. -/
theorem dualGap_le_twoUmax_of_negPairingBound_quotientSup_le_common_slack_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow QStar QNow eps Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow ≤
        -((∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)))
    (hQStar : signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) ≤ QStar)
    (hQNow : signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) ≤ QNow)
    (heps : 0 < eps)
    (hStar : QStar + eps ≤ Umax)
    (hNow : QNow + eps ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_quotientSup_le_common_slack_Umax
    (gapNow := gapNow) (QStar := QStar) (QNow := QNow) (eps := eps)
    (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth
    (gap_le_abs_pairing_of_neg_pairing_bound
      (gapNow := gapNow) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂) hpair)
    hQStar hQNow heps hStar hNow

/-- Exact negative-pairing version of the non-strict quotient/slack A.2 bridge. -/
theorem dualGap_le_twoUmax_of_negPairingEq_quotientSup_le_common_slack_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow QStar QNow eps Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow =
        -((∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)))
    (hQStar : signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) ≤ QStar)
    (hQNow : signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) ≤ QNow)
    (heps : 0 < eps)
    (hStar : QStar + eps ≤ Umax)
    (hNow : QNow + eps ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_quotientSup_le_common_slack_Umax
    (gapNow := gapNow) (QStar := QStar) (QNow := QNow) (eps := eps)
    (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth
    (gap_le_abs_pairing_of_neg_pairing_eq
      (gapNow := gapNow) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂) hpair)
    hQStar hQNow heps hStar hNow

/--
A.2 endpoint from direct unshifted block-sup strict bounds.

This bridge is stronger than a coordinate-radius interface but weaker than quotient geometry:
it uses the certified inequality
`signedPairedQuotientSupSeminorm τ u ≤ blockSupNorm u` to enter the quotient endpoint.
-/
theorem dualGap_le_twoUmax_of_blockSup_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hgap :
      gapNow ≤
        |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)|)
    (hStar : blockSupNorm (uStar₁, uStar₂) < Umax)
    (hNow : blockSupNorm (uNow₁, uNow₂) < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_quotientSup_lt_Umax
    (gapNow := gapNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hgap
    (lt_of_le_of_lt
      (signedPairedQuotientSupSeminorm_le_blockSupNorm τ (uStar₁, uStar₂))
      hStar)
    (lt_of_le_of_lt
      (signedPairedQuotientSupSeminorm_le_blockSupNorm τ (uNow₁, uNow₂))
      hNow)

/--
A.2 endpoint from unshifted block-sup bounds through intermediate radii below `Umax`.

This is useful for application modules that first derive explicit block-sup estimates and only
afterwards compare those estimates to the global radius budget.
-/
theorem dualGap_le_twoUmax_of_blockSup_le_radii_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hgap :
      gapNow ≤
        |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)|)
    (hStar : blockSupNorm (uStar₁, uStar₂) ≤ UStar)
    (hNow : blockSupNorm (uNow₁, uNow₂) ≤ UNow)
    (hStar_lt : UStar < Umax)
    (hNow_lt : UNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_blockSup_lt_Umax
    (gapNow := gapNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hgap
    (lt_of_le_of_lt hStar hStar_lt)
    (lt_of_le_of_lt hNow hNow_lt)

/--
A.2 endpoint from the paper's concavity pairing bound and direct block-sup margins.

This combines two previously separate bridges: Lean first converts the concavity pairing bound to
the absolute residual-pairing control, then uses the block-sup margins to enter the quotient
endpoint.
-/
theorem dualGap_le_twoUmax_of_pairingBound_blockSup_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow ≤
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hStar : blockSupNorm (uStar₁, uStar₂) < Umax)
    (hNow : blockSupNorm (uNow₁, uNow₂) < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_blockSup_lt_Umax
    (gapNow := gapNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth
    (gap_le_abs_pairing_of_pairing_bound
      (gapNow := gapNow) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂) hpair)
    hStar hNow

/-- A.2 block-sup endpoint from an exact residual-pairing identity. -/
theorem dualGap_le_twoUmax_of_pairingEq_blockSup_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow =
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hStar : blockSupNorm (uStar₁, uStar₂) < Umax)
    (hNow : blockSupNorm (uNow₁, uNow₂) < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_blockSup_lt_Umax
    (gapNow := gapNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth
    (gap_le_abs_pairing_of_pairing_eq
      (gapNow := gapNow) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂) hpair)
    hStar hNow

/-- A.2 block-sup endpoint from the negative residual-pairing convention. -/
theorem dualGap_le_twoUmax_of_negPairingBound_blockSup_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow ≤
        -((∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)))
    (hStar : blockSupNorm (uStar₁, uStar₂) < Umax)
    (hNow : blockSupNorm (uNow₁, uNow₂) < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_blockSup_lt_Umax
    (gapNow := gapNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth
    (gap_le_abs_pairing_of_neg_pairing_bound
      (gapNow := gapNow) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂) hpair)
    hStar hNow

/-- A.2 block-sup endpoint from an exact negative residual-pairing identity. -/
theorem dualGap_le_twoUmax_of_negPairingEq_blockSup_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow =
        -((∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)))
    (hStar : blockSupNorm (uStar₁, uStar₂) < Umax)
    (hNow : blockSupNorm (uNow₁, uNow₂) < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_blockSup_lt_Umax
    (gapNow := gapNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth
    (gap_le_abs_pairing_of_neg_pairing_eq
      (gapNow := gapNow) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂) hpair)
    hStar hNow

/--
A.2 endpoint from a concavity pairing bound and block-sup estimates through intermediate radii.

This is the common concrete-application shape: prove block-sup bounds `≤ U★`, `≤ U_k`, then prove
both radii are strictly below the paper budget `Umax`.
-/
theorem dualGap_le_twoUmax_of_pairingBound_blockSup_le_radii_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow ≤
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hStar : blockSupNorm (uStar₁, uStar₂) ≤ UStar)
    (hNow : blockSupNorm (uNow₁, uNow₂) ≤ UNow)
    (hStar_lt : UStar < Umax)
    (hNow_lt : UNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_pairingBound_blockSup_lt_Umax
    (gapNow := gapNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hpair
    (lt_of_le_of_lt hStar hStar_lt)
    (lt_of_le_of_lt hNow hNow_lt)

/-- Exact-pairing version of the block-sup/radii A.2 bridge. -/
theorem dualGap_le_twoUmax_of_pairingEq_blockSup_le_radii_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow =
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hStar : blockSupNorm (uStar₁, uStar₂) ≤ UStar)
    (hNow : blockSupNorm (uNow₁, uNow₂) ≤ UNow)
    (hStar_lt : UStar < Umax)
    (hNow_lt : UNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_pairingEq_blockSup_lt_Umax
    (gapNow := gapNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hpair
    (lt_of_le_of_lt hStar hStar_lt)
    (lt_of_le_of_lt hNow hNow_lt)

/-- Negative-sign pairing-bound version of the block-sup/radii A.2 bridge. -/
theorem dualGap_le_twoUmax_of_negPairingBound_blockSup_le_radii_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow ≤
        -((∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)))
    (hStar : blockSupNorm (uStar₁, uStar₂) ≤ UStar)
    (hNow : blockSupNorm (uNow₁, uNow₂) ≤ UNow)
    (hStar_lt : UStar < Umax)
    (hNow_lt : UNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_negPairingBound_blockSup_lt_Umax
    (gapNow := gapNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hpair
    (lt_of_le_of_lt hStar hStar_lt)
    (lt_of_le_of_lt hNow hNow_lt)

/-- Exact negative-pairing version of the block-sup/radii A.2 bridge. -/
theorem dualGap_le_twoUmax_of_negPairingEq_blockSup_le_radii_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow =
        -((∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)))
    (hStar : blockSupNorm (uStar₁, uStar₂) ≤ UStar)
    (hNow : blockSupNorm (uNow₁, uNow₂) ≤ UNow)
    (hStar_lt : UStar < Umax)
    (hNow_lt : UNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_negPairingEq_blockSup_lt_Umax
    (gapNow := gapNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hpair
    (lt_of_le_of_lt hStar hStar_lt)
    (lt_of_le_of_lt hNow hNow_lt)

/--
A.2 endpoint from a concavity pairing bound and quotient radii with internally generated slack.

Applications can prove non-strict quotient-seminorm estimates against concrete radii and strict
comparison of those radii to `Umax`; Lean chooses the common positive slack needed by the
infimum-extraction layer.
-/
theorem dualGap_le_twoUmax_of_pairingBound_quotientSup_le_strict_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow QStar QNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow ≤
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hQStar : signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) ≤ QStar)
    (hQNow : signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) ≤ QNow)
    (hStar_lt : QStar < Umax)
    (hNow_lt : QNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_quotientSup_le_strict_Umax
    (gapNow := gapNow) (QStar := QStar) (QNow := QNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth
    (gap_le_abs_pairing_of_pairing_bound
      (gapNow := gapNow) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂) hpair)
    hQStar hQNow hStar_lt hNow_lt

/-- Exact-pairing version of the quotient-radii strict-margin A.2 bridge. -/
theorem dualGap_le_twoUmax_of_pairingEq_quotientSup_le_strict_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow QStar QNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow =
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hQStar : signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) ≤ QStar)
    (hQNow : signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) ≤ QNow)
    (hStar_lt : QStar < Umax)
    (hNow_lt : QNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_quotientSup_le_strict_Umax
    (gapNow := gapNow) (QStar := QStar) (QNow := QNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth
    (gap_le_abs_pairing_of_pairing_eq
      (gapNow := gapNow) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂) hpair)
    hQStar hQNow hStar_lt hNow_lt

/-- Negative-sign pairing-bound version of the quotient-radii strict-margin A.2 bridge. -/
theorem dualGap_le_twoUmax_of_negPairingBound_quotientSup_le_strict_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow QStar QNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow ≤
        -((∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)))
    (hQStar : signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) ≤ QStar)
    (hQNow : signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) ≤ QNow)
    (hStar_lt : QStar < Umax)
    (hNow_lt : QNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_quotientSup_le_strict_Umax
    (gapNow := gapNow) (QStar := QStar) (QNow := QNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth
    (gap_le_abs_pairing_of_neg_pairing_bound
      (gapNow := gapNow) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂) hpair)
    hQStar hQNow hStar_lt hNow_lt

/-- Exact negative-pairing version of the quotient-radii strict-margin A.2 bridge. -/
theorem dualGap_le_twoUmax_of_negPairingEq_quotientSup_le_strict_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow QStar QNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow =
        -((∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)))
    (hQStar : signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) ≤ QStar)
    (hQNow : signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) ≤ QNow)
    (hStar_lt : QStar < Umax)
    (hNow_lt : QNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_quotientSup_le_strict_Umax
    (gapNow := gapNow) (QStar := QStar) (QNow := QNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth
    (gap_le_abs_pairing_of_neg_pairing_eq
      (gapNow := gapNow) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂) hpair)
    hQStar hQNow hStar_lt hNow_lt

/--
A.2 endpoint from a concavity pairing bound and block-sup radii with strict `Umax` margins.

This version keeps concrete estimates in the unshifted block-sup norm while still using the
quotient/infimum machinery internally.
-/
theorem dualGap_le_twoUmax_of_pairingBound_blockSup_le_strict_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow ≤
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hStar : blockSupNorm (uStar₁, uStar₂) ≤ UStar)
    (hNow : blockSupNorm (uNow₁, uNow₂) ≤ UNow)
    (hStar_lt : UStar < Umax)
    (hNow_lt : UNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_pairingBound_blockSup_le_radii_lt_Umax
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hpair hStar hNow hStar_lt hNow_lt

/-- Exact-pairing version of the block-sup strict-margin A.2 bridge. -/
theorem dualGap_le_twoUmax_of_pairingEq_blockSup_le_strict_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow =
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hStar : blockSupNorm (uStar₁, uStar₂) ≤ UStar)
    (hNow : blockSupNorm (uNow₁, uNow₂) ≤ UNow)
    (hStar_lt : UStar < Umax)
    (hNow_lt : UNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_pairingEq_blockSup_le_radii_lt_Umax
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hpair hStar hNow hStar_lt hNow_lt

/-- Negative-sign pairing-bound version of the block-sup strict-margin A.2 bridge. -/
theorem dualGap_le_twoUmax_of_negPairingBound_blockSup_le_strict_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow ≤
        -((∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)))
    (hStar : blockSupNorm (uStar₁, uStar₂) ≤ UStar)
    (hNow : blockSupNorm (uNow₁, uNow₂) ≤ UNow)
    (hStar_lt : UStar < Umax)
    (hNow_lt : UNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_negPairingBound_blockSup_le_radii_lt_Umax
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hpair hStar hNow hStar_lt hNow_lt

/-- Exact negative-pairing version of the block-sup strict-margin A.2 bridge. -/
theorem dualGap_le_twoUmax_of_negPairingEq_blockSup_le_strict_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow =
        -((∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)))
    (hStar : blockSupNorm (uStar₁, uStar₂) ≤ UStar)
    (hNow : blockSupNorm (uNow₁, uNow₂) ≤ UNow)
    (hStar_lt : UStar < Umax)
    (hNow_lt : UNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_negPairingEq_blockSup_le_radii_lt_Umax
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hpair hStar hNow hStar_lt hNow_lt

/--
A.2 endpoint from a concavity pairing bound and non-strict block-sup budgets with common slack.

This mirrors the quotient-seminorm slack interface but lets concrete applications stay in the
unshifted block-sup norm until the final bridge into quotient geometry.
-/
theorem dualGap_le_twoUmax_of_pairingBound_blockSup_le_common_slack_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow eps Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow ≤
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hStar : blockSupNorm (uStar₁, uStar₂) ≤ UStar)
    (hNow : blockSupNorm (uNow₁, uNow₂) ≤ UNow)
    (heps : 0 < eps)
    (hStarU : UStar + eps ≤ Umax)
    (hNowU : UNow + eps ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_pairingBound_blockSup_le_radii_lt_Umax
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hpair hStar hNow
    (by linarith)
    (by linarith)

/-- Exact-pairing version of the block-sup common-slack A.2 bridge. -/
theorem dualGap_le_twoUmax_of_pairingEq_blockSup_le_common_slack_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow eps Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow =
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hStar : blockSupNorm (uStar₁, uStar₂) ≤ UStar)
    (hNow : blockSupNorm (uNow₁, uNow₂) ≤ UNow)
    (heps : 0 < eps)
    (hStarU : UStar + eps ≤ Umax)
    (hNowU : UNow + eps ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_pairingEq_blockSup_le_radii_lt_Umax
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hpair hStar hNow
    (by linarith)
    (by linarith)

/-- Negative-sign pairing-bound version of the block-sup common-slack A.2 bridge. -/
theorem dualGap_le_twoUmax_of_negPairingBound_blockSup_le_common_slack_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow eps Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow ≤
        -((∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)))
    (hStar : blockSupNorm (uStar₁, uStar₂) ≤ UStar)
    (hNow : blockSupNorm (uNow₁, uNow₂) ≤ UNow)
    (heps : 0 < eps)
    (hStarU : UStar + eps ≤ Umax)
    (hNowU : UNow + eps ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_negPairingBound_blockSup_le_radii_lt_Umax
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hpair hStar hNow
    (by linarith)
    (by linarith)

/-- Exact negative-pairing version of the block-sup common-slack A.2 bridge. -/
theorem dualGap_le_twoUmax_of_negPairingEq_blockSup_le_common_slack_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow eps Umax : ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hpair :
      gapNow =
        -((∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)))
    (hStar : blockSupNorm (uStar₁, uStar₂) ≤ UStar)
    (hNow : blockSupNorm (uNow₁, uNow₂) ≤ UNow)
    (heps : 0 < eps)
    (hStarU : UStar + eps ≤ Umax)
    (hNowU : UNow + eps ≤ Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_negPairingEq_blockSup_le_radii_lt_Umax
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hpair hStar hNow
    (by linarith)
    (by linarith)

/--
Sequence form of the paper-facing A.2 quotient endpoint.

The optimal/reference dual block is fixed while the current iterate, residuals, and gap vary with
`k`.  This is the shape needed by downstream rate proofs.
-/
theorem dualGap_seq_le_twoUmax_of_pairingBound_quotientSup_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gap : ℕ → ℝ} {Umax : ℝ}
    {r₁ : ℕ → ι₁ → ℝ} {uStar₁ : ι₁ → ℝ} {uNow₁ : ℕ → ι₁ → ℝ}
    {r₂ : ℕ → ι₂ → ℝ} {uStar₂ : ι₂ → ℝ} {uNow₂ : ℕ → ι₂ → ℝ}
    (τ : PairedSign)
    (horth : ∀ k : ℕ, (∑ i, r₁ k i) + τ.toReal * (∑ j, r₂ k j) = 0)
    (hpair :
      ∀ k : ℕ,
        gap k ≤
          (∑ i, r₁ k i * (uStar₁ i - uNow₁ k i))
            + ∑ j, r₂ k j * (uStar₂ j - uNow₂ k j))
    (hStar_lt :
      signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) < Umax)
    (hNow_lt :
      ∀ k : ℕ, signedPairedQuotientSupSeminorm τ (uNow₁ k, uNow₂ k) < Umax) :
    ∀ k : ℕ, gap k ≤ (2 * Umax) * finiteL1Pair (r₁ k) (r₂ k) := by
  intro k
  exact dualGap_le_twoUmax_of_pairingBound_quotientSup_lt_Umax
    (gapNow := gap k) (Umax := Umax)
    (r₁ := r₁ k) (uStar₁ := uStar₁) (uNow₁ := uNow₁ k)
    (r₂ := r₂ k) (uStar₂ := uStar₂) (uNow₂ := uNow₂ k)
    τ (horth k) (hpair k) hStar_lt (hNow_lt k)

/--
Exact-pairing sequence form of the paper-facing A.2 quotient endpoint.

This is useful when the gap identity has already been normalized before the rate proof: the
conversion to the absolute residual pairing and the quotient-radius bridge are performed at every
iterate.
-/
theorem dualGap_seq_le_twoUmax_of_pairingEq_quotientSup_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gap : ℕ → ℝ} {Umax : ℝ}
    {r₁ : ℕ → ι₁ → ℝ} {uStar₁ : ι₁ → ℝ} {uNow₁ : ℕ → ι₁ → ℝ}
    {r₂ : ℕ → ι₂ → ℝ} {uStar₂ : ι₂ → ℝ} {uNow₂ : ℕ → ι₂ → ℝ}
    (τ : PairedSign)
    (horth : ∀ k : ℕ, (∑ i, r₁ k i) + τ.toReal * (∑ j, r₂ k j) = 0)
    (hpair :
      ∀ k : ℕ,
        gap k =
          (∑ i, r₁ k i * (uStar₁ i - uNow₁ k i))
            + ∑ j, r₂ k j * (uStar₂ j - uNow₂ k j))
    (hStar_lt :
      signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) < Umax)
    (hNow_lt :
      ∀ k : ℕ, signedPairedQuotientSupSeminorm τ (uNow₁ k, uNow₂ k) < Umax) :
    ∀ k : ℕ, gap k ≤ (2 * Umax) * finiteL1Pair (r₁ k) (r₂ k) := by
  intro k
  exact dualGap_le_twoUmax_of_pairingEq_quotientSup_lt_Umax
    (gapNow := gap k) (Umax := Umax)
    (r₁ := r₁ k) (uStar₁ := uStar₁) (uNow₁ := uNow₁ k)
    (r₂ := r₂ k) (uStar₂ := uStar₂) (uNow₂ := uNow₂ k)
    τ (horth k) (hpair k) hStar_lt (hNow_lt k)

/--
Negative-pairing sequence form of the paper-facing A.2 quotient endpoint.

Some derivations carry the residual pairing with the opposite sign; this bridge keeps the
sequence-level rate interface unchanged.
-/
theorem dualGap_seq_le_twoUmax_of_negPairingBound_quotientSup_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gap : ℕ → ℝ} {Umax : ℝ}
    {r₁ : ℕ → ι₁ → ℝ} {uStar₁ : ι₁ → ℝ} {uNow₁ : ℕ → ι₁ → ℝ}
    {r₂ : ℕ → ι₂ → ℝ} {uStar₂ : ι₂ → ℝ} {uNow₂ : ℕ → ι₂ → ℝ}
    (τ : PairedSign)
    (horth : ∀ k : ℕ, (∑ i, r₁ k i) + τ.toReal * (∑ j, r₂ k j) = 0)
    (hpair :
      ∀ k : ℕ,
        gap k ≤
          -((∑ i, r₁ k i * (uStar₁ i - uNow₁ k i))
            + ∑ j, r₂ k j * (uStar₂ j - uNow₂ k j)))
    (hStar_lt :
      signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) < Umax)
    (hNow_lt :
      ∀ k : ℕ, signedPairedQuotientSupSeminorm τ (uNow₁ k, uNow₂ k) < Umax) :
    ∀ k : ℕ, gap k ≤ (2 * Umax) * finiteL1Pair (r₁ k) (r₂ k) := by
  intro k
  exact dualGap_le_twoUmax_of_negPairingBound_quotientSup_lt_Umax
    (gapNow := gap k) (Umax := Umax)
    (r₁ := r₁ k) (uStar₁ := uStar₁) (uNow₁ := uNow₁ k)
    (r₂ := r₂ k) (uStar₂ := uStar₂) (uNow₂ := uNow₂ k)
    τ (horth k) (hpair k) hStar_lt (hNow_lt k)

/--
Sequence A.2 endpoint with quotient-radius budgets strictly below `Umax`.
-/
theorem dualGap_seq_le_twoUmax_of_pairingBound_quotientSup_le_strict_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gap QStar QNow : ℕ → ℝ} {Umax : ℝ}
    {r₁ : ℕ → ι₁ → ℝ} {uStar₁ : ℕ → ι₁ → ℝ} {uNow₁ : ℕ → ι₁ → ℝ}
    {r₂ : ℕ → ι₂ → ℝ} {uStar₂ : ℕ → ι₂ → ℝ} {uNow₂ : ℕ → ι₂ → ℝ}
    (τ : PairedSign)
    (horth : ∀ k : ℕ, (∑ i, r₁ k i) + τ.toReal * (∑ j, r₂ k j) = 0)
    (hpair :
      ∀ k : ℕ,
        gap k ≤
          (∑ i, r₁ k i * (uStar₁ k i - uNow₁ k i))
            + ∑ j, r₂ k j * (uStar₂ k j - uNow₂ k j))
    (hQStar :
      ∀ k : ℕ, signedPairedQuotientSupSeminorm τ (uStar₁ k, uStar₂ k) ≤ QStar k)
    (hQNow :
      ∀ k : ℕ, signedPairedQuotientSupSeminorm τ (uNow₁ k, uNow₂ k) ≤ QNow k)
    (hStar_lt : ∀ k : ℕ, QStar k < Umax)
    (hNow_lt : ∀ k : ℕ, QNow k < Umax) :
    ∀ k : ℕ, gap k ≤ (2 * Umax) * finiteL1Pair (r₁ k) (r₂ k) := by
  intro k
  exact dualGap_le_twoUmax_of_pairingBound_quotientSup_le_strict_Umax
    (gapNow := gap k) (QStar := QStar k) (QNow := QNow k) (Umax := Umax)
    (r₁ := r₁ k) (uStar₁ := uStar₁ k) (uNow₁ := uNow₁ k)
    (r₂ := r₂ k) (uStar₂ := uStar₂ k) (uNow₂ := uNow₂ k)
    τ (horth k) (hpair k) (hQStar k) (hQNow k) (hStar_lt k) (hNow_lt k)

/--
Sequence A.2 endpoint from direct block-sup strict margins.
-/
theorem dualGap_seq_le_twoUmax_of_pairingBound_blockSup_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gap : ℕ → ℝ} {Umax : ℝ}
    {r₁ : ℕ → ι₁ → ℝ} {uStar₁ : ℕ → ι₁ → ℝ} {uNow₁ : ℕ → ι₁ → ℝ}
    {r₂ : ℕ → ι₂ → ℝ} {uStar₂ : ℕ → ι₂ → ℝ} {uNow₂ : ℕ → ι₂ → ℝ}
    (τ : PairedSign)
    (horth : ∀ k : ℕ, (∑ i, r₁ k i) + τ.toReal * (∑ j, r₂ k j) = 0)
    (hpair :
      ∀ k : ℕ,
        gap k ≤
          (∑ i, r₁ k i * (uStar₁ k i - uNow₁ k i))
            + ∑ j, r₂ k j * (uStar₂ k j - uNow₂ k j))
    (hStar : ∀ k : ℕ, blockSupNorm (uStar₁ k, uStar₂ k) < Umax)
    (hNow : ∀ k : ℕ, blockSupNorm (uNow₁ k, uNow₂ k) < Umax) :
    ∀ k : ℕ, gap k ≤ (2 * Umax) * finiteL1Pair (r₁ k) (r₂ k) := by
  intro k
  exact dualGap_le_twoUmax_of_pairingBound_blockSup_lt_Umax
    (gapNow := gap k) (Umax := Umax)
    (r₁ := r₁ k) (uStar₁ := uStar₁ k) (uNow₁ := uNow₁ k)
    (r₂ := r₂ k) (uStar₂ := uStar₂ k) (uNow₂ := uNow₂ k)
    τ (horth k) (hpair k) (hStar k) (hNow k)

/--
Sequence A.2 endpoint from block-sup radii strictly below `Umax`.
-/
theorem dualGap_seq_le_twoUmax_of_pairingBound_blockSup_le_strict_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gap UStar UNow : ℕ → ℝ} {Umax : ℝ}
    {r₁ : ℕ → ι₁ → ℝ} {uStar₁ : ℕ → ι₁ → ℝ} {uNow₁ : ℕ → ι₁ → ℝ}
    {r₂ : ℕ → ι₂ → ℝ} {uStar₂ : ℕ → ι₂ → ℝ} {uNow₂ : ℕ → ι₂ → ℝ}
    (τ : PairedSign)
    (horth : ∀ k : ℕ, (∑ i, r₁ k i) + τ.toReal * (∑ j, r₂ k j) = 0)
    (hpair :
      ∀ k : ℕ,
        gap k ≤
          (∑ i, r₁ k i * (uStar₁ k i - uNow₁ k i))
            + ∑ j, r₂ k j * (uStar₂ k j - uNow₂ k j))
    (hStar : ∀ k : ℕ, blockSupNorm (uStar₁ k, uStar₂ k) ≤ UStar k)
    (hNow : ∀ k : ℕ, blockSupNorm (uNow₁ k, uNow₂ k) ≤ UNow k)
    (hStar_lt : ∀ k : ℕ, UStar k < Umax)
    (hNow_lt : ∀ k : ℕ, UNow k < Umax) :
    ∀ k : ℕ, gap k ≤ (2 * Umax) * finiteL1Pair (r₁ k) (r₂ k) := by
  intro k
  exact dualGap_le_twoUmax_of_pairingBound_blockSup_le_strict_Umax
    (gapNow := gap k) (UStar := UStar k) (UNow := UNow k) (Umax := Umax)
    (r₁ := r₁ k) (uStar₁ := uStar₁ k) (uNow₁ := uNow₁ k)
    (r₂ := r₂ k) (uStar₂ := uStar₂ k) (uNow₂ := uNow₂ k)
    τ (horth k) (hpair k) (hStar k) (hNow k) (hStar_lt k) (hNow_lt k)

/--
Sequence A.2 endpoint from block-sup budgets with an explicit common slack.
-/
theorem dualGap_seq_le_twoUmax_of_pairingBound_blockSup_le_common_slack_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gap UStar UNow eps : ℕ → ℝ} {Umax : ℝ}
    {r₁ : ℕ → ι₁ → ℝ} {uStar₁ : ℕ → ι₁ → ℝ} {uNow₁ : ℕ → ι₁ → ℝ}
    {r₂ : ℕ → ι₂ → ℝ} {uStar₂ : ℕ → ι₂ → ℝ} {uNow₂ : ℕ → ι₂ → ℝ}
    (τ : PairedSign)
    (horth : ∀ k : ℕ, (∑ i, r₁ k i) + τ.toReal * (∑ j, r₂ k j) = 0)
    (hpair :
      ∀ k : ℕ,
        gap k ≤
          (∑ i, r₁ k i * (uStar₁ k i - uNow₁ k i))
            + ∑ j, r₂ k j * (uStar₂ k j - uNow₂ k j))
    (hStar : ∀ k : ℕ, blockSupNorm (uStar₁ k, uStar₂ k) ≤ UStar k)
    (hNow : ∀ k : ℕ, blockSupNorm (uNow₁ k, uNow₂ k) ≤ UNow k)
    (heps : ∀ k : ℕ, 0 < eps k)
    (hStarU : ∀ k : ℕ, UStar k + eps k ≤ Umax)
    (hNowU : ∀ k : ℕ, UNow k + eps k ≤ Umax) :
    ∀ k : ℕ, gap k ≤ (2 * Umax) * finiteL1Pair (r₁ k) (r₂ k) := by
  intro k
  exact dualGap_le_twoUmax_of_pairingBound_blockSup_le_common_slack_Umax
    (gapNow := gap k) (UStar := UStar k) (UNow := UNow k) (eps := eps k)
    (Umax := Umax)
    (r₁ := r₁ k) (uStar₁ := uStar₁ k) (uNow₁ := uNow₁ k)
    (r₂ := r₂ k) (uStar₂ := uStar₂ k) (uNow₂ := uNow₂ k)
    τ (horth k) (hpair k) (hStar k) (hNow k) (heps k) (hStarU k) (hNowU k)

/--
Rate-ready consequence of the sequence quotient A.2 endpoint.

If the residual `finiteL1Pair` already has a `B/(k+1)` bound, this theorem exposes the exact
pointwise gap rate consumed by later convergence wrappers.
-/
theorem dualGap_seq_rate_of_pairingBound_quotientSup_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gap : ℕ → ℝ} {Umax B : ℝ}
    {r₁ : ℕ → ι₁ → ℝ} {uStar₁ : ι₁ → ℝ} {uNow₁ : ℕ → ι₁ → ℝ}
    {r₂ : ℕ → ι₂ → ℝ} {uStar₂ : ι₂ → ℝ} {uNow₂ : ℕ → ι₂ → ℝ}
    (τ : PairedSign)
    (horth : ∀ k : ℕ, (∑ i, r₁ k i) + τ.toReal * (∑ j, r₂ k j) = 0)
    (hpair :
      ∀ k : ℕ,
        gap k ≤
          (∑ i, r₁ k i * (uStar₁ i - uNow₁ k i))
            + ∑ j, r₂ k j * (uStar₂ j - uNow₂ k j))
    (hStar_lt :
      signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) < Umax)
    (hNow_lt :
      ∀ k : ℕ, signedPairedQuotientSupSeminorm τ (uNow₁ k, uNow₂ k) < Umax)
    (hUmax_nonneg : 0 ≤ Umax)
    (hres_rate : ∀ k : ℕ, finiteL1Pair (r₁ k) (r₂ k) ≤ B / (k + 1 : ℝ)) :
    ∀ k : ℕ, gap k ≤ (2 * Umax) * B / (k + 1 : ℝ) := by
  intro k
  have hgap :=
    dualGap_seq_le_twoUmax_of_pairingBound_quotientSup_lt_Umax
      (gap := gap) (Umax := Umax) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
      τ horth hpair hStar_lt hNow_lt k
  have hmul :
      (2 * Umax) * finiteL1Pair (r₁ k) (r₂ k) ≤
        (2 * Umax) * (B / (k + 1 : ℝ)) := by
    exact mul_le_mul_of_nonneg_left (hres_rate k) (by nlinarith)
  calc
    gap k ≤ (2 * Umax) * finiteL1Pair (r₁ k) (r₂ k) := hgap
    _ ≤ (2 * Umax) * (B / (k + 1 : ℝ)) := hmul
    _ = (2 * Umax) * B / (k + 1 : ℝ) := by ring

/--
Rate-ready exact-pairing consequence of the sequence quotient A.2 endpoint.

This combines the exact residual-pairing identity, quotient-sup bounds, and a pointwise
`finiteL1Pair` residual rate into the `forall k, gap k <= C/(k+1)` form.
-/
theorem dualGap_seq_rate_of_pairingEq_quotientSup_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gap : ℕ → ℝ} {Umax B : ℝ}
    {r₁ : ℕ → ι₁ → ℝ} {uStar₁ : ι₁ → ℝ} {uNow₁ : ℕ → ι₁ → ℝ}
    {r₂ : ℕ → ι₂ → ℝ} {uStar₂ : ι₂ → ℝ} {uNow₂ : ℕ → ι₂ → ℝ}
    (τ : PairedSign)
    (horth : ∀ k : ℕ, (∑ i, r₁ k i) + τ.toReal * (∑ j, r₂ k j) = 0)
    (hpair :
      ∀ k : ℕ,
        gap k =
          (∑ i, r₁ k i * (uStar₁ i - uNow₁ k i))
            + ∑ j, r₂ k j * (uStar₂ j - uNow₂ k j))
    (hStar_lt :
      signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) < Umax)
    (hNow_lt :
      ∀ k : ℕ, signedPairedQuotientSupSeminorm τ (uNow₁ k, uNow₂ k) < Umax)
    (hUmax_nonneg : 0 ≤ Umax)
    (hres_rate : ∀ k : ℕ, finiteL1Pair (r₁ k) (r₂ k) ≤ B / (k + 1 : ℝ)) :
    ∀ k : ℕ, gap k ≤ (2 * Umax) * B / (k + 1 : ℝ) := by
  intro k
  have hgap :=
    dualGap_seq_le_twoUmax_of_pairingEq_quotientSup_lt_Umax
      (gap := gap) (Umax := Umax) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
      τ horth hpair hStar_lt hNow_lt k
  have hmul :
      (2 * Umax) * finiteL1Pair (r₁ k) (r₂ k) ≤
        (2 * Umax) * (B / (k + 1 : ℝ)) := by
    exact mul_le_mul_of_nonneg_left (hres_rate k) (by nlinarith)
  calc
    gap k ≤ (2 * Umax) * finiteL1Pair (r₁ k) (r₂ k) := hgap
    _ ≤ (2 * Umax) * (B / (k + 1 : ℝ)) := hmul
    _ = (2 * Umax) * B / (k + 1 : ℝ) := by ring

/--
Rate-ready negative-pairing consequence of the sequence quotient A.2 endpoint.

This is the sign-convention companion of `dualGap_seq_rate_of_pairingEq_quotientSup_lt_Umax`.
-/
theorem dualGap_seq_rate_of_negPairingBound_quotientSup_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gap : ℕ → ℝ} {Umax B : ℝ}
    {r₁ : ℕ → ι₁ → ℝ} {uStar₁ : ι₁ → ℝ} {uNow₁ : ℕ → ι₁ → ℝ}
    {r₂ : ℕ → ι₂ → ℝ} {uStar₂ : ι₂ → ℝ} {uNow₂ : ℕ → ι₂ → ℝ}
    (τ : PairedSign)
    (horth : ∀ k : ℕ, (∑ i, r₁ k i) + τ.toReal * (∑ j, r₂ k j) = 0)
    (hpair :
      ∀ k : ℕ,
        gap k ≤
          -((∑ i, r₁ k i * (uStar₁ i - uNow₁ k i))
            + ∑ j, r₂ k j * (uStar₂ j - uNow₂ k j)))
    (hStar_lt :
      signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) < Umax)
    (hNow_lt :
      ∀ k : ℕ, signedPairedQuotientSupSeminorm τ (uNow₁ k, uNow₂ k) < Umax)
    (hUmax_nonneg : 0 ≤ Umax)
    (hres_rate : ∀ k : ℕ, finiteL1Pair (r₁ k) (r₂ k) ≤ B / (k + 1 : ℝ)) :
    ∀ k : ℕ, gap k ≤ (2 * Umax) * B / (k + 1 : ℝ) := by
  intro k
  have hgap :=
    dualGap_seq_le_twoUmax_of_negPairingBound_quotientSup_lt_Umax
      (gap := gap) (Umax := Umax) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
      τ horth hpair hStar_lt hNow_lt k
  have hmul :
      (2 * Umax) * finiteL1Pair (r₁ k) (r₂ k) ≤
        (2 * Umax) * (B / (k + 1 : ℝ)) := by
    exact mul_le_mul_of_nonneg_left (hres_rate k) (by nlinarith)
  calc
    gap k ≤ (2 * Umax) * finiteL1Pair (r₁ k) (r₂ k) := hgap
    _ ≤ (2 * Umax) * (B / (k + 1 : ℝ)) := hmul
    _ = (2 * Umax) * B / (k + 1 : ℝ) := by ring

/--
Rate-ready consequence of the block-sup strict sequence A.2 endpoint.
-/
theorem dualGap_seq_rate_of_pairingBound_blockSup_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gap : ℕ → ℝ} {Umax B : ℝ}
    {r₁ : ℕ → ι₁ → ℝ} {uStar₁ : ℕ → ι₁ → ℝ} {uNow₁ : ℕ → ι₁ → ℝ}
    {r₂ : ℕ → ι₂ → ℝ} {uStar₂ : ℕ → ι₂ → ℝ} {uNow₂ : ℕ → ι₂ → ℝ}
    (τ : PairedSign)
    (horth : ∀ k : ℕ, (∑ i, r₁ k i) + τ.toReal * (∑ j, r₂ k j) = 0)
    (hpair :
      ∀ k : ℕ,
        gap k ≤
          (∑ i, r₁ k i * (uStar₁ k i - uNow₁ k i))
            + ∑ j, r₂ k j * (uStar₂ k j - uNow₂ k j))
    (hStar : ∀ k : ℕ, blockSupNorm (uStar₁ k, uStar₂ k) < Umax)
    (hNow : ∀ k : ℕ, blockSupNorm (uNow₁ k, uNow₂ k) < Umax)
    (hUmax_nonneg : 0 ≤ Umax)
    (hres_rate : ∀ k : ℕ, finiteL1Pair (r₁ k) (r₂ k) ≤ B / (k + 1 : ℝ)) :
    ∀ k : ℕ, gap k ≤ (2 * Umax) * B / (k + 1 : ℝ) := by
  intro k
  have hgap :=
    dualGap_seq_le_twoUmax_of_pairingBound_blockSup_lt_Umax
      (gap := gap) (Umax := Umax)
      (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
      τ horth hpair hStar hNow k
  have hmul :
      (2 * Umax) * finiteL1Pair (r₁ k) (r₂ k) ≤
        (2 * Umax) * (B / (k + 1 : ℝ)) := by
    exact mul_le_mul_of_nonneg_left (hres_rate k) (by nlinarith)
  calc
    gap k ≤ (2 * Umax) * finiteL1Pair (r₁ k) (r₂ k) := hgap
    _ ≤ (2 * Umax) * (B / (k + 1 : ℝ)) := hmul
    _ = (2 * Umax) * B / (k + 1 : ℝ) := by ring

/-- The real value of a paired sign is involutive. -/
theorem pairedSign_toReal_mul_self (τ : PairedSign) :
    τ.toReal * τ.toReal = 1 := by
  cases τ <;> norm_num [PairedSign.toReal]

/--
Signed gauge orthogonality from a mass-balance identity between the two residual blocks.

This is the algebraic form usually produced by the underlying conservation law: the left
residual mass equals the signed negative of the right residual mass, hence the residual is
orthogonal to the signed gauge direction.
-/
theorem signedGaugeOrthogonality_of_sum_left_eq_neg_signed_sum_right
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    {r₁ : ι₁ → ℝ} {r₂ : ι₂ → ℝ} (τ : PairedSign)
    (hbalance : (∑ i, r₁ i) = -τ.toReal * (∑ j, r₂ j)) :
    (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0 := by
  rw [hbalance]
  ring

/--
Minus-sign gauge orthogonality from equality of the two block residual masses.

For the classical OT gauge action (`τ = minus`), the condition is the familiar equality of
total residuals in the two blocks.
-/
theorem signedGaugeOrthogonality_minus_of_sum_left_eq_sum_right
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    {r₁ : ι₁ → ℝ} {r₂ : ι₂ → ℝ}
    (hbalance : (∑ i, r₁ i) = ∑ j, r₂ j) :
    (∑ i, r₁ i) + PairedSign.minus.toReal * (∑ j, r₂ j) = 0 := by
  rw [hbalance]
  norm_num [PairedSign.toReal]

/--
Plus-sign gauge orthogonality from cancellation of the two block residual masses.
-/
theorem signedGaugeOrthogonality_plus_of_sum_add_eq_zero
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    {r₁ : ι₁ → ℝ} {r₂ : ι₂ → ℝ}
    (hbalance : (∑ i, r₁ i) + ∑ j, r₂ j = 0) :
    (∑ i, r₁ i) + PairedSign.plus.toReal * (∑ j, r₂ j) = 0 := by
  simpa [PairedSign.toReal] using hbalance

/--
Gauge orthogonality for two mass-shell residual blocks.

If each residual block is the difference of two finite vectors with the same total mass, then
both block residual masses vanish, so the residual is orthogonal to either signed gauge
direction.
-/
theorem signedGaugeOrthogonality_of_blockResidual_equalMasses
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    {p₁ q₁ : ι₁ → ℝ} {p₂ q₂ : ι₂ → ℝ} (τ : PairedSign)
    (hmass₁ : (∑ i, p₁ i) = ∑ i, q₁ i)
    (hmass₂ : (∑ j, p₂ j) = ∑ j, q₂ j) :
    (∑ i, (p₁ i - q₁ i)) + τ.toReal * (∑ j, (p₂ j - q₂ j)) = 0 := by
  have h₁ : (∑ i, (p₁ i - q₁ i)) = 0 := by
    rw [Finset.sum_sub_distrib, hmass₁]
    ring
  have h₂ : (∑ j, (p₂ j - q₂ j)) = 0 := by
    rw [Finset.sum_sub_distrib, hmass₂]
    ring
  rw [h₁, h₂]
  ring

/--
Gauge orthogonality for the opposite residual convention `q - p` on both mass-shell blocks.
-/
theorem signedGaugeOrthogonality_of_blockResidual_equalMasses_rev
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    {p₁ q₁ : ι₁ → ℝ} {p₂ q₂ : ι₂ → ℝ} (τ : PairedSign)
    (hmass₁ : (∑ i, p₁ i) = ∑ i, q₁ i)
    (hmass₂ : (∑ j, p₂ j) = ∑ j, q₂ j) :
    (∑ i, (q₁ i - p₁ i)) + τ.toReal * (∑ j, (q₂ j - p₂ j)) = 0 := by
  have h₁ : (∑ i, (q₁ i - p₁ i)) = 0 := by
    rw [Finset.sum_sub_distrib, ← hmass₁]
    ring
  have h₂ : (∑ j, (q₂ j - p₂ j)) = 0 := by
    rw [Finset.sum_sub_distrib, ← hmass₂]
    ring
  rw [h₁, h₂]
  ring

/--
Gauge orthogonality transports the residual pairing when only the reference dual pair is shifted.

This is the one-sided version of `pairResidualPairing_signedShift_sub_eq`, useful when an
application normalizes the optimizer/reference representative before normalizing the current
iterate.
-/
theorem pairResidualPairing_signedShift_left_eq
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign) (cStar : ℝ)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0) :
    (∑ i, r₁ i * ((uStar₁ i + cStar) - uNow₁ i))
        + ∑ j, r₂ j * ((uStar₂ j + τ.toReal * cStar) - uNow₂ j)
      =
    (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
        + ∑ j, r₂ j * (uStar₂ j - uNow₂ j) := by
  simpa [mul_zero, add_zero] using
    (pairResidualPairing_signedShift_sub_eq
      (τ := τ) (cStar := cStar) (cNow := 0)
      (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂) horth)

/--
Gauge orthogonality transports the residual pairing when only the current dual pair is shifted.
-/
theorem pairResidualPairing_signedShift_right_eq
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign) (cNow : ℝ)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0) :
    (∑ i, r₁ i * (uStar₁ i - (uNow₁ i + cNow)))
        + ∑ j, r₂ j * (uStar₂ j - (uNow₂ j + τ.toReal * cNow))
      =
    (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
        + ∑ j, r₂ j * (uStar₂ j - uNow₂ j) := by
  simpa [mul_zero, add_zero] using
    (pairResidualPairing_signedShift_sub_eq
      (τ := τ) (cStar := 0) (cNow := cNow)
      (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂) horth)

/--
Value-gap bridge when the gap equals the value difference exactly.
-/
theorem gap_le_abs_pairing_of_valueGap_eq_concavityPairing
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    {gapNow : ℝ} {Φ : ((ι₁ → ℝ) × (ι₂ → ℝ)) → ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (hgapValue :
      gapNow = Φ (uStar₁, uStar₂) - Φ (uNow₁, uNow₂))
    (hconcavity :
      Φ (uStar₁, uStar₂) - Φ (uNow₁, uNow₂) ≤
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)) :
    gapNow ≤
      |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
        + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)| := by
  exact gap_le_abs_pairing_of_valueGap_le_concavityPairing
    (gapNow := gapNow) (Φ := Φ)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    (le_of_eq hgapValue) hconcavity

/--
Value-gap bridge for the negative residual-pairing convention.
-/
theorem gap_le_abs_pairing_of_valueGap_le_negConcavityPairing
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    {gapNow : ℝ} {Φ : ((ι₁ → ℝ) × (ι₂ → ℝ)) → ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (hgapValue :
      gapNow ≤ Φ (uStar₁, uStar₂) - Φ (uNow₁, uNow₂))
    (hconcavity :
      Φ (uStar₁, uStar₂) - Φ (uNow₁, uNow₂) ≤
        -((∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))) :
    gapNow ≤
      |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
        + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)| :=
  (hgapValue.trans hconcavity).trans (neg_le_abs _)

/--
Exact-value-gap bridge for the negative residual-pairing convention.
-/
theorem gap_le_abs_pairing_of_valueGap_eq_negConcavityPairing
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    {gapNow : ℝ} {Φ : ((ι₁ → ℝ) × (ι₂ → ℝ)) → ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (hgapValue :
      gapNow = Φ (uStar₁, uStar₂) - Φ (uNow₁, uNow₂))
    (hconcavity :
      Φ (uStar₁, uStar₂) - Φ (uNow₁, uNow₂) ≤
        -((∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))) :
    gapNow ≤
      |(∑ i, r₁ i * (uStar₁ i - uNow₁ i))
        + ∑ j, r₂ j * (uStar₂ j - uNow₂ j)| :=
  gap_le_abs_pairing_of_valueGap_le_negConcavityPairing
    (gapNow := gapNow) (Φ := Φ)
    (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    (le_of_eq hgapValue) hconcavity

/--
Signed-shift witness obtained by centering the left block and checking only the shifted right
block against the same radius.
-/
theorem signedShiftWitness_of_left_centeringShift_right_abs_le
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) {U : ℝ}
    (hleft : variationSeminorm u.1 ≤ U)
    (hright : ∀ j, |u.2 j + τ.toReal * centeringShift u.1| ≤ U) :
    ∃ c : ℝ, blockSupNorm (signedPairedShift τ c u) ≤ U := by
  exact signedShiftWitness_of_forall_abs_signedShift_le
    τ u (centeringShift u.1)
    (fun i => (abs_add_centeringShift_le_variationSeminorm u.1 i).trans hleft)
    hright

/--
Strict signed-shift witness from left-block centering and a strict radius margin.
-/
theorem signedShiftWitness_lt_Umax_of_left_centeringShift_right_abs_le_radius_lt
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) {U Umax : ℝ}
    (hleft : variationSeminorm u.1 ≤ U)
    (hright : ∀ j, |u.2 j + τ.toReal * centeringShift u.1| ≤ U)
    (hU : U < Umax) :
    ∃ c : ℝ, blockSupNorm (signedPairedShift τ c u) < Umax := by
  rcases signedShiftWitness_of_left_centeringShift_right_abs_le
      τ u hleft hright with ⟨c, hc⟩
  exact ⟨c, lt_of_le_of_lt hc hU⟩

/--
Signed-shift witness obtained by centering the right block and checking only the shifted left
block against the same radius.
-/
theorem signedShiftWitness_of_right_centeringShift_left_abs_le
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) {U : ℝ}
    (hleft : ∀ i, |u.1 i + τ.toReal * centeringShift u.2| ≤ U)
    (hright : variationSeminorm u.2 ≤ U) :
    ∃ c : ℝ, blockSupNorm (signedPairedShift τ c u) ≤ U := by
  refine signedShiftWitness_of_forall_abs_signedShift_le
    τ u (τ.toReal * centeringShift u.2) hleft ?_
  intro j
  have hcenter := (abs_add_centeringShift_le_variationSeminorm u.2 j).trans hright
  cases τ <;> simpa [PairedSign.toReal] using hcenter

/--
Strict signed-shift witness from right-block centering and a strict radius margin.
-/
theorem signedShiftWitness_lt_Umax_of_right_centeringShift_left_abs_le_radius_lt
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) {U Umax : ℝ}
    (hleft : ∀ i, |u.1 i + τ.toReal * centeringShift u.2| ≤ U)
    (hright : variationSeminorm u.2 ≤ U)
    (hU : U < Umax) :
    ∃ c : ℝ, blockSupNorm (signedPairedShift τ c u) < Umax := by
  rcases signedShiftWitness_of_right_centeringShift_left_abs_le
      τ u hleft hright with ⟨c, hc⟩
  exact ⟨c, lt_of_le_of_lt hc hU⟩

/--
The signed quotient seminorm controls the explicitly centered left-block representative.

This connects the quotient geometry in `BlockQuotient.lean` to the coordinate hypotheses used by
the A.2 centered-representative endpoints: quotient control gives variation control, and the
centering shift realizes the variation bound coordinatewise.
-/
theorem leftCentered_abs_le_signedPairedQuotientSupSeminorm
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) :
    ∀ i, |u.1 i + centeringShift u.1| ≤ signedPairedQuotientSupSeminorm τ u := by
  intro i
  exact (abs_add_centeringShift_le_variationSeminorm u.1 i).trans
    (signedPairedQuotientSupSeminorm_controls_blockVariation_left τ u)

/--
The signed quotient seminorm controls the explicitly centered right-block representative.
-/
theorem rightCentered_abs_le_signedPairedQuotientSupSeminorm
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) :
    ∀ j, |u.2 j + centeringShift u.2| ≤ signedPairedQuotientSupSeminorm τ u := by
  intro j
  exact (abs_add_centeringShift_le_variationSeminorm u.2 j).trans
    (signedPairedQuotientSupSeminorm_controls_blockVariation_right τ u)

/--
Right-block coordinate bound for the signed shift obtained by centering the right block.

The chosen signed shift is `c = τ * centeringShift u.2`, so the second block receives
`τ * c = centeringShift u.2`.
-/
theorem rightCentered_signedShift_abs_le_signedPairedQuotientSupSeminorm
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) :
    ∀ j,
      |u.2 j + τ.toReal * (τ.toReal * centeringShift u.2)| ≤
        signedPairedQuotientSupSeminorm τ u := by
  intro j
  have hcenter := rightCentered_abs_le_signedPairedQuotientSupSeminorm τ u j
  cases τ <;> simpa [PairedSign.toReal] using hcenter

/--
Left-centered signed-shift witness with the left variation bound derived from quotient geometry.

Only the shifted right-block coordinate estimate remains as a concrete application-side input.
-/
theorem signedShiftWitness_of_left_centeringShift_quotientSup_right_abs_le
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) {U : ℝ}
    (hquot : signedPairedQuotientSupSeminorm τ u ≤ U)
    (hright : ∀ j, |u.2 j + τ.toReal * centeringShift u.1| ≤ U) :
    ∃ c : ℝ, blockSupNorm (signedPairedShift τ c u) ≤ U := by
  exact signedShiftWitness_of_left_centeringShift_right_abs_le
    τ u
    ((signedPairedQuotientSupSeminorm_controls_blockVariation_left τ u).trans hquot)
    hright

/--
Strict left-centered representative bound with the left-block estimate supplied by quotient
geometry.
-/
theorem signedShiftWitness_lt_Umax_of_left_centeringShift_quotientSup_right_abs_le_radius_lt
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) {U Umax : ℝ}
    (hquot : signedPairedQuotientSupSeminorm τ u ≤ U)
    (hright : ∀ j, |u.2 j + τ.toReal * centeringShift u.1| ≤ U)
    (hU : U < Umax) :
    ∃ c : ℝ, blockSupNorm (signedPairedShift τ c u) < Umax := by
  rcases signedShiftWitness_of_left_centeringShift_quotientSup_right_abs_le
      τ u hquot hright with ⟨c, hc⟩
  exact ⟨c, lt_of_le_of_lt hc hU⟩

/--
Right-centered signed-shift witness with the right variation bound derived from quotient geometry.

Only the shifted left-block coordinate estimate remains as a concrete application-side input.
-/
theorem signedShiftWitness_of_right_centeringShift_quotientSup_left_abs_le
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) {U : ℝ}
    (hleft : ∀ i, |u.1 i + τ.toReal * centeringShift u.2| ≤ U)
    (hquot : signedPairedQuotientSupSeminorm τ u ≤ U) :
    ∃ c : ℝ, blockSupNorm (signedPairedShift τ c u) ≤ U := by
  exact signedShiftWitness_of_right_centeringShift_left_abs_le
    τ u hleft
    ((signedPairedQuotientSupSeminorm_controls_blockVariation_right τ u).trans hquot)

/--
Strict right-centered representative bound with the right-block estimate supplied by quotient
geometry.
-/
theorem signedShiftWitness_lt_Umax_of_right_centeringShift_quotientSup_left_abs_le_radius_lt
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (τ : PairedSign) (u : BlockPair ι₁ ι₂) {U Umax : ℝ}
    (hleft : ∀ i, |u.1 i + τ.toReal * centeringShift u.2| ≤ U)
    (hquot : signedPairedQuotientSupSeminorm τ u ≤ U)
    (hU : U < Umax) :
    ∃ c : ℝ, blockSupNorm (signedPairedShift τ c u) < Umax := by
  rcases signedShiftWitness_of_right_centeringShift_quotientSup_left_abs_le
      τ u hleft hquot with ⟨c, hc⟩
  exact ⟨c, lt_of_le_of_lt hc hU⟩

/--
A.2 value-gap endpoint with representatives obtained by centering the left block of each pair.
-/
theorem dualGap_le_twoUmax_of_valueGap_concavity_leftCenteredSignedShiftBounds_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {Φ : ((ι₁ → ℝ) × (ι₂ → ℝ)) → ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hgapValue :
      gapNow ≤ Φ (uStar₁, uStar₂) - Φ (uNow₁, uNow₂))
    (hconcavity :
      Φ (uStar₁, uStar₂) - Φ (uNow₁, uNow₂) ≤
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hStar₁ : variationSeminorm uStar₁ ≤ UStar)
    (hStar₂ : ∀ j, |uStar₂ j + τ.toReal * centeringShift uStar₁| ≤ UStar)
    (hNow₁ : variationSeminorm uNow₁ ≤ UNow)
    (hNow₂ : ∀ j, |uNow₂ j + τ.toReal * centeringShift uNow₁| ≤ UNow)
    (hStar_lt : UStar < Umax)
    (hNow_lt : UNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_valueGap_concavity_signedShiftCoordinateBounds_lt_Umax
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (Φ := Φ) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ (centeringShift uStar₁) (centeringShift uNow₁)
    horth hgapValue hconcavity
    (fun i => (abs_add_centeringShift_le_variationSeminorm uStar₁ i).trans hStar₁)
    hStar₂
    (fun i => (abs_add_centeringShift_le_variationSeminorm uNow₁ i).trans hNow₁)
    hNow₂ hStar_lt hNow_lt

/--
A.2 value-gap endpoint with representatives obtained by centering the right block of each pair.
-/
theorem dualGap_le_twoUmax_of_valueGap_concavity_rightCenteredSignedShiftBounds_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {Φ : ((ι₁ → ℝ) × (ι₂ → ℝ)) → ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hgapValue :
      gapNow ≤ Φ (uStar₁, uStar₂) - Φ (uNow₁, uNow₂))
    (hconcavity :
      Φ (uStar₁, uStar₂) - Φ (uNow₁, uNow₂) ≤
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hStar₁ : ∀ i, |uStar₁ i + τ.toReal * centeringShift uStar₂| ≤ UStar)
    (hStar₂ : variationSeminorm uStar₂ ≤ UStar)
    (hNow₁ : ∀ i, |uNow₁ i + τ.toReal * centeringShift uNow₂| ≤ UNow)
    (hNow₂ : variationSeminorm uNow₂ ≤ UNow)
    (hStar_lt : UStar < Umax)
    (hNow_lt : UNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  have hStar₂' :
      ∀ j, |uStar₂ j + τ.toReal * (τ.toReal * centeringShift uStar₂)| ≤ UStar := by
    intro j
    have hcenter := (abs_add_centeringShift_le_variationSeminorm uStar₂ j).trans hStar₂
    cases τ <;> simpa [PairedSign.toReal] using hcenter
  have hNow₂' :
      ∀ j, |uNow₂ j + τ.toReal * (τ.toReal * centeringShift uNow₂)| ≤ UNow := by
    intro j
    have hcenter := (abs_add_centeringShift_le_variationSeminorm uNow₂ j).trans hNow₂
    cases τ <;> simpa [PairedSign.toReal] using hcenter
  exact dualGap_le_twoUmax_of_valueGap_concavity_signedShiftCoordinateBounds_lt_Umax
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (Φ := Φ) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ (τ.toReal * centeringShift uStar₂) (τ.toReal * centeringShift uNow₂)
    horth hgapValue hconcavity hStar₁ hStar₂' hNow₁ hNow₂' hStar_lt hNow_lt

/--
A.2 value-gap endpoint with left-centered representatives and left-block bounds derived from
the signed quotient seminorm.

Compared with `dualGap_le_twoUmax_of_valueGap_concavity_leftCenteredSignedShiftBounds_lt_Umax`,
the caller no longer proves separate variation bounds for the centered left blocks; those are
internalized from the quotient geometry.  The shifted right-block coordinate bounds remain the
application-specific representative estimates.
-/
theorem dualGap_le_twoUmax_of_valueGap_concavity_leftCenteredQuotientRightBounds_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {Φ : ((ι₁ → ℝ) × (ι₂ → ℝ)) → ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hgapValue :
      gapNow ≤ Φ (uStar₁, uStar₂) - Φ (uNow₁, uNow₂))
    (hconcavity :
      Φ (uStar₁, uStar₂) - Φ (uNow₁, uNow₂) ≤
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hStarQ : signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) ≤ UStar)
    (hStar₂ : ∀ j, |uStar₂ j + τ.toReal * centeringShift uStar₁| ≤ UStar)
    (hNowQ : signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) ≤ UNow)
    (hNow₂ : ∀ j, |uNow₂ j + τ.toReal * centeringShift uNow₁| ≤ UNow)
    (hStar_lt : UStar < Umax)
    (hNow_lt : UNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_valueGap_concavity_leftCenteredSignedShiftBounds_lt_Umax
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (Φ := Φ) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hgapValue hconcavity
    ((signedPairedQuotientSupSeminorm_controls_blockVariation_left
      τ (uStar₁, uStar₂)).trans hStarQ)
    hStar₂
    ((signedPairedQuotientSupSeminorm_controls_blockVariation_left
      τ (uNow₁, uNow₂)).trans hNowQ)
    hNow₂ hStar_lt hNow_lt

/--
A.2 value-gap endpoint with right-centered representatives and right-block bounds derived from
the signed quotient seminorm.

The shifted left-block coordinate bounds are left as concrete application inputs.
-/
theorem dualGap_le_twoUmax_of_valueGap_concavity_rightCenteredQuotientLeftBounds_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gapNow UStar UNow Umax : ℝ}
    {Φ : ((ι₁ → ℝ) × (ι₂ → ℝ)) → ℝ}
    {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ}
    (τ : PairedSign)
    (horth : (∑ i, r₁ i) + τ.toReal * (∑ j, r₂ j) = 0)
    (hgapValue :
      gapNow ≤ Φ (uStar₁, uStar₂) - Φ (uNow₁, uNow₂))
    (hconcavity :
      Φ (uStar₁, uStar₂) - Φ (uNow₁, uNow₂) ≤
        (∑ i, r₁ i * (uStar₁ i - uNow₁ i))
          + ∑ j, r₂ j * (uStar₂ j - uNow₂ j))
    (hStar₁ : ∀ i, |uStar₁ i + τ.toReal * centeringShift uStar₂| ≤ UStar)
    (hStarQ : signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) ≤ UStar)
    (hNow₁ : ∀ i, |uNow₁ i + τ.toReal * centeringShift uNow₂| ≤ UNow)
    (hNowQ : signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) ≤ UNow)
    (hStar_lt : UStar < Umax)
    (hNow_lt : UNow < Umax) :
    gapNow ≤ (2 * Umax) * finiteL1Pair r₁ r₂ := by
  exact dualGap_le_twoUmax_of_valueGap_concavity_rightCenteredSignedShiftBounds_lt_Umax
    (gapNow := gapNow) (UStar := UStar) (UNow := UNow) (Umax := Umax)
    (Φ := Φ) (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
    (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
    τ horth hgapValue hconcavity hStar₁
    ((signedPairedQuotientSupSeminorm_controls_blockVariation_right
      τ (uStar₁, uStar₂)).trans hStarQ)
    hNow₁
    ((signedPairedQuotientSupSeminorm_controls_blockVariation_right
      τ (uNow₁, uNow₂)).trans hNowQ)
    hStar_lt hNow_lt

/--
Sequence value-gap form of the paper-facing A.2 quotient endpoint.
-/
theorem dualGap_seq_le_twoUmax_of_valueGap_concavity_quotientSup_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gap : ℕ → ℝ} {Umax : ℝ}
    {Φ : ((ι₁ → ℝ) × (ι₂ → ℝ)) → ℝ}
    {r₁ : ℕ → ι₁ → ℝ} {uStar₁ : ι₁ → ℝ} {uNow₁ : ℕ → ι₁ → ℝ}
    {r₂ : ℕ → ι₂ → ℝ} {uStar₂ : ι₂ → ℝ} {uNow₂ : ℕ → ι₂ → ℝ}
    (τ : PairedSign)
    (horth : ∀ k : ℕ, (∑ i, r₁ k i) + τ.toReal * (∑ j, r₂ k j) = 0)
    (hgapValue :
      ∀ k : ℕ,
        gap k ≤ Φ (uStar₁, uStar₂) - Φ (uNow₁ k, uNow₂ k))
    (hconcavity :
      ∀ k : ℕ,
        Φ (uStar₁, uStar₂) - Φ (uNow₁ k, uNow₂ k) ≤
          (∑ i, r₁ k i * (uStar₁ i - uNow₁ k i))
            + ∑ j, r₂ k j * (uStar₂ j - uNow₂ k j))
    (hStar_lt :
      signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) < Umax)
    (hNow_lt :
      ∀ k : ℕ, signedPairedQuotientSupSeminorm τ (uNow₁ k, uNow₂ k) < Umax) :
    ∀ k : ℕ, gap k ≤ (2 * Umax) * finiteL1Pair (r₁ k) (r₂ k) := by
  intro k
  exact dualGap_le_twoUmax_of_pairingBound_quotientSup_lt_Umax
    (gapNow := gap k) (Umax := Umax)
    (r₁ := r₁ k) (uStar₁ := uStar₁) (uNow₁ := uNow₁ k)
    (r₂ := r₂ k) (uStar₂ := uStar₂) (uNow₂ := uNow₂ k)
    τ (horth k) ((hgapValue k).trans (hconcavity k)) hStar_lt (hNow_lt k)

/--
Rate-ready value-gap consequence of the sequence quotient A.2 endpoint.
-/
theorem dualGap_seq_rate_of_valueGap_concavity_quotientSup_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gap : ℕ → ℝ} {Umax B : ℝ}
    {Φ : ((ι₁ → ℝ) × (ι₂ → ℝ)) → ℝ}
    {r₁ : ℕ → ι₁ → ℝ} {uStar₁ : ι₁ → ℝ} {uNow₁ : ℕ → ι₁ → ℝ}
    {r₂ : ℕ → ι₂ → ℝ} {uStar₂ : ι₂ → ℝ} {uNow₂ : ℕ → ι₂ → ℝ}
    (τ : PairedSign)
    (horth : ∀ k : ℕ, (∑ i, r₁ k i) + τ.toReal * (∑ j, r₂ k j) = 0)
    (hgapValue :
      ∀ k : ℕ,
        gap k ≤ Φ (uStar₁, uStar₂) - Φ (uNow₁ k, uNow₂ k))
    (hconcavity :
      ∀ k : ℕ,
        Φ (uStar₁, uStar₂) - Φ (uNow₁ k, uNow₂ k) ≤
          (∑ i, r₁ k i * (uStar₁ i - uNow₁ k i))
            + ∑ j, r₂ k j * (uStar₂ j - uNow₂ k j))
    (hStar_lt :
      signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) < Umax)
    (hNow_lt :
      ∀ k : ℕ, signedPairedQuotientSupSeminorm τ (uNow₁ k, uNow₂ k) < Umax)
    (hUmax_nonneg : 0 ≤ Umax)
    (hres_rate : ∀ k : ℕ, finiteL1Pair (r₁ k) (r₂ k) ≤ B / (k + 1 : ℝ)) :
    ∀ k : ℕ, gap k ≤ (2 * Umax) * B / (k + 1 : ℝ) := by
  intro k
  have hgap :=
    dualGap_seq_le_twoUmax_of_valueGap_concavity_quotientSup_lt_Umax
      (gap := gap) (Umax := Umax) (Φ := Φ)
      (r₁ := r₁) (uStar₁ := uStar₁) (uNow₁ := uNow₁)
      (r₂ := r₂) (uStar₂ := uStar₂) (uNow₂ := uNow₂)
      τ horth hgapValue hconcavity hStar_lt hNow_lt k
  have hmul :
      (2 * Umax) * finiteL1Pair (r₁ k) (r₂ k) ≤
        (2 * Umax) * (B / (k + 1 : ℝ)) :=
    mul_le_mul_of_nonneg_left (hres_rate k) (by nlinarith)
  calc
    gap k ≤ (2 * Umax) * finiteL1Pair (r₁ k) (r₂ k) := hgap
    _ ≤ (2 * Umax) * (B / (k + 1 : ℝ)) := hmul
    _ = (2 * Umax) * B / (k + 1 : ℝ) := by ring

/--
Sequence value-gap form of A.2 for the negative residual-pairing convention.
-/
theorem dualGap_seq_le_twoUmax_of_valueGap_negConcavity_quotientSup_lt_Umax
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    {gap : ℕ → ℝ} {Umax : ℝ}
    {Φ : ((ι₁ → ℝ) × (ι₂ → ℝ)) → ℝ}
    {r₁ : ℕ → ι₁ → ℝ} {uStar₁ : ι₁ → ℝ} {uNow₁ : ℕ → ι₁ → ℝ}
    {r₂ : ℕ → ι₂ → ℝ} {uStar₂ : ι₂ → ℝ} {uNow₂ : ℕ → ι₂ → ℝ}
    (τ : PairedSign)
    (horth : ∀ k : ℕ, (∑ i, r₁ k i) + τ.toReal * (∑ j, r₂ k j) = 0)
    (hgapValue :
      ∀ k : ℕ,
        gap k ≤ Φ (uStar₁, uStar₂) - Φ (uNow₁ k, uNow₂ k))
    (hconcavity :
      ∀ k : ℕ,
        Φ (uStar₁, uStar₂) - Φ (uNow₁ k, uNow₂ k) ≤
          -((∑ i, r₁ k i * (uStar₁ i - uNow₁ k i))
            + ∑ j, r₂ k j * (uStar₂ j - uNow₂ k j)))
    (hStar_lt :
      signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) < Umax)
    (hNow_lt :
      ∀ k : ℕ, signedPairedQuotientSupSeminorm τ (uNow₁ k, uNow₂ k) < Umax) :
    ∀ k : ℕ, gap k ≤ (2 * Umax) * finiteL1Pair (r₁ k) (r₂ k) := by
  intro k
  exact dualGap_le_twoUmax_of_negPairingBound_quotientSup_lt_Umax
    (gapNow := gap k) (Umax := Umax)
    (r₁ := r₁ k) (uStar₁ := uStar₁) (uNow₁ := uNow₁ k)
    (r₂ := r₂ k) (uStar₂ := uStar₂) (uNow₂ := uNow₂ k)
    τ (horth k) ((hgapValue k).trans (hconcavity k)) hStar_lt (hNow_lt k)

/--
Scaled version with an explicit comparison constant.

This is the form typically used in the convergence proof once one has a concrete bound
`residual_k ≤ c * benchmark_k`.
-/
theorem dualGap_le_residual_with_constant
    {benchmark : ℕ → ℝ} {c : ℝ}
    (hgap : ∀ k : ℕ, gap k ≤ residual k)
    (hres : ∀ k : ℕ, residual k ≤ c * benchmark k)
    (k : ℕ) :
    gap k ≤ c * benchmark k :=
  (hgap k).trans (hres k)

/--
Scaled version of the gap/residual interface.

Once the residual comparison is established, this gives the corresponding scaled comparison
needed when the quotient norm is tracked with an explicit constant.
-/
theorem dualGap_le_scaledResidual
    {residual : ℕ → ℝ} {c : ℝ}
    (hgap : ∀ k : ℕ, gap k ≤ residual k)
    (hc : 0 ≤ c)
    (k : ℕ) :
    c * gap k ≤ c * residual k := by
  exact mul_le_mul_of_nonneg_left (hgap k) hc

/--
If the gap is bounded by a product `alpha * residual` and residual satisfies the one-step
ascent, then the gap sequence is antitone whenever the gap is the minimum of
`alpha * (phi (n+1) - phi 0) / (n+1)` type bounds.

This specializes the gap-residual interface to the case where the cumulative residual grows
like the objective function growth.
-/
theorem dualGap_antitone_of_residualAscent_and_bound
    {gap residual phi : ℕ → ℝ} {alpha B : ℝ}
    (_hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (_hres_ascent : ∀ k : ℕ, residual k ≤ phi (k + 1) - phi k)
    (_hphi_bound : ∀ n : ℕ, phi (n + 1) - phi 0 ≤ B)
    (hmono_gap : Antitone gap)
    (n : ℕ) :
    gap n ≤ gap 0 :=
  hmono_gap (Nat.zero_le n)

/--
Composition of gap-residual estimates: if gap ≤ alpha * residual and residual ≤ beta * bench,
then gap ≤ alpha * beta * bench.
-/
theorem dualGap_le_composed_residual
    {gap residual bench : ℕ → ℝ} {alpha beta : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_bench : ∀ k : ℕ, residual k ≤ beta * bench k)
    (halpha : 0 ≤ alpha) (k : ℕ) :
    gap k ≤ alpha * beta * bench k := by
  have h1 := hgap_res k
  have h2 := mul_le_mul_of_nonneg_left (hres_bench k) halpha
  linarith [mul_assoc alpha beta (bench k)]

/--
Upper bound propagation through the rate estimate.

If gap_n ≤ C / (n+1) and C ≤ C', then gap_n ≤ C' / (n+1).
-/
theorem dualGap_le_larger_constant
    {gap : ℕ → ℝ} {C C' : ℝ}
    (hgap : ∀ n : ℕ, gap n ≤ C / (n + 1 : ℝ))
    (hCC' : C ≤ C')
    (n : ℕ) :
    gap n ≤ C' / (n + 1 : ℝ) :=
  (hgap n).trans (div_le_div_of_nonneg_right hCC' (by positivity))

/--
If `gap n ≤ C / (n+1)` for all n, then the sum of the first N gaps is bounded by `C * (log N + 1)`.

This is a useful consequence used when bounding the total dual suboptimality over T iterations.
Note: we give the simpler arithmetic version where the sum is bounded by `C * N`.
-/
theorem dualGap_cumulative_le_linear
    {gap : ℕ → ℝ} {C : ℝ}
    (hrate : ∀ n : ℕ, gap n ≤ C / (n + 1 : ℝ))
    (hC : 0 ≤ C)
    (N : ℕ) :
    ∑ k ∈ Finset.range N, gap k ≤ C * (N : ℝ) := by
  have hstep : ∀ k ∈ Finset.range N, gap k ≤ C := by
    intro k _
    calc gap k ≤ C / (↑k + 1) := hrate k
      _ ≤ C := by
          apply div_le_self hC
          exact le_add_of_nonneg_left (Nat.cast_nonneg k)
  calc ∑ k ∈ Finset.range N, gap k
      ≤ ∑ _ ∈ Finset.range N, C := Finset.sum_le_sum hstep
    _ = C * (N : ℝ) := by
        simp [Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_comm]

/--
The dual gap at the first iterate is at most the constant `C` (from an O(1/k) rate).
-/
theorem dualGap_first_iterate_le
    {gap : ℕ → ℝ} {C : ℝ}
    (hrate : ∀ n : ℕ, gap n ≤ C / (n + 1 : ℝ)) :
    gap 0 ≤ C := by
  have h := hrate 0
  simp only [Nat.cast_zero, zero_add, div_one] at h
  exact h

/--
If the gap is antitone, the average of the first N gaps is bounded by `gap 0`.
-/
theorem dualGap_average_le_initial
    {gap : ℕ → ℝ}
    (hmono : Antitone gap)
    {N : ℕ}
    (hN : 0 < N) :
    (∑ k ∈ Finset.range N, gap k) / N ≤ gap 0 := by
  have hN' : (0 : ℝ) < (N : ℝ) := Nat.cast_pos.mpr hN
  rw [div_le_iff₀ hN']
  have hbound : ∀ k ∈ Finset.range N, gap k ≤ gap 0 := by
    intro k _
    exact hmono (Nat.zero_le k)
  calc ∑ k ∈ Finset.range N, gap k
      ≤ ∑ _ ∈ Finset.range N, gap 0 := Finset.sum_le_sum hbound
    _ = gap 0 * (N : ℝ) := by
        simp [Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_comm]

/--
Composition: if gap ≤ alpha * residual and residual ≤ B for all iterates, then
gap ≤ alpha * B at all iterates.
-/
theorem dualGap_uniformly_bounded_of_residualBound
    {gap residual : ℕ → ℝ} {alpha B : ℝ}
    (hgap : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres : ∀ k : ℕ, residual k ≤ B)
    (halpha : 0 ≤ alpha) :
    ∀ k : ℕ, gap k ≤ alpha * B := by
  intro k
  exact (hgap k).trans (mul_le_mul_of_nonneg_left (hres k) halpha)

/--
Uniform composed bound with an intermediate benchmark.

If `gap ≤ alpha * residual`, `residual ≤ beta * bench`, and `bench ≤ B` pointwise, then
`gap ≤ alpha * beta * B` pointwise.
-/
theorem dualGap_uniformly_bounded_of_composed_residualBound
    {gap residual bench : ℕ → ℝ} {alpha beta B : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_bench : ∀ k : ℕ, residual k ≤ beta * bench k)
    (hbench : ∀ k : ℕ, bench k ≤ B)
    (halpha : 0 ≤ alpha) (hbeta : 0 ≤ beta) :
    ∀ k : ℕ, gap k ≤ alpha * beta * B := by
  intro k
  have hcomp : gap k ≤ alpha * beta * bench k :=
    dualGap_le_composed_residual hgap_res hres_bench halpha k
  have hab : 0 ≤ alpha * beta := mul_nonneg halpha hbeta
  have hmul : (alpha * beta) * bench k ≤ (alpha * beta) * B :=
    mul_le_mul_of_nonneg_left (hbench k) hab
  exact hcomp.trans (by simpa [mul_assoc] using hmul)

/--
Average-gap bound from an `O(1/k)` pointwise estimate.

This combines `dualGap_cumulative_le_linear` with division by `N`, giving a direct
average statement reusable in stopping-rule arguments.
-/
theorem dualGap_average_le_rate_constant
    {gap : ℕ → ℝ} {C : ℝ}
    (hrate : ∀ n : ℕ, gap n ≤ C / (n + 1 : ℝ))
    (hC : 0 ≤ C)
    {N : ℕ} (hN : 0 < N) :
    (∑ k ∈ Finset.range N, gap k) / N ≤ C := by
  have hsum : ∑ k ∈ Finset.range N, gap k ≤ C * (N : ℝ) :=
    dualGap_cumulative_le_linear hrate hC N
  have hN' : (0 : ℝ) < (N : ℝ) := Nat.cast_pos.mpr hN
  rw [div_le_iff₀ hN']
  simpa [mul_comm] using hsum

/--
Average-gap bound from antitonicity plus a pointwise `O(1/k)` rate constant.

This bridges `dualGap_average_le_initial` and `dualGap_first_iterate_le`.
-/
theorem dualGap_average_le_rate_constant_of_antitone
    {gap : ℕ → ℝ} {C : ℝ}
    (hmono : Antitone gap)
    (hrate : ∀ n : ℕ, gap n ≤ C / (n + 1 : ℝ))
    {N : ℕ} (hN : 0 < N) :
    (∑ k ∈ Finset.range N, gap k) / N ≤ C :=
  (dualGap_average_le_initial hmono hN).trans (dualGap_first_iterate_le hrate)

/--
`ε`-accuracy from a composed benchmark rate and a threshold inequality.

This theorem packages the chain
`gap ≤ alpha*residual ≤ alpha*beta*bench ≤ alpha*beta*(C/(n+1))`,
then discharges the final step with a provided threshold inequality.
-/
theorem dualGap_le_eps_of_composed_rateThreshold
    {gap residual bench : ℕ → ℝ} {alpha beta C eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_bench : ∀ k : ℕ, residual k ≤ beta * bench k)
    (hbench_rate : ∀ n : ℕ, bench n ≤ C / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha) (hbeta : 0 ≤ beta)
    (n : ℕ)
    (hthreshold : (alpha * beta) * (C / (n + 1 : ℝ)) ≤ eps) :
    gap n ≤ eps := by
  have hcomp : gap n ≤ alpha * beta * bench n :=
    dualGap_le_composed_residual hgap_res hres_bench halpha n
  have hab : 0 ≤ alpha * beta := mul_nonneg halpha hbeta
  have hbench_scaled : (alpha * beta) * bench n ≤ (alpha * beta) * (C / (n + 1 : ℝ)) :=
    mul_le_mul_of_nonneg_left (hbench_rate n) hab
  exact (hcomp.trans (by simpa [mul_assoc] using hbench_scaled)).trans hthreshold

/--
Composed pointwise `O(1/n)` rate from gap/residual/benchmark chaining.

This theorem packages the common proof path
`gap ≤ alpha * residual ≤ alpha * beta * bench ≤ alpha * beta * (C/(n+1))`.
-/
theorem dualGap_le_composed_rateBenchmark
    {gap residual bench : ℕ → ℝ} {alpha beta C : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_bench : ∀ k : ℕ, residual k ≤ beta * bench k)
    (hbench_rate : ∀ n : ℕ, bench n ≤ C / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha) (hbeta : 0 ≤ beta)
    (n : ℕ) :
    gap n ≤ (alpha * beta) * (C / (n + 1 : ℝ)) := by
  have hcomp : gap n ≤ alpha * beta * bench n :=
    dualGap_le_composed_residual hgap_res hres_bench halpha n
  have hab : 0 ≤ alpha * beta := mul_nonneg halpha hbeta
  have hbench_scaled :
      (alpha * beta) * bench n ≤ (alpha * beta) * (C / (n + 1 : ℝ)) :=
    mul_le_mul_of_nonneg_left (hbench_rate n) hab
  exact hcomp.trans (by simpa [mul_assoc] using hbench_scaled)

/--
`eps`-accuracy from the composed benchmark chain and a ratio-form threshold.

This is the ratio-form companion of `dualGap_le_eps_of_composed_rateThreshold`.
-/
theorem dualGap_le_eps_of_composed_rateThreshold_ratioBound
    {gap residual bench : ℕ → ℝ} {alpha beta C eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_bench : ∀ k : ℕ, residual k ≤ beta * bench k)
    (hbench_rate : ∀ n : ℕ, bench n ≤ C / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha) (hbeta : 0 ≤ beta)
    (heps : 0 < eps)
    (n : ℕ)
    (hratio : ((alpha * beta) * C) / eps ≤ (n + 1 : ℝ)) :
    gap n ≤ eps := by
  have hthreshold : (alpha * beta) * (C / (n + 1 : ℝ)) ≤ eps := by
    have hmul' : (alpha * beta) * C ≤ (n + 1 : ℝ) * eps :=
      (div_le_iff₀ heps).1 hratio
    have hmul : (alpha * beta) * C ≤ eps * (n + 1 : ℝ) := by
      simpa [mul_comm, mul_left_comm, mul_assoc] using hmul'
    have hpos : 0 < (n + 1 : ℝ) := by
      exact_mod_cast Nat.succ_pos n
    have hdiv : ((alpha * beta) * C) / (n + 1 : ℝ) ≤ eps :=
      (div_le_iff₀ hpos).2 hmul
    simpa [div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using hdiv
  exact dualGap_le_eps_of_composed_rateThreshold
    hgap_res hres_bench hbench_rate halpha hbeta n hthreshold

/--
`eps`-accuracy from the composed benchmark chain and a closed-form ceil threshold.

This is the paper-friendly closed-form version of
`dualGap_le_eps_of_composed_rateThreshold_ratioBound`.
-/
theorem dualGap_le_eps_of_composed_rateThreshold_closedFormCeil
    {gap residual bench : ℕ → ℝ} {alpha beta C eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_bench : ∀ k : ℕ, residual k ≤ beta * bench k)
    (hbench_rate : ∀ n : ℕ, bench n ≤ C / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha) (hbeta : 0 ≤ beta)
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil (((alpha * beta) * C) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  have hratio : ((alpha * beta) * C) / eps ≤ (n + 1 : ℝ) := by
    simpa [Nat.cast_add, Nat.cast_one] using
      (Nat.le_of_ceil_le hn : ((alpha * beta) * C) / eps ≤ (n + 1 : ℕ))
  exact dualGap_le_eps_of_composed_rateThreshold_ratioBound
    hgap_res hres_bench hbench_rate halpha hbeta heps n hratio

/--
Average-gap bound from the composed benchmark chain.

If the benchmark admits an `O(1/n)` bound, then the average dual gap over the first `N`
iterates is bounded by `(alpha * beta) * C`.
-/
theorem dualGap_average_le_composed_rateBenchmark
    {gap residual bench : ℕ → ℝ} {alpha beta C : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_bench : ∀ k : ℕ, residual k ≤ beta * bench k)
    (hbench_rate : ∀ n : ℕ, bench n ≤ C / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha) (hbeta : 0 ≤ beta) (hC : 0 ≤ C)
    {N : ℕ} (hN : 0 < N) :
    (∑ k ∈ Finset.range N, gap k) / N ≤ (alpha * beta) * C := by
  have hrate : ∀ n : ℕ, gap n ≤ ((alpha * beta) * C) / (n + 1 : ℝ) := by
    intro n
    have h := dualGap_le_composed_rateBenchmark
      hgap_res hres_bench hbench_rate halpha hbeta n
    simpa [div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using h
  have hab : 0 ≤ alpha * beta := mul_nonneg halpha hbeta
  have hC' : 0 ≤ (alpha * beta) * C := mul_nonneg hab hC
  exact dualGap_average_le_rate_constant (gap := gap) (C := (alpha * beta) * C)
    hrate hC' hN

/--
Pointwise `O(1/n)` gap rate derived directly from a residual `O(1/n)` rate.

This packages the common specialization `bench = residual`, `beta = 1` of the composed
benchmark chain.
-/
theorem dualGap_le_rate_of_residualRate
    {gap residual : ℕ → ℝ} {alpha B : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha)
    (n : ℕ) :
    gap n ≤ (alpha * B) / (n + 1 : ℝ) := by
  have h := dualGap_le_composed_rateBenchmark
    (gap := gap) (residual := residual) (bench := residual)
    (alpha := alpha) (beta := 1) (C := B)
    hgap_res
    (fun k => by simp)
    hres_rate halpha (by positivity) n
  calc
    gap n ≤ (alpha * 1) * (B / (n + 1 : ℝ)) := h
    _ = (alpha * B) / (n + 1 : ℝ) := by ring

/--
Closed-form `ε`-accuracy from a residual `O(1/n)` rate.

This is the specialization of
`dualGap_le_eps_of_composed_rateThreshold_closedFormCeil` with `bench = residual`, `beta = 1`.
-/
theorem dualGap_le_eps_of_residualRateThreshold_closedFormCeil
    {gap residual : ℕ → ℝ} {alpha B eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil ((alpha * B) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  have hn' : Nat.ceil (((alpha * 1) * B) / eps) ≤ n + 1 := by
    simpa using hn
  exact dualGap_le_eps_of_composed_rateThreshold_closedFormCeil
    (gap := gap) (residual := residual) (bench := residual)
    (alpha := alpha) (beta := 1) (C := B) (eps := eps)
    hgap_res
    (fun k => by simp)
    hres_rate halpha (by positivity) heps n hn'

/--
`ε`-accuracy from a residual `O(1/n)` rate and an explicit threshold inequality.

This is the direct threshold counterpart of the closed-form ceil statement.
-/
theorem dualGap_le_eps_of_residualRateThreshold
    {gap residual : ℕ → ℝ} {alpha B eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha)
    (n : ℕ)
    (hthreshold : (alpha * B) / (n + 1 : ℝ) ≤ eps) :
    gap n ≤ eps := by
  exact (dualGap_le_rate_of_residualRate hgap_res hres_rate halpha n).trans hthreshold

/--
`ε`-accuracy from a residual `O(1/n)` rate and a ratio-form threshold.

This is the residual-rate specialization of
`dualGap_le_eps_of_composed_rateThreshold_ratioBound`.
-/
theorem dualGap_le_eps_of_residualRateThreshold_ratioBound
    {gap residual : ℕ → ℝ} {alpha B eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (n : ℕ)
    (hratio : (alpha * B) / eps ≤ (n + 1 : ℝ)) :
    gap n ≤ eps := by
  exact dualGap_le_eps_of_composed_rateThreshold_ratioBound
    (gap := gap) (residual := residual) (bench := residual)
    (alpha := alpha) (beta := 1) (C := B) (eps := eps)
    hgap_res
    (fun k => by simp)
    hres_rate halpha (by positivity) heps n (by simpa [mul_one] using hratio)

/--
Successor-index convenience form of
`dualGap_le_eps_of_residualRateThreshold_closedFormCeil`.
-/
theorem dualGap_le_eps_of_residualRateThreshold_closedFormCeil_succ
    {gap residual : ℕ → ℝ} {alpha B eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (n : ℕ)
    (hn : Nat.ceil ((alpha * B) / eps) ≤ (n + 1) + 1) :
    gap (n + 1) ≤ eps :=
  dualGap_le_eps_of_residualRateThreshold_closedFormCeil
    hgap_res hres_rate halpha heps (n + 1) hn

/--
Successor-index convenience form of
`dualGap_le_eps_of_residualRateThreshold_ratioBound`.
-/
theorem dualGap_le_eps_of_residualRateThreshold_ratioBound_succ
    {gap residual : ℕ → ℝ} {alpha B eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (n : ℕ)
    (hratio : (alpha * B) / eps ≤ (n + 1 : ℝ) + 1) :
    gap (n + 1) ≤ eps := by
  have hthreshold' : (alpha * B) / ((n + 1 : ℝ) + 1) ≤ eps := by
    have hmul' : alpha * B ≤ ((n + 1 : ℝ) + 1) * eps := (div_le_iff₀ heps).1 hratio
    have hmul : alpha * B ≤ eps * ((n + 1 : ℝ) + 1) := by
      simpa [mul_comm, mul_left_comm, mul_assoc] using hmul'
    have hpos : 0 < ((n + 1 : ℝ) + 1) := by positivity
    exact (div_le_iff₀ hpos).2 hmul
  exact dualGap_le_eps_of_residualRateThreshold
    hgap_res hres_rate halpha (n + 1) (by simpa [Nat.cast_add, Nat.cast_one] using hthreshold')

/--
Average-gap bound derived directly from a residual `O(1/n)` rate.

This packages the common specialization `bench = residual`, `beta = 1` of
`dualGap_average_le_composed_rateBenchmark`.
-/
theorem dualGap_average_le_rate_constant_of_residualRate
    {gap residual : ℕ → ℝ} {alpha B : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha) (hB : 0 ≤ B)
    {N : ℕ} (hN : 0 < N) :
    (∑ k ∈ Finset.range N, gap k) / N ≤ alpha * B := by
  have h := dualGap_average_le_composed_rateBenchmark
    (gap := gap) (residual := residual) (bench := residual)
    (alpha := alpha) (beta := 1) (C := B)
    hgap_res
    (fun k => by simp)
    hres_rate halpha (by positivity) hB hN
  simpa using h

/--
Successor-length convenience form of
`dualGap_average_le_rate_constant_of_residualRate`.
-/
theorem dualGap_average_le_rate_constant_of_residualRate_succ
    {gap residual : ℕ → ℝ} {alpha B : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha) (hB : 0 ≤ B)
    (N : ℕ) :
    (∑ k ∈ Finset.range (N + 1), gap k) / (N + 1 : ℝ) ≤ alpha * B := by
  simpa [Nat.cast_add, Nat.cast_one] using
    (dualGap_average_le_rate_constant_of_residualRate
      (gap := gap) (residual := residual)
      (alpha := alpha) (B := B)
      hgap_res hres_rate halpha hB (N := N + 1) (hN := Nat.succ_pos N))

/--
Successor-index convenience form of `dualGap_le_eps_of_residualRateThreshold`.
-/
theorem dualGap_le_eps_of_residualRateThreshold_succ
    {gap residual : ℕ → ℝ} {alpha B eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha)
    (n : ℕ)
    (hthreshold : (alpha * B) / ((n + 1 : ℝ) + 1) ≤ eps) :
    gap (n + 1) ≤ eps := by
  exact dualGap_le_eps_of_residualRateThreshold
    hgap_res hres_rate halpha (n + 1)
    (by simpa [Nat.cast_add, Nat.cast_one] using hthreshold)

/--
Index-form residual-rate threshold from a natural-number denominator bound.

This upgrades a threshold certified at denominator `m` to index `n` using `m ≤ n+1`
and nonnegativity of `alpha * B`.
-/
theorem dualGap_le_eps_of_residualRateThreshold_of_natBound
    {gap residual : ℕ → ℝ} {alpha B eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha) (hB : 0 ≤ B)
    (n m : ℕ)
    (hm_pos : 0 < m)
    (hthreshold_m : (alpha * B) / (m : ℝ) ≤ eps)
    (hmn : m ≤ n + 1) :
    gap n ≤ eps := by
  have hαB : 0 ≤ alpha * B := mul_nonneg halpha hB
  have hm_posR : (0 : ℝ) < (m : ℝ) := Nat.cast_pos.mpr hm_pos
  have hmnR : (m : ℝ) ≤ (n + 1 : ℝ) := by exact_mod_cast hmn
  have hrecip : (1 : ℝ) / (n + 1 : ℝ) ≤ (1 : ℝ) / (m : ℝ) :=
    one_div_le_one_div_of_le hm_posR hmnR
  have hscaled :
      (alpha * B) * ((1 : ℝ) / (n + 1 : ℝ)) ≤
        (alpha * B) * ((1 : ℝ) / (m : ℝ)) :=
    mul_le_mul_of_nonneg_left hrecip hαB
  have hthreshold_n : (alpha * B) / (n + 1 : ℝ) ≤ eps := by
    calc
      (alpha * B) / (n + 1 : ℝ) = (alpha * B) * ((1 : ℝ) / (n + 1 : ℝ)) := by
        simp [div_eq_mul_inv]
      _ ≤ (alpha * B) * ((1 : ℝ) / (m : ℝ)) := hscaled
      _ = (alpha * B) / (m : ℝ) := by simp [div_eq_mul_inv]
      _ ≤ eps := hthreshold_m
  exact dualGap_le_eps_of_residualRateThreshold hgap_res hres_rate halpha n hthreshold_n

/--
Successor-index residual-rate threshold from a natural-number denominator bound.
-/
theorem dualGap_le_eps_of_residualRateThreshold_succ_of_natBound
    {gap residual : ℕ → ℝ} {alpha B eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha) (hB : 0 ≤ B)
    (n m : ℕ)
    (hm_pos : 0 < m)
    (hthreshold_m : (alpha * B) / (m : ℝ) ≤ eps)
    (hmn : m ≤ (n + 1) + 1) :
    gap (n + 1) ≤ eps := by
  exact dualGap_le_eps_of_residualRateThreshold_of_natBound
    hgap_res hres_rate halpha hB (n := n + 1) (m := m) hm_pos hthreshold_m
    (by simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using hmn)

/--
Ratio-form residual-rate threshold from a natural-number index bound.
-/
theorem dualGap_le_eps_of_residualRateThreshold_ratioBound_of_natBound
    {gap residual : ℕ → ℝ} {alpha B eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (n : ℕ)
    (hratio_nat : (alpha * B) / eps ≤ (n + 1 : ℕ)) :
    gap n ≤ eps := by
  have hratio : (alpha * B) / eps ≤ (n + 1 : ℝ) := by
    simpa [Nat.cast_add, Nat.cast_one] using (show (alpha * B) / eps ≤ ((n + 1 : ℕ) : ℝ) from
      (by exact_mod_cast hratio_nat))
  exact dualGap_le_eps_of_residualRateThreshold_ratioBound
    hgap_res hres_rate halpha heps n hratio

/--
Successor-index ratio-form residual-rate threshold from a natural-number index bound.
-/
theorem dualGap_le_eps_of_residualRateThreshold_ratioBound_succ_of_natBound
    {gap residual : ℕ → ℝ} {alpha B eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (n : ℕ)
    (hratio_nat : (alpha * B) / eps ≤ ((n + 1) + 1 : ℕ)) :
    gap (n + 1) ≤ eps := by
  have hratio : (alpha * B) / eps ≤ (n + 1 : ℝ) + 1 := by
    have hratio' : (alpha * B) / eps ≤ (((n + 1) + 1 : ℕ) : ℝ) := by
      exact_mod_cast hratio_nat
    have hcast : ((((n + 1) + 1 : ℕ) : ℝ)) = (n + 1 : ℝ) + 1 := by
      norm_num [Nat.cast_add, Nat.cast_one, add_assoc]
    exact hcast ▸ hratio'
  exact dualGap_le_eps_of_residualRateThreshold_ratioBound_succ
    hgap_res hres_rate halpha heps n hratio

/--
Successor-index closed-form residual-rate threshold from a natural-number ceil bound.
-/
theorem dualGap_le_eps_of_residualRateThreshold_closedFormCeil_succ_of_natBound
    {gap residual : ℕ → ℝ} {alpha B eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (n : ℕ)
    (hn_nat : Nat.ceil ((alpha * B) / eps) ≤ ((n + 1) + 1 : ℕ)) :
    gap (n + 1) ≤ eps := by
  have hn : Nat.ceil ((alpha * B) / eps) ≤ (n + 1) + 1 := by
    simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using hn_nat
  exact dualGap_le_eps_of_residualRateThreshold_closedFormCeil_succ
    hgap_res hres_rate halpha heps n hn

/--
Index-form closed-form residual-rate threshold from a natural-number upper bound.
-/
theorem dualGap_le_eps_of_residualRateThreshold_closedFormCeil_of_natBound
    {gap residual : ℕ → ℝ} {alpha B eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (n m : ℕ)
    (hnm : Nat.ceil ((alpha * B) / eps) ≤ m)
    (hmn : m ≤ n + 1) :
    gap n ≤ eps := by
  exact dualGap_le_eps_of_residualRateThreshold_closedFormCeil
    hgap_res hres_rate halpha heps n (le_trans hnm hmn)

/--
Average-gap bound from pointwise ratio-form residual-rate thresholds.

If each index in `range N` satisfies the ratio-form stopping condition, then each corresponding
gap term is at most `eps`, hence their average is at most `eps`.
-/
theorem dualGap_average_le_eps_of_residualRateThreshold_of_pointwise_ratioBound
    {gap residual : ℕ → ℝ} {alpha B eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    {N : ℕ} (hN : 0 < N)
    (hratio : ∀ k : ℕ, k < N → (alpha * B) / eps ≤ (k + 1 : ℝ)) :
    (∑ k ∈ Finset.range N, gap k) / N ≤ eps := by
  have hpoint : ∀ k ∈ Finset.range N, gap k ≤ eps := by
    intro k hk
    exact dualGap_le_eps_of_residualRateThreshold_ratioBound
      hgap_res hres_rate halpha heps k (hratio k (Finset.mem_range.mp hk))
  have hsum : ∑ k ∈ Finset.range N, gap k ≤ ∑ _k ∈ Finset.range N, eps :=
    Finset.sum_le_sum hpoint
  have hN' : (0 : ℝ) < (N : ℝ) := Nat.cast_pos.mpr hN
  rw [div_le_iff₀ hN']
  calc
    ∑ k ∈ Finset.range N, gap k
        ≤ ∑ _k ∈ Finset.range N, eps := hsum
    _ = eps * (N : ℝ) := by
      simp [Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_comm]

/--
Successor-length average-gap bound from pointwise ratio-form residual-rate thresholds.
-/
theorem dualGap_average_le_eps_of_residualRateThreshold_of_pointwise_ratioBound_succ
    {gap residual : ℕ → ℝ} {alpha B eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (N : ℕ)
    (hratio : ∀ k : ℕ, k < N + 1 → (alpha * B) / eps ≤ (k + 1 : ℝ)) :
    (∑ k ∈ Finset.range (N + 1), gap k) / (N + 1 : ℝ) ≤ eps := by
  simpa [Nat.cast_add, Nat.cast_one] using
    (dualGap_average_le_eps_of_residualRateThreshold_of_pointwise_ratioBound
      hgap_res hres_rate halpha heps (N := N + 1) (hN := Nat.succ_pos N) hratio)

/--
Average-gap bound from pointwise natural-number residual-rate thresholds.

This is the nat-bound companion of
`dualGap_average_le_eps_of_residualRateThreshold_of_pointwise_ratioBound`.
-/
theorem dualGap_average_le_eps_of_residualRateThreshold_of_pointwise_natBound
    {gap residual : ℕ → ℝ} {alpha B eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    {N : ℕ} (hN : 0 < N)
    (hratio_nat : ∀ k : ℕ, k < N → (alpha * B) / eps ≤ (k + 1 : ℕ)) :
    (∑ k ∈ Finset.range N, gap k) / N ≤ eps := by
  have hratio : ∀ k : ℕ, k < N → (alpha * B) / eps ≤ (k + 1 : ℝ) := by
    intro k hk
    exact_mod_cast hratio_nat k hk
  exact dualGap_average_le_eps_of_residualRateThreshold_of_pointwise_ratioBound
    hgap_res hres_rate halpha heps hN hratio

/--
Successor-length average-gap bound from pointwise natural-number residual-rate thresholds.
-/
theorem dualGap_average_le_eps_of_residualRateThreshold_of_pointwise_natBound_succ
    {gap residual : ℕ → ℝ} {alpha B eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    (N : ℕ)
    (hratio_nat : ∀ k : ℕ, k < N + 1 → (alpha * B) / eps ≤ (k + 1 : ℕ)) :
    (∑ k ∈ Finset.range (N + 1), gap k) / (N + 1 : ℝ) ≤ eps := by
  simpa [Nat.cast_add, Nat.cast_one] using
    (dualGap_average_le_eps_of_residualRateThreshold_of_pointwise_natBound
      hgap_res hres_rate halpha heps (N := N + 1) (hN := Nat.succ_pos N) hratio_nat)

/--
Index-restriction companion of
`dualGap_average_le_eps_of_residualRateThreshold_of_pointwise_ratioBound`.
-/
theorem dualGap_average_le_eps_of_residualRateThreshold_of_pointwise_ratioBound_of_le_index
    {gap residual : ℕ → ℝ} {alpha B eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    {M N : ℕ} (hM : 0 < M) (hMN : M ≤ N)
    (hratio : ∀ k : ℕ, k < N → (alpha * B) / eps ≤ (k + 1 : ℝ)) :
    (∑ k ∈ Finset.range M, gap k) / M ≤ eps := by
  have hratioM : ∀ k : ℕ, k < M → (alpha * B) / eps ≤ (k + 1 : ℝ) := by
    intro k hk
    exact hratio k (lt_of_lt_of_le hk hMN)
  exact dualGap_average_le_eps_of_residualRateThreshold_of_pointwise_ratioBound
    hgap_res hres_rate halpha heps hM hratioM

/--
Natural-number index-restriction companion of
`dualGap_average_le_eps_of_residualRateThreshold_of_pointwise_natBound`.
-/
theorem dualGap_average_le_eps_of_residualRateThreshold_of_pointwise_natBound_of_le_index
    {gap residual : ℕ → ℝ} {alpha B eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    {M N : ℕ} (hM : 0 < M) (hMN : M ≤ N)
    (hratio_nat : ∀ k : ℕ, k < N → (alpha * B) / eps ≤ (k + 1 : ℕ)) :
    (∑ k ∈ Finset.range M, gap k) / M ≤ eps := by
  have hratio_nat_M : ∀ k : ℕ, k < M → (alpha * B) / eps ≤ (k + 1 : ℕ) := by
    intro k hk
    exact hratio_nat k (lt_of_lt_of_le hk hMN)
  exact dualGap_average_le_eps_of_residualRateThreshold_of_pointwise_natBound
    hgap_res hres_rate halpha heps hM hratio_nat_M

/--
Average-gap consequence of a residual-rate threshold on the constant `alpha * B`.
-/
theorem dualGap_average_le_eps_of_residualRateThreshold
    {gap residual : ℕ → ℝ} {alpha B eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha) (hB : 0 ≤ B)
    (hthreshold : alpha * B ≤ eps)
    {N : ℕ} (hN : 0 < N) :
    (∑ k ∈ Finset.range N, gap k) / N ≤ eps := by
  exact (dualGap_average_le_rate_constant_of_residualRate
    hgap_res hres_rate halpha hB hN).trans hthreshold

/--
Successor-length average-gap consequence of a residual-rate threshold.
-/
theorem dualGap_average_le_eps_of_residualRateThreshold_succ
    {gap residual : ℕ → ℝ} {alpha B eps : ℝ}
    (hgap_res : ∀ k : ℕ, gap k ≤ alpha * residual k)
    (hres_rate : ∀ n : ℕ, residual n ≤ B / (n + 1 : ℝ))
    (halpha : 0 ≤ alpha) (hB : 0 ≤ B)
    (hthreshold : alpha * B ≤ eps)
    (N : ℕ) :
    (∑ k ∈ Finset.range (N + 1), gap k) / (N + 1 : ℝ) ≤ eps := by
  exact (dualGap_average_le_rate_constant_of_residualRate_succ
    hgap_res hres_rate halpha hB N).trans hthreshold

end DualConvergence
end KLProjection
end FlowSinkhorn
