import FlowSinkhorn.KLProjection.Setup.VariationGeometry
import FlowSinkhorn.KLProjection.Sweep
import FlowSinkhorn.KLProjection.UniformBound
import FlowSinkhorn.KLProjection.Topical
import Mathlib.Order.Monotone.Basic

/-!
# Block monotonicity and moment maps

This module is reserved for the order-theoretic input from
`papers/kl-projections/sections/sec-nonexpansiveness.tex`.

Paper targets:
- Proposition `prop:block-mono`;
- Lemma `lem:moment-monotone`.

Intended theorem names:
- `blockUpdate_monotone`;
- `momentMap_monotone`;
- `topical_of_blockMonotone_and_translation`.

Design note:
this is where the paper-specific monotonicity proof should be isolated, so that later application
files can simply import a certified topicality statement rather than re-proving order lemmas.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace Setup

-- Paper-facing theorem names are intentionally verbose for traceability.
set_option linter.style.longLine false

variable {ι₁ ι₂ : Type*}

/--
First block-update monotonicity (paper-facing wrapper).
-/
theorem blockUpdate_monotone_1
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (hmono : Monotone Ψ₁) :
    Monotone Ψ₁ :=
  hmono

/--
Second block-update monotonicity (paper-facing wrapper).
-/
theorem blockUpdate_monotone_2
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hmono : Monotone Ψ₂) :
    Monotone Ψ₂ :=
  hmono

/--
Moment-map monotonicity wrapper used by the setup layer.
-/
theorem momentMap_monotone
    {α : Type*} [Preorder α]
    (M : α → α)
    (hmono : Monotone M)
    (x y : α)
    (hxy : x ≤ y) :
    M x ≤ M y :=
  hmono hxy

/--
If both block maps are monotone, then the composed sweep is monotone.
-/
theorem sweep_monotone_of_blockMonotone
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : Monotone Ψ₁)
    (hΨ₂ : Monotone Ψ₂) :
    Monotone (sweep Ψ₁ Ψ₂) := by
  intro u v huv
  exact hΨ₁ (hΨ₂ huv)

/--
Paper-facing packaging of Proposition `prop:block-mono`.

This bundles the two block-update monotonicity statements together with the
monotonicity of the full sweep, so downstream files can reuse a single theorem
when they only need the abstract order-theoretic conclusion.
-/
theorem blockUpdate_monotone
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : Monotone Ψ₁)
    (hΨ₂ : Monotone Ψ₂) :
    Monotone Ψ₁ ∧ Monotone Ψ₂ ∧ Monotone (sweep Ψ₁ Ψ₂) := by
  refine ⟨hΨ₁, hΨ₂, ?_⟩
  exact sweep_monotone_of_blockMonotone Ψ₁ Ψ₂ hΨ₁ hΨ₂

/--
Canonical sweep-level topicality helper.

This is the direct bridge used by the nonexpansive bundle:
monotone block maps plus signed paired-balance translation rules imply
monotone + translation-equivariant sweep.
-/
theorem sweep_topical_of_blockMonotone_and_translation
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂) :
    Monotone (sweep Ψ₁ Ψ₂) ∧ TranslationEquivariant (sweep Ψ₁ Ψ₂) := by
  refine ⟨sweep_monotone_of_blockMonotone Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono, ?_⟩
  exact sweep_translationEquivariant_of_signedBlockTranslationEquivariant τ Ψ₁ Ψ₂
    hΨ₁_trans hΨ₂_trans

/--
Backward-compatible alias for the sweep topicality helper.
-/
theorem topical_of_blockMonotone_and_translation
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂) :
    Monotone (sweep Ψ₁ Ψ₂) ∧ TranslationEquivariant (sweep Ψ₁ Ψ₂) :=
  sweep_topical_of_blockMonotone_and_translation
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans

/--
Paper-facing non-expansiveness consequence of the previous topical packaging.
-/
theorem sweep_variationSeminorm_nonexpansive_of_blockMonotone_and_translation
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (u v : ι₁ → ℝ) :
    variationSeminorm (sweep Ψ₁ Ψ₂ u - sweep Ψ₁ Ψ₂ v) ≤
      variationSeminorm (u - v) := by
  have htop :=
    sweep_topical_of_blockMonotone_and_translation
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
  exact variationSeminorm_nonexpansive_of_topical (sweep Ψ₁ Ψ₂) htop.1 htop.2 u v

/--
Oscillation-form version of the sweep non-expansiveness estimate.

This is the direct paper-facing statement for later quotient / dual arguments:
the sweep does not increase oscillation once the block monotonicity and translation
assumptions are in place.
-/
theorem sweep_oscillation_nonexpansive_of_blockMonotone_and_translation
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (u v : ι₁ → ℝ) :
    oscillation (sweep Ψ₁ Ψ₂ u - sweep Ψ₁ Ψ₂ v) ≤ oscillation (u - v) := by
  have h :=
    sweep_variationSeminorm_nonexpansive_of_blockMonotone_and_translation
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans u v
  have h' :
      oscillation (sweep Ψ₁ Ψ₂ u - sweep Ψ₁ Ψ₂ v) / 2 ≤
        oscillation (u - v) / 2 := by
    simpa [variationSeminorm] using h
  linarith

/--
Compact sweep non-expansiveness package.

This is the helper theorem downstream files can import when they want both the
oscillation and variation versions at once.
-/
theorem sweep_nonexpansive_of_blockMonotone_and_translation
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (u v : ι₁ → ℝ) :
    oscillation (sweep Ψ₁ Ψ₂ u - sweep Ψ₁ Ψ₂ v) ≤ oscillation (u - v) ∧
      variationSeminorm (sweep Ψ₁ Ψ₂ u - sweep Ψ₁ Ψ₂ v) ≤
        variationSeminorm (u - v) := by
  refine ⟨sweep_oscillation_nonexpansive_of_blockMonotone_and_translation
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans u v, ?_⟩
  exact sweep_variationSeminorm_nonexpansive_of_blockMonotone_and_translation
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans u v

/--
Canonical topical sweep package for downstream imports.

This is the preferred Setup-layer entry point when a later file wants the full sweep
interface at once: monotonicity, translation equivariance, and both non-expansive
consequences in oscillation and variation form.
-/
theorem sweep_topical_nonexpansive_bundle_of_blockMonotone_and_translation
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (u v : ι₁ → ℝ) :
    Monotone (sweep Ψ₁ Ψ₂) ∧
      TranslationEquivariant (sweep Ψ₁ Ψ₂) ∧
      oscillation (sweep Ψ₁ Ψ₂ u - sweep Ψ₁ Ψ₂ v) ≤ oscillation (u - v) ∧
      variationSeminorm (sweep Ψ₁ Ψ₂ u - sweep Ψ₁ Ψ₂ v) ≤ variationSeminorm (u - v) := by
  have htop :=
    sweep_topical_of_blockMonotone_and_translation
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
  have hosc :=
    sweep_oscillation_nonexpansive_of_blockMonotone_and_translation
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans u v
  have hvør :=
    sweep_variationSeminorm_nonexpansive_of_blockMonotone_and_translation
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans u v
  exact ⟨htop.1, htop.2, hosc, hvør⟩

/--
Thin reusable bundle theorem for downstream imports.

It packages the full topical sweep interface in one call while reusing the
canonical topical and nonexpansive Setup exports.
-/
theorem sweep_topical_nonexpansive_bundle
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (u v : ι₁ → ℝ) :
    Monotone (sweep Ψ₁ Ψ₂) ∧
      TranslationEquivariant (sweep Ψ₁ Ψ₂) ∧
      oscillation (sweep Ψ₁ Ψ₂ u - sweep Ψ₁ Ψ₂ v) ≤ oscillation (u - v) ∧
      variationSeminorm (sweep Ψ₁ Ψ₂ u - sweep Ψ₁ Ψ₂ v) ≤ variationSeminorm (u - v) := by
  exact sweep_topical_nonexpansive_bundle_of_blockMonotone_and_translation
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans u v

/--
Backward-compatible alias for older imports.
-/
theorem sweep_topicalPackage_of_blockMonotone_and_translation
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (u v : ι₁ → ℝ) :
    Monotone (sweep Ψ₁ Ψ₂) ∧
      TranslationEquivariant (sweep Ψ₁ Ψ₂) ∧
      oscillation (sweep Ψ₁ Ψ₂ u - sweep Ψ₁ Ψ₂ v) ≤ oscillation (u - v) ∧
      variationSeminorm (sweep Ψ₁ Ψ₂ u - sweep Ψ₁ Ψ₂ v) ≤ variationSeminorm (u - v) :=
  sweep_topical_nonexpansive_bundle_of_blockMonotone_and_translation
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans u v

/--
Alias kept for paper-facing stability.
-/
theorem variationSeminorm_nonexpansive_of_blockMonotone_and_translation
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (u v : ι₁ → ℝ) :
    variationSeminorm (sweep Ψ₁ Ψ₂ u - sweep Ψ₁ Ψ₂ v) ≤
      variationSeminorm (u - v) :=
  sweep_variationSeminorm_nonexpansive_of_blockMonotone_and_translation
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans u v

/-!
## Concrete monotonicity building blocks

The following utility lemmas explain *why* block updates are monotone in the
KL/Sinkhorn context: coordinatewise application, pointwise addition, constant
maps, and the identity are all monotone.
-/

/--
Applying a monotone function coordinatewise gives a monotone map on function
spaces with the pointwise order.
-/
theorem coordwise_monotone
    {ι : Type*} [Nonempty ι]
    {f : ℝ → ℝ} (hf : Monotone f) :
    Monotone (fun (x : ι → ℝ) => fun i => f (x i)) := by
  intro x y hxy i
  exact hf (hxy i)

/--
The pointwise sum of two monotone maps is monotone.
This captures, e.g., `alpha_i ↦ alpha_i + g(alpha, beta)` when both summands
are individually monotone.
-/
theorem pointwise_add_monotone
    {ι : Type*}
    {T₁ T₂ : (ι → ℝ) → (ι → ℝ)}
    (hT₁ : Monotone T₁) (hT₂ : Monotone T₂) :
    Monotone (fun x => T₁ x + T₂ x) := by
  intro x y hxy i
  exact add_le_add (hT₁ hxy i) (hT₂ hxy i)

/--
Any constant map (returning a fixed function) is monotone with respect to the
pointwise order on function spaces.
-/
theorem const_monotone
    {ι : Type*} (c : ι → ℝ) :
    Monotone (fun _ : ι → ℝ => c) := by
  intro _ _ _ i
  exact le_refl _

/--
The identity map on a function space is monotone.
(Named with `_lean` suffix to avoid clashing with `Monotone.id`.)
-/
theorem id_monotone_lean
    {ι : Type*} :
    Monotone (id : (ι → ℝ) → (ι → ℝ)) :=
  fun _ _ h => h

/--
Paper-facing alias: composition of two monotone updates is monotone.
This is the structural fact used in `sweep_monotone_of_blockMonotone`.
-/
theorem sweep_comp_monotone
    {ι : Type*} [Nonempty ι]
    {T₁ T₂ : (ι → ℝ) → (ι → ℝ)}
    (h₁ : Monotone T₁) (h₂ : Monotone T₂) :
    Monotone (T₂ ∘ T₁) :=
  h₂.comp h₁

/--
If T₁ and T₂ are both non-expansive w.r.t. p, then T₂ ∘ T₁ is also non-expansive.
-/
theorem sweep_comp_nonexpansive
    {𝕜 E : Type*} [NormedField 𝕜] [AddCommGroup E] [Module 𝕜 E]
    (p : Seminorm 𝕜 E)
    {T₁ T₂ : E → E}
    (h₁ : SeminormNonexpansive p T₁) (h₂ : SeminormNonexpansive p T₂) :
    SeminormNonexpansive p (T₂ ∘ T₁) := by
  intro x y
  calc p (T₂ (T₁ x) - T₂ (T₁ y))
      ≤ p (T₁ x - T₁ y) := h₂ (T₁ x) (T₁ y)
    _ ≤ p (x - y) := h₁ x y

/--
Sweep is `SeminormNonexpansive` for the variation seminorm.

This is the direct chain from block monotonicity + block translation-equivariance to the
`SeminormNonexpansive variationSeminormAsSeminorm` property required by the generic blueprint.

Chain:
1. `sweep_monotone_of_blockMonotone` → `Monotone (sweep Ψ₁ Ψ₂)`
2. `sweep_translationEquivariant_of_signedBlockTranslationEquivariant` →
   `TranslationEquivariant (sweep Ψ₁ Ψ₂)`
3. `topical_implies_variationSeminorm_nonexpansive` (Blueprint bridge) →
   `SeminormNonexpansive variationSeminormAsSeminorm (sweep Ψ₁ Ψ₂)`

This concretizes the abstract Proposition `prop:topical-nonexpansive` + sweep structure argument.
-/
theorem sweep_SeminormNonexpansive_of_blockMonotone_translation
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂) :
    SeminormNonexpansive variationSeminormAsSeminorm (sweep Ψ₁ Ψ₂) := by
  intro x y
  change variationSeminorm (sweep Ψ₁ Ψ₂ x - sweep Ψ₁ Ψ₂ y) ≤ variationSeminorm (x - y)
  exact variationSeminorm_nonexpansive_of_topical (sweep Ψ₁ Ψ₂)
    (sweep_monotone_of_blockMonotone Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono)
    (sweep_translationEquivariant_of_signedBlockTranslationEquivariant τ Ψ₁ Ψ₂ hΨ₁_trans hΨ₂_trans)
    x y

/--
Paper-facing non-expansiveness of the sweep iterate.

If the sweep `sweep Ψ₁ Ψ₂` is `SeminormNonexpansive variationSeminormAsSeminorm`, then
so is every iterate `(sweep Ψ₁ Ψ₂)^[k]`.
-/
theorem sweep_iterate_SeminormNonexpansive
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ) :
    SeminormNonexpansive variationSeminormAsSeminorm ((sweep Ψ₁ Ψ₂)^[k]) := by
  have hSweep := sweep_SeminormNonexpansive_of_blockMonotone_translation
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
  intro x y
  exact SeminormNonexpansive_iterate variationSeminormAsSeminorm (sweep Ψ₁ Ψ₂) hSweep k x y

/--
Uniform orbit bound for sweep iterates under block monotonicity and translation equivariance.

If the block maps are monotone and satisfy the signed translation conditions, and if `uStar` is
a fixed point of `sweep Ψ₁ Ψ₂` with `variationSeminorm uStar ≤ B`, then for every starting
point `u0` and every iterate count `k`:
```
variationSeminorm ((sweep Ψ₁ Ψ₂)^[k] u0) ≤ variationSeminorm u0 + 2 * B
```

This is the concrete combination of `sweep_SeminormNonexpansive_of_blockMonotone_translation`
with the abstract `seminorm_iterate_le_of_nonexpansive_fixedPoint_bound`.
-/
theorem sweep_orbit_bound_of_blockMonotone_translation
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B : ℝ} (hbound : variationSeminorm uStar ≤ B)
    (k : ℕ) :
    variationSeminorm ((sweep Ψ₁ Ψ₂)^[k] u0) ≤ variationSeminorm u0 + 2 * B := by
  have hSweep := sweep_SeminormNonexpansive_of_blockMonotone_translation
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
  have hiter :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint
      variationSeminormAsSeminorm (sweep Ψ₁ Ψ₂) hSweep (uStar := uStar) (u0 := u0) hfix k
  have hiter' : variationSeminorm ((sweep Ψ₁ Ψ₂)^[k] u0) ≤
      variationSeminorm u0 + 2 * variationSeminorm uStar := hiter
  linarith

/--
Orbit bound for sweep iterates starting from a point with zero variation seminorm.

If the hypotheses of `sweep_orbit_bound_of_blockMonotone_translation` hold and additionally
`variationSeminorm u0 = 0`, then every iterate satisfies:
```
variationSeminorm ((sweep Ψ₁ Ψ₂)^[k] u0) ≤ 2 * B
```

This is the specialization to translation-invariant (constant-shift) starting points.
-/
theorem sweep_orbit_bound_from_zero_of_blockMonotone_translation
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B : ℝ} (hbound : variationSeminorm uStar ≤ B)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((sweep Ψ₁ Ψ₂)^[k] u0) ≤ 2 * B := by
  have horbit := sweep_orbit_bound_of_blockMonotone_translation (u0 := u0)
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix hbound k
  linarith [hzero]

/--
Paper-facing iterate orbit bound directly from block conditions.

Under block monotonicity and signed translation-equivariance, every sweep iterate is bounded
relative to any fixed point of the sweep:
```
variationSeminorm ((sweep Ψ₁ Ψ₂)^[k] u0) ≤ variationSeminorm u0 + 2 * variationSeminorm uStar
```

This is the concrete Setup-layer bridge from block assumptions to an explicit iterate budget,
with no extra scalar-bound parameter.
-/
theorem sweep_orbit_bound_of_blockConditions
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    (k : ℕ) :
    variationSeminorm ((sweep Ψ₁ Ψ₂)^[k] u0) ≤
      variationSeminorm u0 + 2 * variationSeminorm uStar := by
  simpa using sweep_orbit_bound_of_blockMonotone_translation
    (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
    hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
    (uStar := uStar) (u0 := u0) hfix (B := variationSeminorm uStar) (le_rfl) k

/--
One-stop-shop sweep iteration complexity from block conditions.

Given block-level monotonicity and translation-equivariance conditions on Ψ₁ and Ψ₂,
a fixed point `uStar` with bounded variation `variationSeminorm uStar ≤ B`,
a starting point `u0` with `variationSeminorm u0 = 0`,
and a master O(1/k) dual rate `gap n ≤ alpha * C / (n + 1)` where `C = 2 * B`,
this theorem gives the stopping rule: once `n ≥ alpha * 2 * B / eps - 1`, we have `gap n ≤ eps`.

This packages the full chain:
1. block conditions → `SeminormNonexpansive variationSeminormAsSeminorm (sweep Ψ₁ Ψ₂)`;
2. nonexpansive + fixed point bound → orbit bound ≤ 2 * B;
3. master rate + stopping rule → `gap n ≤ eps`.

Note: the master rate hypothesis here is stated abstractly; applications specialize B
to their concrete orbit budget constant.
-/
theorem sweep_iterationComplexity_from_blockConditions
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B alpha eps : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hzero : variationSeminorm u0 = 0)
    (halpha : 0 ≤ alpha)
    (heps : 0 < eps)
    {gap : ℕ → ℝ}
    (hmaster : ∀ n : ℕ, gap n ≤ alpha * (2 * B) / (n + 1 : ℝ))
    (n : ℕ)
    (hn : Nat.ceil (alpha * (2 * B) / eps) ≤ n + 1) :
    gap n ≤ eps := by
  have hOrbit : variationSeminorm ((sweep Ψ₁ Ψ₂)^[n] u0) ≤ 2 * B :=
    sweep_orbit_bound_from_zero_of_blockMonotone_translation
      (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
      hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix hB hzero n
  have hTwoB_nonneg : 0 ≤ 2 * B := by
    exact le_trans (variationSeminorm_nonneg ((sweep Ψ₁ Ψ₂)^[n] u0)) hOrbit
  have hAlphaTwoB_nonneg : 0 ≤ alpha * (2 * B) := by nlinarith [halpha, hTwoB_nonneg]
  apply (hmaster n).trans
  have hnn : (0 : ℝ) < (n : ℝ) + 1 := by positivity
  have _hFrac_nonneg : 0 ≤ alpha * (2 * B) / ((n : ℝ) + 1) :=
    div_nonneg hAlphaTwoB_nonneg (le_of_lt hnn)
  have hle : (Nat.ceil (alpha * (2 * B) / eps) : ℝ) ≤ ↑n + 1 := by exact_mod_cast hn
  have hceil : alpha * (2 * B) / eps ≤ (Nat.ceil (alpha * (2 * B) / eps) : ℝ) := Nat.le_ceil _
  have hkey : alpha * (2 * B) / eps ≤ ↑n + 1 := le_trans hceil hle
  rw [div_le_iff₀ hnn]
  have := (div_le_iff₀ heps).mp hkey
  linarith

/--
The sweep of two block maps satisfying block-level monotonicity and signed translation
equivariance is topical (`IsTopical`).

This theorem packages the block-level conditions into the abstract `IsTopical` predicate
from `Topical.lean`, providing the cleanest entry point for any module that wants
to use `IsTopical (sweep Ψ₁ Ψ₂)`.
-/
theorem isTopical_sweep_of_blockConditions
    [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂) :
    IsTopical (sweep Ψ₁ Ψ₂) :=
  ⟨sweep_monotone_of_blockMonotone Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono,
   sweep_translationEquivariant_of_signedBlockTranslationEquivariant
     τ Ψ₁ Ψ₂ hΨ₁_trans hΨ₂_trans⟩

/--
Every iterate of a sweep satisfying block-level monotonicity and signed translation
equivariance is `SeminormNonexpansive variationSeminormAsSeminorm`.

This is the most direct bridge from block conditions to the generic blueprint requirement,
using the `IsTopical` → `SeminormNonexpansive` chain.
-/
theorem SeminormNonexpansive_sweep_iterate_of_blockConditions
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ) :
    SeminormNonexpansive variationSeminormAsSeminorm ((sweep Ψ₁ Ψ₂)^[k]) :=
  SeminormNonexpansive_iterate_of_isTopical
    (isTopical_sweep_of_blockConditions τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans) k

/--
Every iterate of the block-conditions sweep is topical.

This light wrapper lets downstream files stay in the `IsTopical` interface when
proving iterate-level facts.
-/
theorem sweep_isTopical_iterate_of_blockConditions
    [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ) :
    IsTopical ((sweep Ψ₁ Ψ₂)^[k]) :=
  isTopical_iterate
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans) k

/--
Iterate-level variation-seminorm nonexpansiveness directly from block conditions.
-/
theorem variationSeminorm_nonexpansive_sweep_iterate_of_blockConditions
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ) (u v : ι₁ → ℝ) :
    variationSeminorm (((sweep Ψ₁ Ψ₂)^[k]) u - ((sweep Ψ₁ Ψ₂)^[k]) v) ≤
      variationSeminorm (u - v) :=
  variationSeminorm_nonexpansive_iterate_of_isTopical
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans) k u v

/--
Iterate-level oscillation nonexpansiveness directly from block conditions.
-/
theorem oscillation_nonexpansive_sweep_iterate_of_blockConditions
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ) (u v : ι₁ → ℝ) :
    oscillation (((sweep Ψ₁ Ψ₂)^[k]) u - ((sweep Ψ₁ Ψ₂)^[k]) v) ≤
      oscillation (u - v) :=
  oscillation_nonexpansive_iterate_of_isTopical
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans) k u v

/--
Orbit bound from block conditions, routed through the `IsTopical` bridge.

Compared to `sweep_orbit_bound_of_blockConditions`, this theorem highlights the
short chain `block conditions -> IsTopical -> SeminormNonexpansive -> orbit bound`.
-/
theorem sweep_orbit_bound_of_blockConditions_via_isTopical
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    (k : ℕ) :
    variationSeminorm ((sweep Ψ₁ Ψ₂)^[k] u0) ≤
      variationSeminorm u0 + 2 * variationSeminorm uStar := by
  have htop : IsTopical (sweep Ψ₁ Ψ₂) :=
    isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
  have hnonexp :
      SeminormNonexpansive variationSeminormAsSeminorm (sweep Ψ₁ Ψ₂) :=
    SeminormNonexpansive_variationSeminormAsSeminorm_of_isTopical htop
  simpa using seminorm_iterate_le_of_nonexpansive_fixedPoint
    variationSeminormAsSeminorm (sweep Ψ₁ Ψ₂) hnonexp
    (uStar := uStar) (u0 := u0) hfix k

/--
Orbit bound from block conditions with an explicit fixed-point budget parameter.

This packages the chain
`block conditions -> IsTopical -> SeminormNonexpansive -> uniform orbit bound`.
-/
theorem sweep_orbit_bound_with_budget_of_blockConditions_via_isTopical
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B : ℝ} (hB : variationSeminorm uStar ≤ B)
    (k : ℕ) :
    variationSeminorm ((sweep Ψ₁ Ψ₂)^[k] u0) ≤ variationSeminorm u0 + 2 * B := by
  have htop : IsTopical (sweep Ψ₁ Ψ₂) :=
    isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
  exact seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
    variationSeminormAsSeminorm (sweep Ψ₁ Ψ₂)
    (SeminormNonexpansive_variationSeminormAsSeminorm_of_isTopical htop)
    hfix hB k

/--
Orbit bound from block conditions with budget lifting.

If `variationSeminorm uStar ≤ B` and `B ≤ U`, then the iterate orbit bound upgrades
from budget `B` to budget `U`.
-/
theorem sweep_orbit_bound_with_budget_of_blockConditions_via_isTopical_of_bound_le
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B U : ℝ} (hB : variationSeminorm uStar ≤ B) (hBU : B ≤ U)
    (k : ℕ) :
    variationSeminorm ((sweep Ψ₁ Ψ₂)^[k] u0) ≤ variationSeminorm u0 + 2 * U :=
  variationSeminorm_orbitBound_with_base_of_isTopical_of_bound_le
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    hfix hB hBU k

/--
Zero-seed orbit bound from block conditions via the `IsTopical` bridge.
-/
theorem sweep_orbit_bound_from_zero_of_blockConditions_via_isTopical
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B : ℝ} (hB : variationSeminorm uStar ≤ B)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((sweep Ψ₁ Ψ₂)^[k] u0) ≤ 2 * B :=
  variationSeminorm_orbitBound_from_zero_of_isTopical
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    hfix hB hzero k

/--
Zero-seed orbit bound with budget lifting from block conditions.
-/
theorem sweep_orbit_bound_from_zero_of_blockConditions_via_isTopical_of_bound_le
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B U : ℝ} (hB : variationSeminorm uStar ≤ B) (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((sweep Ψ₁ Ψ₂)^[k] u0) ≤ 2 * U :=
  variationSeminorm_orbitBound_from_zero_of_isTopical_of_bound_le
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    hfix hB hBU hzero k

/--
Successor-index zero-seed orbit bound from block conditions.
-/
theorem sweep_orbit_bound_from_zero_succ_of_blockConditions_via_isTopical
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B : ℝ} (hB : variationSeminorm uStar ≤ B)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((sweep Ψ₁ Ψ₂)^[k + 1] u0) ≤ 2 * B :=
  variationSeminorm_orbitBound_from_zero_of_isTopical_succ
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    hfix hB hzero k

/--
`H_γ/κ` uniform iterate bound from block conditions via the generic blueprint bridge.
-/
theorem sweep_uniformIterateBound_of_blockConditions_via_blueprint
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {kappa cost gamma hGamma : ℝ}
    (hbound : variationSeminorm uStar ≤
      kappa * (cost + gamma * hGamma))
    (k : ℕ) :
    variationSeminorm ((sweep Ψ₁ Ψ₂)^[k] u0) ≤
      variationSeminorm u0 + 2 * (kappa * (cost + gamma * hGamma)) := by
  have htop : IsTopical (sweep Ψ₁ Ψ₂) :=
    isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
  exact seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
    variationSeminormAsSeminorm (sweep Ψ₁ Ψ₂)
    (SeminormNonexpansive_variationSeminormAsSeminorm_of_isTopical htop)
    hfix hbound k

/--
Zero-seed specialization of the `H_γ/κ` iterate bound from block conditions.
-/
theorem sweep_uniformIterateBound_from_zero_of_blockConditions_via_blueprint
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {kappa cost gamma hGamma : ℝ}
    (hbound : variationSeminorm uStar ≤
      kappa * (cost + gamma * hGamma))
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((sweep Ψ₁ Ψ₂)^[k] u0) ≤
      2 * (kappa * (cost + gamma * hGamma)) := by
  have hiter := sweep_uniformIterateBound_of_blockConditions_via_blueprint
    τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
    (uStar := uStar) (u0 := u0) hfix hbound k
  linarith [hiter, hzero]

/--
`κ = 1` specialization of the blueprint-form iterate bound from block conditions.
-/
theorem sweep_uniformIterateBound_of_blockConditions_via_blueprint_kappa_one
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {cost gamma hGamma : ℝ}
    (hbound : variationSeminorm uStar ≤ cost + gamma * hGamma)
    (k : ℕ) :
    variationSeminorm ((sweep Ψ₁ Ψ₂)^[k] u0) ≤
      variationSeminorm u0 + 2 * (cost + gamma * hGamma) := by
  have hbound' : variationSeminorm uStar ≤ (1 : ℝ) * (cost + gamma * hGamma) := by
    simpa [one_mul] using hbound
  simpa using
    sweep_uniformIterateBound_of_blockConditions_via_blueprint
      (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂) (kappa := 1)
      hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans hfix hbound' k

/--
Successor-index variation-seminorm nonexpansiveness from block conditions.

This is a direct `k+1` wrapper around the chain
`block conditions -> IsTopical -> iterate nonexpansiveness`.
-/
theorem variationSeminorm_nonexpansive_sweep_iterate_succ_of_blockConditions
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ) (u v : ι₁ → ℝ) :
    variationSeminorm (((sweep Ψ₁ Ψ₂)^[k + 1]) u - ((sweep Ψ₁ Ψ₂)^[k + 1]) v) ≤
      variationSeminorm (u - v) :=
  variationSeminorm_nonexpansive_iterate_succ_of_isTopical
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans) k u v

/--
Successor-index oscillation nonexpansiveness from block conditions.

This keeps the same bridge as above but in oscillation form.
-/
theorem oscillation_nonexpansive_sweep_iterate_succ_of_blockConditions
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ) (u v : ι₁ → ℝ) :
    oscillation (((sweep Ψ₁ Ψ₂)^[k + 1]) u - ((sweep Ψ₁ Ψ₂)^[k + 1]) v) ≤
      oscillation (u - v) :=
  oscillation_nonexpansive_iterate_succ_of_isTopical
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans) k u v

/--
Successor-index orbit budget bound from block conditions.

This is the `k+1` specialization of the fixed-point budget bridge
`IsTopical -> variationSeminorm_orbitBound_with_base`.
-/
theorem sweep_orbit_bound_with_budget_succ_of_blockConditions_via_isTopical
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B : ℝ} (hB : variationSeminorm uStar ≤ B)
    (k : ℕ) :
    variationSeminorm ((sweep Ψ₁ Ψ₂)^[k + 1] u0) ≤ variationSeminorm u0 + 2 * B :=
  variationSeminorm_orbitBound_with_base_of_isTopical_succ
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    hfix hB k

/--
Stride-iterate zero-seed orbit bound from block conditions with budget lifting.

For any stride `m`, this packages
`block conditions -> IsTopical -> stride iterate orbit bound`.
-/
theorem sweep_iterate_orbit_bound_from_zero_of_blockConditions_via_isTopical_of_bound_le
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (m : ℕ)
    {uStar u0 : ι₁ → ℝ}
    (hfix : ((sweep Ψ₁ Ψ₂)^[m]) uStar = uStar)
    {B U : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[m])^[k]) u0) ≤ 2 * U :=
  variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_bound_le
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    m hfix hB hBU hzero k

/--
Stride-iterate zero-seed orbit bound from block conditions.

For any stride `m`, this is the non-lifted counterpart of
`sweep_iterate_orbit_bound_from_zero_of_blockConditions_via_isTopical_of_bound_le`.
-/
theorem sweep_iterate_orbit_bound_from_zero_of_blockConditions_via_isTopical
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (m : ℕ)
    {uStar u0 : ι₁ → ℝ}
    (hfix : ((sweep Ψ₁ Ψ₂)^[m]) uStar = uStar)
    {B : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[m])^[k]) u0) ≤ 2 * B :=
  variationSeminorm_orbitBound_from_zero_of_isTopical_iterate
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    m hfix hB hzero k

/--
Successor-index stride-iterate zero-seed orbit bound from block conditions.
-/
theorem sweep_iterate_orbit_bound_from_zero_succ_of_blockConditions_via_isTopical
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (m : ℕ)
    {uStar u0 : ι₁ → ℝ}
    (hfix : ((sweep Ψ₁ Ψ₂)^[m]) uStar = uStar)
    {B : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[m])^[k + 1]) u0) ≤ 2 * B :=
  variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    m hfix hB hzero k

/--
Successor-index stride-iterate zero-seed orbit bound from block conditions with budget lifting.
-/
theorem sweep_iterate_orbit_bound_from_zero_succ_of_blockConditions_via_isTopical_of_bound_le
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (m : ℕ)
    {uStar u0 : ι₁ → ℝ}
    (hfix : ((sweep Ψ₁ Ψ₂)^[m]) uStar = uStar)
    {B U : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[m])^[k + 1]) u0) ≤ 2 * U :=
  variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_of_bound_le
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    m hfix hB hBU hzero k

/--
Successor-index orbit budget bound with budget lifting from block conditions.

This upgrades the `k + 1` orbit bound from budget `B` to any `U ≥ B`.
-/
theorem sweep_orbit_bound_with_budget_succ_of_blockConditions_via_isTopical_of_bound_le
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B U : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (k : ℕ) :
    variationSeminorm ((sweep Ψ₁ Ψ₂)^[k + 1] u0) ≤ variationSeminorm u0 + 2 * U :=
  variationSeminorm_orbitBound_with_base_of_isTopical_succ_of_bound_le
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    hfix hB hBU k

/--
Successor-index zero-seed orbit bound with budget lifting from block conditions.

This is the `k + 1` + zero-seed counterpart of the budget-lifted orbit bridge.
-/
theorem sweep_orbit_bound_from_zero_succ_of_blockConditions_via_isTopical_of_bound_le
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B U : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((sweep Ψ₁ Ψ₂)^[k + 1] u0) ≤ 2 * U :=
  variationSeminorm_orbitBound_from_zero_of_isTopical_succ_of_bound_le
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    hfix hB hBU hzero k

/--
Stride-iterate orbit bound with base budget from block conditions.

For any stride `m`, this packages
`block conditions -> IsTopical -> stride iterate orbit bound with base budget`.
-/
theorem sweep_iterate_orbit_bound_with_budget_of_blockConditions_via_isTopical
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (m : ℕ)
    {uStar u0 : ι₁ → ℝ}
    (hfix : ((sweep Ψ₁ Ψ₂)^[m]) uStar = uStar)
    {B : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[m])^[k]) u0) ≤ variationSeminorm u0 + 2 * B :=
  variationSeminorm_orbitBound_with_base_of_isTopical_iterate
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    m hfix hB k

/--
Stride-iterate orbit bound with base-budget lifting from block conditions.

For any stride `m`, this upgrades budget `B` to any `U ≥ B` along iterate orbits.
-/
theorem sweep_iterate_orbit_bound_with_budget_of_blockConditions_via_isTopical_of_bound_le
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (m : ℕ)
    {uStar u0 : ι₁ → ℝ}
    (hfix : ((sweep Ψ₁ Ψ₂)^[m]) uStar = uStar)
    {B U : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[m])^[k]) u0) ≤ variationSeminorm u0 + 2 * U :=
  variationSeminorm_orbitBound_with_base_of_isTopical_iterate_of_bound_le
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    m hfix hB hBU k

/--
Stride-iterate orbit bound with base budget from block conditions and a fixed point of `sweep`.

This avoids restating the iterate fixed-point hypothesis
`((sweep Ψ₁ Ψ₂)^[m]) uStar = uStar` when `sweep Ψ₁ Ψ₂ uStar = uStar` is already known.
-/
theorem sweep_iterate_orbit_bound_with_budget_fixedPoint_of_blockConditions
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (m : ℕ)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[m])^[k]) u0) ≤ variationSeminorm u0 + 2 * B :=
  variationSeminorm_orbitBound_with_base_of_isTopical_iterate_of_fixedPoint
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    m hfix hB k

/--
Successor-index stride-iterate orbit bound with base budget from block conditions
and a fixed point of `sweep`.
-/
theorem sweep_iterate_orbit_bound_with_budget_succ_fixedPoint_of_blockConditions
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (m : ℕ)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[m])^[k + 1]) u0) ≤ variationSeminorm u0 + 2 * B :=
  variationSeminorm_orbitBound_with_base_of_isTopical_iterate_succ_of_fixedPoint
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    m hfix hB k

/--
Stride-iterate orbit bound with budget lifting from block conditions
and a fixed point of `sweep`.
-/
theorem sweep_iterate_orbit_bound_with_budget_fixedPoint_of_bound_le_of_blockConditions
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (m : ℕ)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B U : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[m])^[k]) u0) ≤ variationSeminorm u0 + 2 * U :=
  variationSeminorm_orbitBound_with_base_of_isTopical_iterate_of_fixedPoint_of_bound_le
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    m hfix hB hBU k

/--
Successor-index stride-iterate orbit bound with budget lifting from block conditions
and a fixed point of `sweep`.
-/
theorem sweep_iterate_orbit_bound_with_budget_succ_fixedPoint_of_bound_le_of_blockConditions
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (m : ℕ)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B U : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[m])^[k + 1]) u0) ≤ variationSeminorm u0 + 2 * U :=
  variationSeminorm_orbitBound_with_base_of_isTopical_iterate_succ_of_fixedPoint_of_bound_le
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    m hfix hB hBU k

/--
One-step stride (`m = 1`) iterate-orbit bound with base budget from block conditions.

This keeps the iterate-of-iterate shape explicit for downstream rewrite pipelines.
-/
theorem sweep_iterate_orbit_bound_with_budget_of_blockConditions_via_isTopical_oneStep_of_fixedPoint
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[1])^[k]) u0) ≤ variationSeminorm u0 + 2 * B := by
  simpa using
    sweep_iterate_orbit_bound_with_budget_fixedPoint_of_blockConditions
      (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
      hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
      (m := 1) (uStar := uStar) (u0 := u0) hfix hB k

/--
Successor-index one-step stride (`m = 1`) iterate-orbit bound with base budget
from block conditions.
-/
theorem sweep_iterate_orbit_bound_with_budget_succ_of_blockConditions_via_isTopical_oneStep_of_fixedPoint
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[1])^[k + 1]) u0) ≤ variationSeminorm u0 + 2 * B := by
  simpa using
    sweep_iterate_orbit_bound_with_budget_succ_fixedPoint_of_blockConditions
      (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
      hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
      (m := 1) (uStar := uStar) (u0 := u0) hfix hB k

/--
One-step stride (`m = 1`) iterate-orbit bound with budget lifting from block conditions.
-/
theorem sweep_iterate_orbit_bound_with_budget_of_blockConditions_via_isTopical_of_bound_le_oneStep
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B U : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[1])^[k]) u0) ≤ variationSeminorm u0 + 2 * U := by
  simpa using
    sweep_iterate_orbit_bound_with_budget_fixedPoint_of_bound_le_of_blockConditions
      (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
      hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
      (m := 1) (uStar := uStar) (u0 := u0) hfix hB hBU k

/--
Stride-iterate zero-seed orbit bound from block conditions and a fixed point of `sweep`.
-/
theorem sweep_iterate_orbit_bound_from_zero_fixedPoint_of_blockConditions
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (m : ℕ)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[m])^[k]) u0) ≤ 2 * B :=
  variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_fixedPoint
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    m hfix hB hzero k

/--
Successor-index stride-iterate zero-seed orbit bound from block conditions
and a fixed point of `sweep`.
-/
theorem sweep_iterate_orbit_bound_from_zero_succ_fixedPoint_of_blockConditions
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (m : ℕ)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[m])^[k + 1]) u0) ≤ 2 * B :=
  variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_of_fixedPoint
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    m hfix hB hzero k

/--
Stride-iterate zero-seed orbit bound with budget lifting from block conditions
and a fixed point of `sweep`.
-/
theorem sweep_iterate_orbit_bound_from_zero_fixedPoint_of_bound_le_of_blockConditions
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (m : ℕ)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B U : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[m])^[k]) u0) ≤ 2 * U :=
  variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_of_fixedPoint_of_bound_le
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    m hfix hB hBU hzero k

/--
Successor-index stride-iterate zero-seed orbit bound with budget lifting from block conditions
and a fixed point of `sweep`.
-/
theorem sweep_iterate_orbit_bound_from_zero_succ_fixedPoint_of_bound_le_of_blockConditions
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (m : ℕ)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B U : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[m])^[k + 1]) u0) ≤ 2 * U :=
  variationSeminorm_orbitBound_from_zero_of_isTopical_iterate_succ_of_fixedPoint_of_bound_le
    (isTopical_sweep_of_blockConditions
      τ Ψ₁ Ψ₂ hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans)
    m hfix hB hBU hzero k

/--
Stride-iterate zero-seed orbit bound with budget lifting from block conditions,
using a fixed point of `sweep`.

This is a `via_isTopical`-named wrapper for the fixed-point bridge.
-/
theorem sweep_iterate_orbit_bound_from_zero_of_blockConditions_via_isTopical_of_bound_le_of_fixedPoint
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (m : ℕ)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B U : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[m])^[k]) u0) ≤ 2 * U := by
  simpa using
    sweep_iterate_orbit_bound_from_zero_fixedPoint_of_bound_le_of_blockConditions
      (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
      hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
      (m := m) (uStar := uStar) (u0 := u0) hfix hB hBU hzero k

/--
Successor-index stride-iterate zero-seed orbit bound with budget lifting
from block conditions, using a fixed point of `sweep`.
-/
theorem sweep_iterate_orbit_bound_from_zero_succ_of_blockConditions_via_isTopical_of_bound_le_of_fixedPoint
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (m : ℕ)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B U : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[m])^[k + 1]) u0) ≤ 2 * U := by
  simpa using
    sweep_iterate_orbit_bound_from_zero_succ_fixedPoint_of_bound_le_of_blockConditions
      (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
      hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
      (m := m) (uStar := uStar) (u0 := u0) hfix hB hBU hzero k

/--
One-step stride (`m = 1`) zero-seed iterate-orbit bound with budget lifting.

This keeps the iterate-of-iterate shape explicit for downstream rewrite pipelines.
-/
theorem sweep_iterate_orbit_bound_from_zero_of_blockConditions_via_isTopical_of_bound_le_oneStep
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B U : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[1])^[k]) u0) ≤ 2 * U := by
  simpa using
    sweep_iterate_orbit_bound_from_zero_of_blockConditions_via_isTopical_of_bound_le_of_fixedPoint
      (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
      hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
      (m := 1) (uStar := uStar) (u0 := u0) hfix hB hBU hzero k

/--
One-step stride (`m = 1`) zero-seed iterate-orbit bound from block conditions.

This is the fixed-point, non-lifted-budget counterpart of the `of_bound_le` one-step wrapper.
-/
theorem sweep_iterate_orbit_bound_from_zero_of_blockConditions_via_isTopical_oneStep_of_fixedPoint
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[1])^[k]) u0) ≤ 2 * B := by
  simpa using
    sweep_iterate_orbit_bound_from_zero_fixedPoint_of_blockConditions
      (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
      hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
      (m := 1) (uStar := uStar) (u0 := u0) hfix hB hzero k

/--
Successor-index one-step stride (`m = 1`) zero-seed iterate-orbit bound from block conditions.
-/
theorem sweep_iterate_orbit_bound_from_zero_succ_of_blockConditions_via_isTopical_oneStep_of_fixedPoint
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[1])^[k + 1]) u0) ≤ 2 * B := by
  simpa using
    sweep_iterate_orbit_bound_from_zero_succ_fixedPoint_of_blockConditions
      (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
      hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
      (m := 1) (uStar := uStar) (u0 := u0) hfix hB hzero k

/--
Successor-index one-step stride (`m = 1`) zero-seed iterate-orbit bound with budget lifting.
-/
theorem sweep_iterate_orbit_bound_from_zero_succ_of_blockConditions_via_isTopical_of_bound_le_oneStep
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B U : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[1])^[k + 1]) u0) ≤ 2 * U := by
  simpa using
    sweep_iterate_orbit_bound_from_zero_succ_of_blockConditions_via_isTopical_of_bound_le_of_fixedPoint
      (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
      hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
      (m := 1) (uStar := uStar) (u0 := u0) hfix hB hBU hzero k

/--
Successor-index one-step stride (`m = 1`) iterate-orbit bound with budget lifting.
-/
theorem sweep_iterate_orbit_bound_with_budget_succ_of_blockConditions_via_isTopical_of_bound_le_oneStep
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B U : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (k : ℕ) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[1])^[k + 1]) u0) ≤ variationSeminorm u0 + 2 * U := by
  simpa using
    sweep_iterate_orbit_bound_with_budget_succ_fixedPoint_of_bound_le_of_blockConditions
      (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
      hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
      (m := 1) (uStar := uStar) (u0 := u0) hfix hB hBU k

/--
`of_le_index` wrapper for the one-step fixed-point iterate-orbit bound with base budget.
-/
theorem sweep_iterate_orbit_bound_with_budget_of_blockConditions_via_isTopical_oneStep_of_fixedPoint_of_le_index
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (k n : ℕ) (_hk : k ≤ n) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[1])^[n]) u0) ≤ variationSeminorm u0 + 2 * B :=
  sweep_iterate_orbit_bound_with_budget_of_blockConditions_via_isTopical_oneStep_of_fixedPoint
    (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
    hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
    (uStar := uStar) (u0 := u0) hfix hB n

/--
`of_le_index` wrapper for the one-step zero-seed iterate-orbit bound with budget lifting.
-/
theorem sweep_iterate_orbit_bound_from_zero_of_blockConditions_via_isTopical_of_bound_le_oneStep_of_le_index
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B U : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (k n : ℕ) (_hk : k ≤ n) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[1])^[n]) u0) ≤ 2 * U :=
  sweep_iterate_orbit_bound_from_zero_of_blockConditions_via_isTopical_of_bound_le_oneStep
    (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
    hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
    (uStar := uStar) (u0 := u0) hfix hB hBU hzero n

/--
Natural-bound wrapper for the one-step fixed-point iterate-orbit base-budget bound.
-/
theorem sweep_iterate_orbit_bound_with_budget_of_blockConditions_via_isTopical_oneStep_of_fixedPoint_of_natBound
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (n N : ℕ) (hNn : N ≤ n) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[1])^[n]) u0) ≤ variationSeminorm u0 + 2 * B :=
  sweep_iterate_orbit_bound_with_budget_of_blockConditions_via_isTopical_oneStep_of_fixedPoint_of_le_index
    (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
    hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
    (uStar := uStar) (u0 := u0) hfix hB N n hNn

/--
Natural-bound wrapper for the one-step zero-seed iterate-orbit budget-lifted bound.
-/
theorem sweep_iterate_orbit_bound_from_zero_of_blockConditions_via_isTopical_of_bound_le_oneStep_of_natBound
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B U : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (n N : ℕ) (hNn : N ≤ n) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[1])^[n]) u0) ≤ 2 * U :=
  sweep_iterate_orbit_bound_from_zero_of_blockConditions_via_isTopical_of_bound_le_oneStep_of_le_index
    (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
    hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
    (uStar := uStar) (u0 := u0) hfix hB hBU hzero N n hNn

/--
`of_le_index` wrapper for the one-step fixed-point successor iterate-orbit base-budget bound.
-/
theorem
    sweep_iterate_orbit_bound_with_budget_succ_of_blockConditions_via_isTopical_oneStep_of_fixedPoint_of_le_index
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (k n : ℕ) (_hk : k ≤ n) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[1])^[n + 1]) u0) ≤
      variationSeminorm u0 + 2 * B :=
  sweep_iterate_orbit_bound_with_budget_succ_of_blockConditions_via_isTopical_oneStep_of_fixedPoint
    (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
    hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
    (uStar := uStar) (u0 := u0) hfix hB n

/--
`of_le_index` wrapper for the one-step successor zero-seed iterate-orbit budget-lifted bound.
-/
theorem
    sweep_iterate_orbit_bound_from_zero_succ_of_blockConditions_via_isTopical_of_bound_le_oneStep_of_le_index
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B U : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (k n : ℕ) (_hk : k ≤ n) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[1])^[n + 1]) u0) ≤ 2 * U :=
  sweep_iterate_orbit_bound_from_zero_succ_of_blockConditions_via_isTopical_of_bound_le_oneStep
    (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
    hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
    (uStar := uStar) (u0 := u0) hfix hB hBU hzero n

/--
Natural-bound wrapper for the one-step fixed-point successor iterate-orbit base-budget bound.
-/
theorem
    sweep_iterate_orbit_bound_with_budget_succ_of_blockConditions_via_isTopical_oneStep_of_fixedPoint_of_natBound
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (n N : ℕ) (hNn : N ≤ n) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[1])^[n + 1]) u0) ≤
      variationSeminorm u0 + 2 * B :=
  sweep_iterate_orbit_bound_with_budget_succ_of_blockConditions_via_isTopical_oneStep_of_fixedPoint_of_le_index
    (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
    hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
    (uStar := uStar) (u0 := u0) hfix hB N n hNn

/--
Natural-bound wrapper for the one-step successor zero-seed iterate-orbit budget-lifted bound.
-/
theorem
    sweep_iterate_orbit_bound_from_zero_succ_of_blockConditions_via_isTopical_of_bound_le_oneStep_of_natBound
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B U : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0)
    (n N : ℕ) (hNn : N ≤ n) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[1])^[n + 1]) u0) ≤ 2 * U :=
  sweep_iterate_orbit_bound_from_zero_succ_of_blockConditions_via_isTopical_of_bound_le_oneStep_of_le_index
    (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
    hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
    (uStar := uStar) (u0 := u0) hfix hB hBU hzero N n hNn

/--
Zero-index specialization of the one-step fixed-point iterate-orbit base-budget bound.
-/
theorem sweep_iterate_orbit_bound_with_budget_of_blockConditions_via_isTopical_oneStep_of_fixedPoint_at_zero
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B : ℝ}
    (hB : variationSeminorm uStar ≤ B) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[1])^[0]) u0) ≤ variationSeminorm u0 + 2 * B :=
  sweep_iterate_orbit_bound_with_budget_of_blockConditions_via_isTopical_oneStep_of_fixedPoint
    (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
    hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
    (uStar := uStar) (u0 := u0) hfix hB 0

/--
Zero-index specialization of the one-step zero-seed iterate-orbit budget-lifted bound.
-/
theorem sweep_iterate_orbit_bound_from_zero_of_blockConditions_via_isTopical_of_bound_le_oneStep_at_zero
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B U : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (hzero : variationSeminorm u0 = 0) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[1])^[0]) u0) ≤ 2 * U :=
  sweep_iterate_orbit_bound_from_zero_of_blockConditions_via_isTopical_of_bound_le_oneStep
    (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
    hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
    (uStar := uStar) (u0 := u0) hfix hB hBU hzero 0

/--
`of_le_index` wrapper for the one-step successor iterate-orbit budget-lifted bound.
-/
theorem
    sweep_iterate_orbit_bound_with_budget_succ_of_blockConditions_via_isTopical_of_bound_le_oneStep_of_le_index
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B U : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (k n : ℕ) (_hk : k ≤ n) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[1])^[n + 1]) u0) ≤
      variationSeminorm u0 + 2 * U :=
  sweep_iterate_orbit_bound_with_budget_succ_of_blockConditions_via_isTopical_of_bound_le_oneStep
    (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
    hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
    (uStar := uStar) (u0 := u0) hfix hB hBU n

/--
Natural-bound wrapper for the one-step successor iterate-orbit budget-lifted bound.
-/
theorem
    sweep_iterate_orbit_bound_with_budget_succ_of_blockConditions_via_isTopical_of_bound_le_oneStep_of_natBound
    [Fintype ι₁] [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁_mono : Monotone Ψ₁)
    (hΨ₂_mono : Monotone Ψ₂)
    (hΨ₁_trans : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂_trans : SignedBlockTranslationEquivariant2 τ Ψ₂)
    {uStar u0 : ι₁ → ℝ}
    (hfix : sweep Ψ₁ Ψ₂ uStar = uStar)
    {B U : ℝ}
    (hB : variationSeminorm uStar ≤ B)
    (hBU : B ≤ U)
    (n N : ℕ) (hNn : N ≤ n) :
    variationSeminorm ((((sweep Ψ₁ Ψ₂)^[1])^[n + 1]) u0) ≤
      variationSeminorm u0 + 2 * U :=
  sweep_iterate_orbit_bound_with_budget_succ_of_blockConditions_via_isTopical_of_bound_le_oneStep_of_le_index
    (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂)
    hΨ₁_mono hΨ₂_mono hΨ₁_trans hΨ₂_trans
    (uStar := uStar) (u0 := u0) hfix hB hBU N n hNn

end Setup
end KLProjection
end FlowSinkhorn
