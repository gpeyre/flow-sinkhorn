import FlowSinkhorn.KLProjection.Sweep

/-!
# Signed paired-balance translation structure

This module is the Lean-side home for the translation-equivariance material from
`papers/kl-projections/sections/sec-nonexpansiveness.tex`.

Paper targets:
- Equation `eq:paired-balance-tau`;
- Proposition `prop:translation-equivariance`.

Intended theorem names:
- `blockTranslation_of_pairedBalance`;
- `sweep_translationEquivariant`;
- `sweep_signedPairedShift_commutes`.

Current status:
- the abstract consequence
  `sweep_translationEquivariant_of_signedBlockTranslationEquivariant`
  is already certified in `FlowSinkhorn.KLProjection.Sweep`.

Future role:
this file should bridge the concrete paired-balance identity proved in the paper to the abstract
block translation hypotheses currently used by the certified sweep theorem.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace Setup

variable {ι₁ ι₂ : Type*}

/--
Paper-facing alias for the first paired-balance block relation.
-/
theorem blockTranslation_of_pairedBalance_1
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁) :
    SignedBlockTranslationEquivariant1 τ Ψ₁ :=
  hΨ₁

/--
Paper-facing alias for the second paired-balance block relation.
-/
theorem blockTranslation_of_pairedBalance_2
    (τ : PairedSign)
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂) :
    SignedBlockTranslationEquivariant2 τ Ψ₂ :=
  hΨ₂

/--
Setup-layer wrapper of Proposition `prop:translation-equivariance`:
signed paired-balance relations on both blocks imply sweep translation equivariance.
-/
theorem sweep_translationEquivariant
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂) :
    TranslationEquivariant (sweep Ψ₁ Ψ₂) := by
  intro u₁ c
  funext i
  calc
    sweep Ψ₁ Ψ₂ (fun k => u₁ k + c) i
        = Ψ₁ (Ψ₂ (fun k => u₁ k + c)) i := by
            rfl
    _ = Ψ₁ (fun j => Ψ₂ u₁ j + τ.toReal * c) i := by
          rw [hΨ₂ u₁ c]
    _ = Ψ₁ (Ψ₂ u₁) i + τ.toReal * (τ.toReal * c) := by
          rw [hΨ₁ (Ψ₂ u₁) (τ.toReal * c)]
    _ = Ψ₁ (Ψ₂ u₁) i + c := by
          rw [← mul_assoc, PairedSign.toReal_mul_self, one_mul]
    _ = sweep Ψ₁ Ψ₂ u₁ i + c := by
          rfl

/--
Concrete commutation form used in notebook-facing explanations:
the sweep commutes with constant shifts.
-/
theorem sweep_signedPairedShift_commutes
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (u₁ : ι₁ → ℝ) (c : ℝ) :
    sweep Ψ₁ Ψ₂ (fun i => u₁ i + c) = fun i => sweep Ψ₁ Ψ₂ u₁ i + c :=
  sweep_translationEquivariant τ Ψ₁ Ψ₂ hΨ₁ hΨ₂ u₁ c

/--
The sweep preserves differences after a common constant shift.

This is the quotient-level form most convenient for later convergence arguments.
-/
theorem sweep_translationEquivariant_sub
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (u v : ι₁ → ℝ) (c : ℝ) :
    sweep Ψ₁ Ψ₂ (fun i => u i + c) - sweep Ψ₁ Ψ₂ (fun i => v i + c) =
      sweep Ψ₁ Ψ₂ u - sweep Ψ₁ Ψ₂ v := by
  funext i
  simp [sweep_translationEquivariant τ Ψ₁ Ψ₂ hΨ₁ hΨ₂ u c,
    sweep_translationEquivariant τ Ψ₁ Ψ₂ hΨ₁ hΨ₂ v c]

/--
The composition of the sweep with itself is translation equivariant when the sweep is.
-/
theorem sweep_translationEquivariant_self_compose
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂) :
    TranslationEquivariant (sweep Ψ₁ Ψ₂ ∘ sweep Ψ₁ Ψ₂) := by
  intro u c
  simp only [Function.comp]
  rw [sweep_translationEquivariant τ Ψ₁ Ψ₂ hΨ₁ hΨ₂]
  rw [sweep_translationEquivariant τ Ψ₁ Ψ₂ hΨ₁ hΨ₂]

/--
The k-th iterate of a translation-equivariant sweep is translation equivariant.
-/
theorem sweep_iterate_translationEquivariant
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ) :
    TranslationEquivariant ((sweep Ψ₁ Ψ₂)^[k]) := by
  induction k with
  | zero => intro u c; simp
  | succ k ih =>
      intro u c
      rw [Function.iterate_succ_apply', Function.iterate_succ_apply']
      rw [ih u c]
      exact sweep_translationEquivariant τ Ψ₁ Ψ₂ hΨ₁ hΨ₂ _ c

/--
Successor-index convenience form of iterate translation equivariance.
-/
theorem sweep_iterate_translationEquivariant_succ
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ) :
    TranslationEquivariant ((sweep Ψ₁ Ψ₂)^[k + 1]) := by
  simpa [Nat.succ_eq_add_one] using
    sweep_iterate_translationEquivariant τ Ψ₁ Ψ₂ hΨ₁ hΨ₂ (k + 1)

/--
Composition of two sweep iterates is translation equivariant.
-/
theorem sweep_iterate_comp_translationEquivariant
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k m : ℕ) :
    TranslationEquivariant (((sweep Ψ₁ Ψ₂)^[k]) ∘ ((sweep Ψ₁ Ψ₂)^[m])) := by
  intro u c
  have hk :
      TranslationEquivariant ((sweep Ψ₁ Ψ₂)^[k]) :=
    sweep_iterate_translationEquivariant τ Ψ₁ Ψ₂ hΨ₁ hΨ₂ k
  have hm :
      TranslationEquivariant ((sweep Ψ₁ Ψ₂)^[m]) :=
    sweep_iterate_translationEquivariant τ Ψ₁ Ψ₂ hΨ₁ hΨ₂ m
  calc
    ((((sweep Ψ₁ Ψ₂)^[k]) ∘ ((sweep Ψ₁ Ψ₂)^[m])) (fun i => u i + c))
        = ((sweep Ψ₁ Ψ₂)^[k]) (((sweep Ψ₁ Ψ₂)^[m]) (fun i => u i + c)) := by
            rfl
    _ = ((sweep Ψ₁ Ψ₂)^[k]) (fun i => ((sweep Ψ₁ Ψ₂)^[m]) u i + c) := by
          rw [hm u c]
    _ = fun i => ((sweep Ψ₁ Ψ₂)^[k]) (((sweep Ψ₁ Ψ₂)^[m]) u) i + c := by
          exact hk _ c
    _ = fun i => ((((sweep Ψ₁ Ψ₂)^[k]) ∘ ((sweep Ψ₁ Ψ₂)^[m])) u) i + c := by
          rfl

/--
Iterated sweep commutes with constant shifts.

This is the iterate-level companion of `sweep_signedPairedShift_commutes`.
-/
theorem sweep_iterate_signedPairedShift_commutes
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ)
    (u : ι₁ → ℝ) (c : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k]) (fun i => u i + c) =
      fun i => ((sweep Ψ₁ Ψ₂)^[k]) u i + c :=
  sweep_iterate_translationEquivariant τ Ψ₁ Ψ₂ hΨ₁ hΨ₂ k u c

/--
Iterated sweep commutes with constant shifts under a direct sweep translation-equivariance
hypothesis.

This bridge reduces assumptions downstream: callers can use any proof of
`TranslationEquivariant (sweep Ψ₁ Ψ₂)` without re-providing signed block witnesses.
-/
theorem sweep_iterate_signedPairedShift_commutes_of_translationEquivariant
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hSweep : TranslationEquivariant (sweep Ψ₁ Ψ₂))
    (k : ℕ)
    (u : ι₁ → ℝ) (c : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k]) (fun i => u i + c) =
      fun i => ((sweep Ψ₁ Ψ₂)^[k]) u i + c := by
  induction k with
  | zero =>
      simp
  | succ k ih =>
      rw [Function.iterate_succ_apply', Function.iterate_succ_apply']
      rw [ih]
      exact hSweep _ c

/--
Iterate-level translation equivariance from a direct sweep translation-equivariance witness.

This packages
`sweep_iterate_signedPairedShift_commutes_of_translationEquivariant` back into
the predicate `TranslationEquivariant ((sweep Ψ₁ Ψ₂)^[k])`.
-/
theorem sweep_iterate_translationEquivariant_of_translationEquivariant
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hSweep : TranslationEquivariant (sweep Ψ₁ Ψ₂))
    (k : ℕ) :
    TranslationEquivariant ((sweep Ψ₁ Ψ₂)^[k]) := by
  intro u c
  exact sweep_iterate_signedPairedShift_commutes_of_translationEquivariant
    Ψ₁ Ψ₂ hSweep k u c

/--
Composition of two sweep iterates under a direct sweep translation-equivariance witness.
-/
theorem sweep_iterate_comp_translationEquivariant_of_translationEquivariant
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hSweep : TranslationEquivariant (sweep Ψ₁ Ψ₂))
    (k m : ℕ) :
    TranslationEquivariant (((sweep Ψ₁ Ψ₂)^[k]) ∘ ((sweep Ψ₁ Ψ₂)^[m])) := by
  intro u c
  have hk :
      TranslationEquivariant ((sweep Ψ₁ Ψ₂)^[k]) :=
    sweep_iterate_translationEquivariant_of_translationEquivariant
      Ψ₁ Ψ₂ hSweep k
  have hm :
      TranslationEquivariant ((sweep Ψ₁ Ψ₂)^[m]) :=
    sweep_iterate_translationEquivariant_of_translationEquivariant
      Ψ₁ Ψ₂ hSweep m
  calc
    ((((sweep Ψ₁ Ψ₂)^[k]) ∘ ((sweep Ψ₁ Ψ₂)^[m])) (fun i => u i + c))
        = ((sweep Ψ₁ Ψ₂)^[k]) (((sweep Ψ₁ Ψ₂)^[m]) (fun i => u i + c)) := by
            rfl
    _ = ((sweep Ψ₁ Ψ₂)^[k]) (fun i => ((sweep Ψ₁ Ψ₂)^[m]) u i + c) := by
          rw [hm u c]
    _ = fun i => ((sweep Ψ₁ Ψ₂)^[k]) (((sweep Ψ₁ Ψ₂)^[m]) u) i + c := by
          exact hk _ c
    _ = fun i => ((((sweep Ψ₁ Ψ₂)^[k]) ∘ ((sweep Ψ₁ Ψ₂)^[m])) u) i + c := by
          rfl

/--
Successor-index form of iterate translation equivariance under a direct sweep
translation-equivariance witness.
-/
theorem sweep_iterate_translationEquivariant_of_translationEquivariant_succ
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hSweep : TranslationEquivariant (sweep Ψ₁ Ψ₂))
    (k : ℕ) :
    TranslationEquivariant ((sweep Ψ₁ Ψ₂)^[k + 1]) := by
  simpa [Nat.succ_eq_add_one] using
    sweep_iterate_translationEquivariant_of_translationEquivariant
      Ψ₁ Ψ₂ hSweep (k + 1)

/--
Successor-left composition form under a direct sweep translation-equivariance witness.
-/
theorem sweep_iterate_comp_translationEquivariant_of_translationEquivariant_succ_left
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hSweep : TranslationEquivariant (sweep Ψ₁ Ψ₂))
    (k m : ℕ) :
    TranslationEquivariant (((sweep Ψ₁ Ψ₂)^[k + 1]) ∘ ((sweep Ψ₁ Ψ₂)^[m])) := by
  simpa [Nat.succ_eq_add_one] using
    sweep_iterate_comp_translationEquivariant_of_translationEquivariant
      Ψ₁ Ψ₂ hSweep (k + 1) m

/--
Successor-right composition form under a direct sweep translation-equivariance witness.
-/
theorem sweep_iterate_comp_translationEquivariant_of_translationEquivariant_succ_right
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hSweep : TranslationEquivariant (sweep Ψ₁ Ψ₂))
    (k m : ℕ) :
    TranslationEquivariant (((sweep Ψ₁ Ψ₂)^[k]) ∘ ((sweep Ψ₁ Ψ₂)^[m + 1])) := by
  simpa [Nat.succ_eq_add_one] using
    sweep_iterate_comp_translationEquivariant_of_translationEquivariant
      Ψ₁ Ψ₂ hSweep k (m + 1)

/--
Index-relaxed iterate translation-equivariance wrapper.

This is convenient when downstream statements track an upper iterate index `k` but only
need translation equivariance at a smaller index `n ≤ k`.
-/
theorem sweep_iterate_translationEquivariant_of_translationEquivariant_of_le_index
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hSweep : TranslationEquivariant (sweep Ψ₁ Ψ₂))
    (k n : ℕ)
    (_hn : n ≤ k) :
    TranslationEquivariant ((sweep Ψ₁ Ψ₂)^[n]) := by
  exact sweep_iterate_translationEquivariant_of_translationEquivariant
    Ψ₁ Ψ₂ hSweep n

/--
Left-index relaxed composition wrapper under a direct sweep translation-equivariance witness.
-/
theorem sweep_iterate_comp_translationEquivariant_of_translationEquivariant_of_le_index_left
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hSweep : TranslationEquivariant (sweep Ψ₁ Ψ₂))
    (k n m : ℕ)
    (_hn : n ≤ k) :
    TranslationEquivariant (((sweep Ψ₁ Ψ₂)^[n]) ∘ ((sweep Ψ₁ Ψ₂)^[m])) := by
  exact sweep_iterate_comp_translationEquivariant_of_translationEquivariant
    Ψ₁ Ψ₂ hSweep n m

/--
Successor-both composition form under a direct sweep translation-equivariance witness.
-/
theorem sweep_iterate_comp_translationEquivariant_of_translationEquivariant_succ_both
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hSweep : TranslationEquivariant (sweep Ψ₁ Ψ₂))
    (k m : ℕ) :
    TranslationEquivariant (((sweep Ψ₁ Ψ₂)^[k + 1]) ∘ ((sweep Ψ₁ Ψ₂)^[m + 1])) := by
  simpa [Nat.succ_eq_add_one] using
    sweep_iterate_comp_translationEquivariant_of_translationEquivariant
      Ψ₁ Ψ₂ hSweep (k + 1) (m + 1)

/--
Difference invariance for shifted inputs under a direct sweep translation-equivariance
hypothesis.

This is the base-difference companion of
`sweep_iterate_signedPairedShift_commutes_of_translationEquivariant`.
-/
theorem sweep_iterate_translationEquivariant_sub_of_translationEquivariant
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hSweep : TranslationEquivariant (sweep Ψ₁ Ψ₂))
    (k : ℕ)
    (u v : ι₁ → ℝ) (c : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k]) (fun i => u i + c) -
        ((sweep Ψ₁ Ψ₂)^[k]) (fun i => v i + c) =
      ((sweep Ψ₁ Ψ₂)^[k]) u - ((sweep Ψ₁ Ψ₂)^[k]) v := by
  funext i
  simp [sweep_iterate_signedPairedShift_commutes_of_translationEquivariant
      Ψ₁ Ψ₂ hSweep k u c,
    sweep_iterate_signedPairedShift_commutes_of_translationEquivariant
      Ψ₁ Ψ₂ hSweep k v c]

/--
Successor-index form of iterate shift commutation under direct sweep translation equivariance.
-/
theorem sweep_iterate_signedPairedShift_commutes_of_translationEquivariant_succ
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hSweep : TranslationEquivariant (sweep Ψ₁ Ψ₂))
    (k : ℕ)
    (u : ι₁ → ℝ) (c : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k + 1]) (fun i => u i + c) =
      fun i => ((sweep Ψ₁ Ψ₂)^[k + 1]) u i + c := by
  simpa [Nat.succ_eq_add_one] using
    sweep_iterate_signedPairedShift_commutes_of_translationEquivariant
      Ψ₁ Ψ₂ hSweep (k + 1) u c

/--
Successor-index form of shifted-difference invariance under direct sweep
translation equivariance.
-/
theorem sweep_iterate_translationEquivariant_sub_of_translationEquivariant_succ
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hSweep : TranslationEquivariant (sweep Ψ₁ Ψ₂))
    (k : ℕ)
    (u v : ι₁ → ℝ) (c : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k + 1]) (fun i => u i + c) -
        ((sweep Ψ₁ Ψ₂)^[k + 1]) (fun i => v i + c) =
      ((sweep Ψ₁ Ψ₂)^[k + 1]) u - ((sweep Ψ₁ Ψ₂)^[k + 1]) v := by
  simpa [Nat.succ_eq_add_one] using
    sweep_iterate_translationEquivariant_sub_of_translationEquivariant
      Ψ₁ Ψ₂ hSweep (k + 1) u v c

/--
Iterated sweep commutes with subtraction of a constant under direct sweep
translation equivariance.
-/
theorem sweep_iterate_translationEquivariant_neg_of_translationEquivariant
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hSweep : TranslationEquivariant (sweep Ψ₁ Ψ₂))
    (k : ℕ)
    (u : ι₁ → ℝ) (c : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k]) (fun i => u i - c) =
      fun i => ((sweep Ψ₁ Ψ₂)^[k]) u i - c := by
  have h :=
    sweep_iterate_signedPairedShift_commutes_of_translationEquivariant
      Ψ₁ Ψ₂ hSweep k u (-c)
  simpa [sub_eq_add_neg] using h

/--
Shift-difference constant form under a direct sweep translation-equivariance hypothesis.

This is the iterate-level "base form" used in quotient arguments: the shifted-minus-unshifted
iterate is exactly a constant function.
-/
theorem sweep_iterate_shift_difference_eq_const_of_translationEquivariant
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hSweep : TranslationEquivariant (sweep Ψ₁ Ψ₂))
    (k : ℕ)
    (u : ι₁ → ℝ) (c : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k]) (fun i => u i + c) - ((sweep Ψ₁ Ψ₂)^[k]) u = fun _ => c := by
  funext i
  have hi :
      ((sweep Ψ₁ Ψ₂)^[k]) (fun j => u j + c) i = ((sweep Ψ₁ Ψ₂)^[k]) u i + c := by
    simpa using congrArg (fun f => f i)
      (sweep_iterate_signedPairedShift_commutes_of_translationEquivariant
        Ψ₁ Ψ₂ hSweep k u c)
  simp [Pi.sub_apply, hi]

/--
Zero-input specialization of iterate shift commutation under sweep translation equivariance.

This is the zero-form bridge used when downstream proofs normalize around the constant-zero seed.
-/
theorem sweep_iterate_constInput_shift_of_translationEquivariant
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hSweep : TranslationEquivariant (sweep Ψ₁ Ψ₂))
    (k : ℕ)
    (c : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k]) (fun _ : ι₁ => c) =
      fun i => ((sweep Ψ₁ Ψ₂)^[k]) (fun _ : ι₁ => (0 : ℝ)) i + c := by
  simpa using sweep_iterate_signedPairedShift_commutes_of_translationEquivariant
    Ψ₁ Ψ₂ hSweep k (fun _ : ι₁ => (0 : ℝ)) c

/--
Two-constant shifted-difference base form under direct sweep translation equivariance.

This bridge keeps track of distinct gauge shifts on each argument and exposes the exact
additive constant `(c₁ - c₂)` separating the two translated iterates.
-/
theorem sweep_iterate_translationEquivariant_sub_two_consts_of_translationEquivariant
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hSweep : TranslationEquivariant (sweep Ψ₁ Ψ₂))
    (k : ℕ)
    (u v : ι₁ → ℝ) (c₁ c₂ : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k]) (fun i => u i + c₁) -
        ((sweep Ψ₁ Ψ₂)^[k]) (fun i => v i + c₂) =
      fun i => (((sweep Ψ₁ Ψ₂)^[k]) u - ((sweep Ψ₁ Ψ₂)^[k]) v) i + (c₁ - c₂) := by
  funext i
  have hu :
      ((sweep Ψ₁ Ψ₂)^[k]) (fun j => u j + c₁) i = ((sweep Ψ₁ Ψ₂)^[k]) u i + c₁ := by
    simpa using congrArg (fun f => f i)
      (sweep_iterate_signedPairedShift_commutes_of_translationEquivariant
        Ψ₁ Ψ₂ hSweep k u c₁)
  have hv :
      ((sweep Ψ₁ Ψ₂)^[k]) (fun j => v j + c₂) i = ((sweep Ψ₁ Ψ₂)^[k]) v i + c₂ := by
    simpa using congrArg (fun f => f i)
      (sweep_iterate_signedPairedShift_commutes_of_translationEquivariant
        Ψ₁ Ψ₂ hSweep k v c₂)
  calc
    (((sweep Ψ₁ Ψ₂)^[k]) (fun j => u j + c₁) -
        ((sweep Ψ₁ Ψ₂)^[k]) (fun j => v j + c₂)) i
        = (((sweep Ψ₁ Ψ₂)^[k]) u i + c₁) - (((sweep Ψ₁ Ψ₂)^[k]) v i + c₂) := by
            simp [Pi.sub_apply, hu, hv]
    _ = (((sweep Ψ₁ Ψ₂)^[k]) u i - ((sweep Ψ₁ Ψ₂)^[k]) v i) + (c₁ - c₂) := by ring
    _ = (((sweep Ψ₁ Ψ₂)^[k]) u - ((sweep Ψ₁ Ψ₂)^[k]) v) i + (c₁ - c₂) := by
          simp [Pi.sub_apply]

/--
For a fixed base input, the gap between two differently shifted iterates is constant.
-/
theorem sweep_iterate_shift_difference_two_consts_eq_const_of_translationEquivariant
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hSweep : TranslationEquivariant (sweep Ψ₁ Ψ₂))
    (k : ℕ)
    (u : ι₁ → ℝ) (c₁ c₂ : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k]) (fun i => u i + c₁) -
        ((sweep Ψ₁ Ψ₂)^[k]) (fun i => u i + c₂) =
      fun _ => (c₁ - c₂) := by
  funext i
  have hu₁ :
      ((sweep Ψ₁ Ψ₂)^[k]) (fun j => u j + c₁) i = ((sweep Ψ₁ Ψ₂)^[k]) u i + c₁ := by
    simpa using congrArg (fun f => f i)
      (sweep_iterate_signedPairedShift_commutes_of_translationEquivariant
        Ψ₁ Ψ₂ hSweep k u c₁)
  have hu₂ :
      ((sweep Ψ₁ Ψ₂)^[k]) (fun j => u j + c₂) i = ((sweep Ψ₁ Ψ₂)^[k]) u i + c₂ := by
    simpa using congrArg (fun f => f i)
      (sweep_iterate_signedPairedShift_commutes_of_translationEquivariant
        Ψ₁ Ψ₂ hSweep k u c₂)
  calc
    (((sweep Ψ₁ Ψ₂)^[k]) (fun j => u j + c₁) -
        ((sweep Ψ₁ Ψ₂)^[k]) (fun j => u j + c₂)) i
        = (((sweep Ψ₁ Ψ₂)^[k]) u i + c₁) - (((sweep Ψ₁ Ψ₂)^[k]) u i + c₂) := by
            simp [Pi.sub_apply, hu₁, hu₂]
    _ = c₁ - c₂ := by ring

/--
Zero-input specialization of the two-constant shift-gap identity.
-/
theorem sweep_iterate_constInput_difference_eq_const_of_translationEquivariant
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hSweep : TranslationEquivariant (sweep Ψ₁ Ψ₂))
    (k : ℕ)
    (c₁ c₂ : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k]) (fun _ : ι₁ => c₁) -
        ((sweep Ψ₁ Ψ₂)^[k]) (fun _ : ι₁ => c₂) =
      fun _ => (c₁ - c₂) := by
  simpa using
    sweep_iterate_shift_difference_two_consts_eq_const_of_translationEquivariant
      Ψ₁ Ψ₂ hSweep k (fun _ : ι₁ => (0 : ℝ)) c₁ c₂

/--
Zero-input two-constant shift-gap identity under signed paired-balance assumptions.
-/
theorem sweep_iterate_constInput_difference_eq_const
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ)
    (c₁ c₂ : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k]) (fun _ : ι₁ => c₁) -
        ((sweep Ψ₁ Ψ₂)^[k]) (fun _ : ι₁ => c₂) =
      fun _ => (c₁ - c₂) := by
  exact sweep_iterate_constInput_difference_eq_const_of_translationEquivariant
    Ψ₁ Ψ₂ (sweep_translationEquivariant τ Ψ₁ Ψ₂ hΨ₁ hΨ₂) k c₁ c₂

/--
Two-constant shifted-difference base form under signed paired-balance assumptions.

This is the paired-balance wrapper of
`sweep_iterate_translationEquivariant_sub_two_consts_of_translationEquivariant`.
-/
theorem sweep_iterate_translationEquivariant_sub_two_consts
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ)
    (u v : ι₁ → ℝ) (c₁ c₂ : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k]) (fun i => u i + c₁) -
        ((sweep Ψ₁ Ψ₂)^[k]) (fun i => v i + c₂) =
      fun i => (((sweep Ψ₁ Ψ₂)^[k]) u - ((sweep Ψ₁ Ψ₂)^[k]) v) i + (c₁ - c₂) := by
  exact sweep_iterate_translationEquivariant_sub_two_consts_of_translationEquivariant
    Ψ₁ Ψ₂ (sweep_translationEquivariant τ Ψ₁ Ψ₂ hΨ₁ hΨ₂) k u v c₁ c₂

/--
Fixed-base two-constant shift-gap identity under signed paired-balance assumptions.

This is the paired-balance wrapper of
`sweep_iterate_shift_difference_two_consts_eq_const_of_translationEquivariant`.
-/
theorem sweep_iterate_shift_difference_two_consts_eq_const
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ)
    (u : ι₁ → ℝ) (c₁ c₂ : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k]) (fun i => u i + c₁) -
        ((sweep Ψ₁ Ψ₂)^[k]) (fun i => u i + c₂) =
      fun _ => (c₁ - c₂) := by
  exact sweep_iterate_shift_difference_two_consts_eq_const_of_translationEquivariant
    Ψ₁ Ψ₂ (sweep_translationEquivariant τ Ψ₁ Ψ₂ hΨ₁ hΨ₂) k u c₁ c₂

/--
Zero-input shift commutation under signed paired-balance assumptions.

This is the paired-balance wrapper of
`sweep_iterate_constInput_shift_of_translationEquivariant`.
-/
theorem sweep_iterate_constInput_shift
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ)
    (c : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k]) (fun _ : ι₁ => c) =
      fun i => ((sweep Ψ₁ Ψ₂)^[k]) (fun _ : ι₁ => (0 : ℝ)) i + c := by
  exact sweep_iterate_constInput_shift_of_translationEquivariant
    Ψ₁ Ψ₂ (sweep_translationEquivariant τ Ψ₁ Ψ₂ hΨ₁ hΨ₂) k c

/--
The `k`-iterate of the sweep preserves differences after a common shift.

This is the iterate analogue of `sweep_translationEquivariant_sub` and is useful
for quotient arguments that compare two trajectories modulo a shared gauge.
-/
theorem sweep_iterate_translationEquivariant_sub
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ)
    (u v : ι₁ → ℝ) (c : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k]) (fun i => u i + c) -
        ((sweep Ψ₁ Ψ₂)^[k]) (fun i => v i + c) =
      ((sweep Ψ₁ Ψ₂)^[k]) u - ((sweep Ψ₁ Ψ₂)^[k]) v := by
  funext i
  simp [sweep_iterate_translationEquivariant τ Ψ₁ Ψ₂ hΨ₁ hΨ₂ k u c,
    sweep_iterate_translationEquivariant τ Ψ₁ Ψ₂ hΨ₁ hΨ₂ k v c]

/--
Successor-index convenience form of iterate shift-commutation.
-/
theorem sweep_iterate_signedPairedShift_commutes_succ
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ)
    (u : ι₁ → ℝ) (c : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k + 1]) (fun i => u i + c) =
      fun i => ((sweep Ψ₁ Ψ₂)^[k + 1]) u i + c := by
  simpa [Nat.succ_eq_add_one] using
    sweep_iterate_signedPairedShift_commutes τ Ψ₁ Ψ₂ hΨ₁ hΨ₂ (k + 1) u c

/--
The `k`-iterate of the sweep commutes with subtraction of a constant.
-/
theorem sweep_iterate_translationEquivariant_neg
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ)
    (u : ι₁ → ℝ) (c : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k]) (fun i => u i - c) =
      fun i => ((sweep Ψ₁ Ψ₂)^[k]) u i - c := by
  have h :=
    sweep_iterate_signedPairedShift_commutes τ Ψ₁ Ψ₂ hΨ₁ hΨ₂ k u (-c)
  simpa [sub_eq_add_neg] using h

/--
The gap between shifted and unshifted `k`-iterates is exactly a constant function.

This is a convenient normalized form for downstream quotient/seminorm estimates.
-/
theorem sweep_iterate_shift_difference_eq_const
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ)
    (u : ι₁ → ℝ) (c : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k]) (fun i => u i + c) - ((sweep Ψ₁ Ψ₂)^[k]) u = fun _ => c := by
  funext i
  have hi :
      ((sweep Ψ₁ Ψ₂)^[k]) (fun j => u j + c) i = ((sweep Ψ₁ Ψ₂)^[k]) u i + c := by
    simpa using congrArg (fun f => f i)
      (sweep_iterate_signedPairedShift_commutes τ Ψ₁ Ψ₂ hΨ₁ hΨ₂ k u c)
  simp [Pi.sub_apply, hi]

/--
The sweep commutes with subtraction of a constant shift.
-/
theorem sweep_translationEquivariant_neg
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (u : ι₁ → ℝ) (c : ℝ) :
    sweep Ψ₁ Ψ₂ (fun i => u i - c) = fun i => sweep Ψ₁ Ψ₂ u i - c := by
  have h := sweep_translationEquivariant τ Ψ₁ Ψ₂ hΨ₁ hΨ₂ u (-c)
  simp only [sub_eq_add_neg] at h ⊢
  exact h

end Setup
end KLProjection
end FlowSinkhorn
