import FlowSinkhorn.KLProjection.Variation
import FlowSinkhorn.KLProjection.BlockQuotient

noncomputable section

namespace FlowSinkhorn
namespace KLProjection

variable {ι₁ ι₂ : Type*}

/-- First block-translation equivariance relation from Proposition A.3. -/
def SignedBlockTranslationEquivariant1
    (τ : PairedSign) (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ)) : Prop :=
  ∀ u₂ : ι₂ → ℝ, ∀ c : ℝ,
    Ψ₁ (fun j => u₂ j + c) = fun i => Ψ₁ u₂ i + τ.toReal * c

/-- Second block-translation equivariance relation from Proposition A.3. -/
def SignedBlockTranslationEquivariant2
    (τ : PairedSign) (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ)) : Prop :=
  ∀ u₁ : ι₁ → ℝ, ∀ c : ℝ,
    Ψ₂ (fun i => u₁ i + c) = fun j => Ψ₂ u₁ j + τ.toReal * c

/-- Full sweep obtained by composing the two block maps. -/
def sweep (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ)) (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ)) :
    (ι₁ → ℝ) → (ι₁ → ℝ) :=
  fun u₁ => Ψ₁ (Ψ₂ u₁)

@[simp] lemma PairedSign.toReal_mul_self (τ : PairedSign) : τ.toReal * τ.toReal = 1 := by
  cases τ <;> norm_num [PairedSign.toReal]

/--
If both block updates satisfy the signed paired-balance translation rule, then the full sweep is
translation-equivariant. This certifies the final implication in Proposition A.3.
-/
theorem sweep_translationEquivariant_of_signedBlockTranslationEquivariant
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

theorem sweep_eq_comp (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ)) (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ)) :
    sweep Ψ₁ Ψ₂ = Ψ₁ ∘ Ψ₂ := rfl

theorem sweep_apply (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ)) (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ)) (u₁ : ι₁ → ℝ) :
    sweep Ψ₁ Ψ₂ u₁ = Ψ₁ (Ψ₂ u₁) := rfl

theorem sweep_fixedPoint_iff_block_fixedPoint
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ)) (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ)) (u₁ : ι₁ → ℝ) :
    sweep Ψ₁ Ψ₂ u₁ = u₁ ↔ Ψ₁ (Ψ₂ u₁) = u₁ :=
  Iff.rfl

/--
Translation-equivariance is preserved by iteration.
-/
theorem translationEquivariant_iterate
    {ι : Type*} [Nonempty ι]
    {T : (ι → ℝ) → (ι → ℝ)}
    (hT : TranslationEquivariant T) :
    ∀ k : ℕ, TranslationEquivariant (T^[k]) := by
  intro k
  induction k with
  | zero =>
      intro x c
      funext i
      simp
  | succ k ih =>
      intro x c
      calc
        (T^[k + 1]) (fun i => x i + c)
            = T ((T^[k]) (fun i => x i + c)) := by
                rw [Function.iterate_succ_apply']
        _ = T (fun i => (T^[k]) x i + c) := by
              rw [ih x c]
        _ = fun i => T ((T^[k]) x) i + c := hT ((T^[k]) x) c
        _ = fun i => (T^[k + 1]) x i + c := by
              rw [Function.iterate_succ_apply']

/--
Explicit add-constant rule for iterates of a translation-equivariant map.
-/
theorem translationEquivariant_iterate_add_const
    {ι : Type*} [Nonempty ι]
    {T : (ι → ℝ) → (ι → ℝ)}
    (hT : TranslationEquivariant T)
    (k : ℕ) (x : ι → ℝ) (c : ℝ) :
    (T^[k]) (fun i => x i + c) = fun i => (T^[k]) x i + c :=
  translationEquivariant_iterate hT k x c

/--
Common-shift difference invariance for iterates of a translation-equivariant map.
-/
theorem translationEquivariant_iterate_sub_eq_of_add_const
    {ι : Type*} [Nonempty ι]
    {T : (ι → ℝ) → (ι → ℝ)}
    (hT : TranslationEquivariant T)
    (k : ℕ) (u v : ι → ℝ) (c : ℝ) :
    (T^[k]) (fun i => u i + c) - (T^[k]) (fun i => v i + c) =
      (T^[k]) u - (T^[k]) v := by
  ext i
  have hu :
      (T^[k]) (fun j => u j + c) i = (T^[k]) u i + c := by
    simpa using congrArg (fun f : ι → ℝ => f i)
      (translationEquivariant_iterate_add_const hT k u c)
  have hv :
      (T^[k]) (fun j => v j + c) i = (T^[k]) v i + c := by
    simpa using congrArg (fun f : ι → ℝ => f i)
      (translationEquivariant_iterate_add_const hT k v c)
  simp [hu, hv]

/--
Two-shift base-form bridge for iterates of a translation-equivariant map.
-/
theorem translationEquivariant_iterate_sub_two_consts_eq_of_add_const
    {ι : Type*} [Nonempty ι]
    {T : (ι → ℝ) → (ι → ℝ)}
    (hT : TranslationEquivariant T)
    (k : ℕ) (u v : ι → ℝ) (c₁ c₂ : ℝ) :
    (T^[k]) (fun i => u i + c₁) - (T^[k]) (fun i => v i + c₂) =
      fun i => ((T^[k]) u - (T^[k]) v) i + (c₁ - c₂) := by
  ext i
  have hu :
      (T^[k]) (fun j => u j + c₁) i = (T^[k]) u i + c₁ := by
    simpa using congrArg (fun f : ι → ℝ => f i)
      (translationEquivariant_iterate_add_const hT k u c₁)
  have hv :
      (T^[k]) (fun j => v j + c₂) i = (T^[k]) v i + c₂ := by
    simpa using congrArg (fun f : ι → ℝ => f i)
      (translationEquivariant_iterate_add_const hT k v c₂)
  calc
    ((T^[k]) (fun j => u j + c₁) - (T^[k]) (fun j => v j + c₂)) i
        = ((T^[k]) u i + c₁) - ((T^[k]) v i + c₂) := by
            simp [Pi.sub_apply, hu, hv]
    _ = ((T^[k]) u i - (T^[k]) v i) + (c₁ - c₂) := by ring
    _ = ((T^[k]) u - (T^[k]) v) i + (c₁ - c₂) := by simp [Pi.sub_apply]

/--
Shift-gap constant form for iterates of a translation-equivariant map.
-/
theorem translationEquivariant_iterate_shift_difference_eq_const
    {ι : Type*} [Nonempty ι]
    {T : (ι → ℝ) → (ι → ℝ)}
    (hT : TranslationEquivariant T)
    (k : ℕ) (u : ι → ℝ) (c : ℝ) :
    (T^[k]) (fun i => u i + c) - (T^[k]) u = fun _ => c := by
  ext i
  have hu :
      (T^[k]) (fun j => u j + c) i = (T^[k]) u i + c := by
    simpa using congrArg (fun f : ι → ℝ => f i)
      (translationEquivariant_iterate_add_const hT k u c)
  simp [Pi.sub_apply, hu]

/--
Two-shift constant-gap form for iterates of a translation-equivariant map.
-/
theorem translationEquivariant_iterate_shift_difference_two_consts_eq_const
    {ι : Type*} [Nonempty ι]
    {T : (ι → ℝ) → (ι → ℝ)}
    (hT : TranslationEquivariant T)
    (k : ℕ) (u : ι → ℝ) (c₁ c₂ : ℝ) :
    (T^[k]) (fun i => u i + c₁) - (T^[k]) (fun i => u i + c₂) = fun _ => (c₁ - c₂) := by
  ext i
  have hu₁ :
      (T^[k]) (fun j => u j + c₁) i = (T^[k]) u i + c₁ := by
    simpa using congrArg (fun f : ι → ℝ => f i)
      (translationEquivariant_iterate_add_const hT k u c₁)
  have hu₂ :
      (T^[k]) (fun j => u j + c₂) i = (T^[k]) u i + c₂ := by
    simpa using congrArg (fun f : ι → ℝ => f i)
      (translationEquivariant_iterate_add_const hT k u c₂)
  calc
    ((T^[k]) (fun j => u j + c₁) - (T^[k]) (fun j => u j + c₂)) i
        = ((T^[k]) u i + c₁) - ((T^[k]) u i + c₂) := by
            simp [Pi.sub_apply, hu₁, hu₂]
    _ = c₁ - c₂ := by ring

/--
Zero-input specialization of the two-shift constant-gap iterate bridge.
-/
theorem translationEquivariant_iterate_constInput_difference_eq_const
    {ι : Type*} [Nonempty ι]
    {T : (ι → ℝ) → (ι → ℝ)}
    (hT : TranslationEquivariant T)
    (k : ℕ) (c₁ c₂ : ℝ) :
    (T^[k]) (fun _ : ι => c₁) - (T^[k]) (fun _ : ι => c₂) = fun _ => (c₁ - c₂) := by
  simpa using
    translationEquivariant_iterate_shift_difference_two_consts_eq_const
      (hT := hT) k (fun _ : ι => (0 : ℝ)) c₁ c₂

/--
Every iterate of the sweep is translation-equivariant under the signed block assumptions.
-/
theorem sweep_iterate_translationEquivariant_of_signedBlockTranslationEquivariant
    [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ) :
    TranslationEquivariant ((sweep Ψ₁ Ψ₂)^[k]) := by
  exact translationEquivariant_iterate
    (sweep_translationEquivariant_of_signedBlockTranslationEquivariant τ Ψ₁ Ψ₂ hΨ₁ hΨ₂) k

/--
Explicit add-constant rule for sweep iterates.
-/
theorem sweep_iterate_add_const_of_signedBlockTranslationEquivariant
    [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ) (u₁ : ι₁ → ℝ) (c : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k]) (fun i => u₁ i + c) =
      fun i => ((sweep Ψ₁ Ψ₂)^[k]) u₁ i + c :=
  sweep_iterate_translationEquivariant_of_signedBlockTranslationEquivariant
    τ Ψ₁ Ψ₂ hΨ₁ hΨ₂ k u₁ c

/--
Difference of two shifted inputs is invariant under sweep iterates.
-/
theorem sweep_iterate_sub_eq_of_add_const_of_signedBlockTranslationEquivariant
    [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ) (u v : ι₁ → ℝ) (c : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k]) (fun i => u i + c) -
        ((sweep Ψ₁ Ψ₂)^[k]) (fun i => v i + c) =
      ((sweep Ψ₁ Ψ₂)^[k]) u - ((sweep Ψ₁ Ψ₂)^[k]) v :=
  translationEquivariant_iterate_sub_eq_of_add_const
    (hT := sweep_translationEquivariant_of_signedBlockTranslationEquivariant
      τ Ψ₁ Ψ₂ hΨ₁ hΨ₂)
    k u v c

/--
Two-constant shifted-difference base form for sweep iterates under signed block conditions.
-/
theorem sweep_iterate_sub_two_consts_eq_of_add_const_of_signedBlockTranslationEquivariant
    [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ) (u v : ι₁ → ℝ) (c₁ c₂ : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k]) (fun i => u i + c₁) -
        ((sweep Ψ₁ Ψ₂)^[k]) (fun i => v i + c₂) =
      fun i => (((sweep Ψ₁ Ψ₂)^[k]) u - ((sweep Ψ₁ Ψ₂)^[k]) v) i + (c₁ - c₂) :=
  translationEquivariant_iterate_sub_two_consts_eq_of_add_const
    (hT := sweep_translationEquivariant_of_signedBlockTranslationEquivariant
      τ Ψ₁ Ψ₂ hΨ₁ hΨ₂)
    k u v c₁ c₂

/--
Two-constant shift-gap constant form for sweep iterates under signed block conditions.
-/
theorem sweep_iterate_shift_difference_two_consts_eq_const_of_signedBlockTranslationEquivariant
    [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ) (u : ι₁ → ℝ) (c₁ c₂ : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k]) (fun i => u i + c₁) -
        ((sweep Ψ₁ Ψ₂)^[k]) (fun i => u i + c₂) =
      fun _ => (c₁ - c₂) :=
  translationEquivariant_iterate_shift_difference_two_consts_eq_const
    (hT := sweep_translationEquivariant_of_signedBlockTranslationEquivariant
      τ Ψ₁ Ψ₂ hΨ₁ hΨ₂)
    k u c₁ c₂

/--
Iterates of `sweep Ψ₁ Ψ₂` coincide with iterates of the explicit composition `Ψ₁ ∘ Ψ₂`.
-/
theorem sweep_iterate_eq_comp_iterate
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (k : ℕ) :
    (sweep Ψ₁ Ψ₂)^[k] = (Ψ₁ ∘ Ψ₂)^[k] := by
  simp [sweep_eq_comp]

/--
Successor-index apply form for sweep iterates.
-/
theorem sweep_iterate_succ_apply
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (k : ℕ) (u : ι₁ → ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k + 1]) u =
      sweep Ψ₁ Ψ₂ (((sweep Ψ₁ Ψ₂)^[k]) u) := by
  simpa [Nat.succ_eq_add_one] using
    (Function.iterate_succ_apply' (f := sweep Ψ₁ Ψ₂) k u)

/--
Single-shift constant-gap form for sweep iterates under signed block conditions.
-/
theorem sweep_iterate_shift_difference_eq_const_of_signedBlockTranslationEquivariant
    [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ) (u : ι₁ → ℝ) (c : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k]) (fun i => u i + c) -
        ((sweep Ψ₁ Ψ₂)^[k]) u =
      fun _ => c :=
  translationEquivariant_iterate_shift_difference_eq_const
    (hT := sweep_translationEquivariant_of_signedBlockTranslationEquivariant
      τ Ψ₁ Ψ₂ hΨ₁ hΨ₂)
    k u c

/--
Constant-input specialization for sweep iterates under signed block conditions.
-/
theorem sweep_iterate_constInput_difference_eq_const_of_signedBlockTranslationEquivariant
    [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ) (c₁ c₂ : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k]) (fun _ : ι₁ => c₁) -
        ((sweep Ψ₁ Ψ₂)^[k]) (fun _ : ι₁ => c₂) =
      fun _ => (c₁ - c₂) :=
  translationEquivariant_iterate_constInput_difference_eq_const
    (hT := sweep_translationEquivariant_of_signedBlockTranslationEquivariant
      τ Ψ₁ Ψ₂ hΨ₁ hΨ₂)
    k c₁ c₂

/--
Successor-index constant-input specialization for sweep iterates
under signed block conditions.
-/
theorem sweep_iterate_constInput_difference_eq_const_succ_of_signedBlockTranslationEquivariant
    [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ) (c₁ c₂ : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k + 1]) (fun _ : ι₁ => c₁) -
        ((sweep Ψ₁ Ψ₂)^[k + 1]) (fun _ : ι₁ => c₂) =
      fun _ => (c₁ - c₂) := by
  exact sweep_iterate_constInput_difference_eq_const_of_signedBlockTranslationEquivariant
    (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂) hΨ₁ hΨ₂ (k + 1) c₁ c₂

/--
Constant-input difference invariance for iterates of the explicit composition `Ψ₁ ∘ Ψ₂`.

This is the composition-form bridge for
`sweep_iterate_constInput_difference_eq_const_of_signedBlockTranslationEquivariant`.
-/
theorem sweep_comp_iterate_constInput_difference_eq_const_of_signedBlockTranslationEquivariant
    [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ) (c₁ c₂ : ℝ) :
    ((Ψ₁ ∘ Ψ₂)^[k]) (fun _ : ι₁ => c₁) -
        ((Ψ₁ ∘ Ψ₂)^[k]) (fun _ : ι₁ => c₂) =
      fun _ => (c₁ - c₂) := by
  simpa [sweep_iterate_eq_comp_iterate Ψ₁ Ψ₂ k] using
    (sweep_iterate_constInput_difference_eq_const_of_signedBlockTranslationEquivariant
      (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂) hΨ₁ hΨ₂ k c₁ c₂)

/--
Common-shift difference invariance for iterates of the explicit composition `Ψ₁ ∘ Ψ₂`.

This is a bridge form of
`sweep_iterate_sub_eq_of_add_const_of_signedBlockTranslationEquivariant`
rewritten through `sweep_iterate_eq_comp_iterate`.
-/
theorem sweep_comp_iterate_sub_eq_of_add_const_of_signedBlockTranslationEquivariant
    [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ) (u v : ι₁ → ℝ) (c : ℝ) :
    ((Ψ₁ ∘ Ψ₂)^[k]) (fun i => u i + c) -
        ((Ψ₁ ∘ Ψ₂)^[k]) (fun i => v i + c) =
      ((Ψ₁ ∘ Ψ₂)^[k]) u - ((Ψ₁ ∘ Ψ₂)^[k]) v := by
  simpa [sweep_iterate_eq_comp_iterate Ψ₁ Ψ₂ k] using
    (sweep_iterate_sub_eq_of_add_const_of_signedBlockTranslationEquivariant
      (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂) hΨ₁ hΨ₂ k u v c)

/--
Successor-index version of
`sweep_comp_iterate_sub_eq_of_add_const_of_signedBlockTranslationEquivariant`.
-/
theorem sweep_comp_iterate_sub_eq_of_add_const_succ_of_signedBlockTranslationEquivariant
    [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ) (u v : ι₁ → ℝ) (c : ℝ) :
    ((Ψ₁ ∘ Ψ₂)^[k + 1]) (fun i => u i + c) -
        ((Ψ₁ ∘ Ψ₂)^[k + 1]) (fun i => v i + c) =
      ((Ψ₁ ∘ Ψ₂)^[k + 1]) u - ((Ψ₁ ∘ Ψ₂)^[k + 1]) v := by
  exact sweep_comp_iterate_sub_eq_of_add_const_of_signedBlockTranslationEquivariant
    (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂) hΨ₁ hΨ₂ (k + 1) u v c

/--
Index-comparison convenience wrapper for constant-input differences of sweep iterates.

If `m ≤ n`, this restates the constant-input difference identity at index `n`.
-/
theorem sweep_iterate_constInput_difference_eq_const_of_le_index_of_signedBlockTranslationEquivariant
    [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (m n : ℕ) (hmn : m ≤ n) (c₁ c₂ : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[n]) (fun _ : ι₁ => c₁) -
        ((sweep Ψ₁ Ψ₂)^[n]) (fun _ : ι₁ => c₂) =
      fun _ => (c₁ - c₂) := by
  rcases Nat.exists_eq_add_of_le hmn with ⟨t, rfl⟩
  simpa using
    (sweep_iterate_constInput_difference_eq_const_of_signedBlockTranslationEquivariant
      (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂) hΨ₁ hΨ₂ (m + t) c₁ c₂)

/--
Index-comparison convenience wrapper for constant-input differences
of composition iterates `Ψ₁ ∘ Ψ₂` under signed block conditions.
-/
theorem
    sweep_comp_iterate_constInput_difference_eq_const_of_le_index_of_signedBlockTranslationEquivariant
    [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (m n : ℕ) (hmn : m ≤ n) (c₁ c₂ : ℝ) :
    ((Ψ₁ ∘ Ψ₂)^[n]) (fun _ : ι₁ => c₁) -
        ((Ψ₁ ∘ Ψ₂)^[n]) (fun _ : ι₁ => c₂) =
      fun _ => (c₁ - c₂) := by
  rcases Nat.exists_eq_add_of_le hmn with ⟨t, rfl⟩
  simpa using
    (sweep_comp_iterate_constInput_difference_eq_const_of_signedBlockTranslationEquivariant
      (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂) hΨ₁ hΨ₂ (m + t) c₁ c₂)

/--
Successor-index convenience wrapper for common-shift difference invariance
of sweep iterates under signed block conditions.
-/
theorem sweep_iterate_sub_eq_of_add_const_succ_of_signedBlockTranslationEquivariant
    [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (k : ℕ) (u v : ι₁ → ℝ) (c : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[k + 1]) (fun i => u i + c) -
        ((sweep Ψ₁ Ψ₂)^[k + 1]) (fun i => v i + c) =
      ((sweep Ψ₁ Ψ₂)^[k + 1]) u - ((sweep Ψ₁ Ψ₂)^[k + 1]) v := by
  exact sweep_iterate_sub_eq_of_add_const_of_signedBlockTranslationEquivariant
    (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂) hΨ₁ hΨ₂ (k + 1) u v c

/--
Index-comparison convenience wrapper for common-shift difference invariance
of sweep iterates under signed block conditions.
-/
theorem sweep_iterate_sub_eq_of_add_const_of_le_index_of_signedBlockTranslationEquivariant
    [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (m n : ℕ) (hmn : m ≤ n) (u v : ι₁ → ℝ) (c : ℝ) :
    ((sweep Ψ₁ Ψ₂)^[n]) (fun i => u i + c) -
        ((sweep Ψ₁ Ψ₂)^[n]) (fun i => v i + c) =
      ((sweep Ψ₁ Ψ₂)^[n]) u - ((sweep Ψ₁ Ψ₂)^[n]) v := by
  rcases Nat.exists_eq_add_of_le hmn with ⟨t, rfl⟩
  simpa using
    (sweep_iterate_sub_eq_of_add_const_of_signedBlockTranslationEquivariant
      (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂) hΨ₁ hΨ₂ (m + t) u v c)

/--
Index-comparison convenience wrapper for common-shift difference invariance
of composition iterates `Ψ₁ ∘ Ψ₂` under signed block conditions.
-/
theorem sweep_comp_iterate_sub_eq_of_add_const_of_le_index_of_signedBlockTranslationEquivariant
    [Nonempty ι₁]
    (τ : PairedSign)
    (Ψ₁ : (ι₂ → ℝ) → (ι₁ → ℝ))
    (Ψ₂ : (ι₁ → ℝ) → (ι₂ → ℝ))
    (hΨ₁ : SignedBlockTranslationEquivariant1 τ Ψ₁)
    (hΨ₂ : SignedBlockTranslationEquivariant2 τ Ψ₂)
    (m n : ℕ) (hmn : m ≤ n) (u v : ι₁ → ℝ) (c : ℝ) :
    ((Ψ₁ ∘ Ψ₂)^[n]) (fun i => u i + c) -
        ((Ψ₁ ∘ Ψ₂)^[n]) (fun i => v i + c) =
      ((Ψ₁ ∘ Ψ₂)^[n]) u - ((Ψ₁ ∘ Ψ₂)^[n]) v := by
  rcases Nat.exists_eq_add_of_le hmn with ⟨t, rfl⟩
  simpa using
    (sweep_comp_iterate_sub_eq_of_add_const_of_signedBlockTranslationEquivariant
      (τ := τ) (Ψ₁ := Ψ₁) (Ψ₂ := Ψ₂) hΨ₁ hΨ₂ (m + t) u v c)

end KLProjection
end FlowSinkhorn
