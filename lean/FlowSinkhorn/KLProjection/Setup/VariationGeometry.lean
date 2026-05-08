import FlowSinkhorn.KLProjection.Variation
import FlowSinkhorn.KLProjection.BlockQuotient
import FlowSinkhorn.KLProjection.Topical

/-!
# Variation geometry and quotient seminorms

This module is the Lean-side home for the geometric setup used in
`papers/kl-projections/sections/sec-nonexpansiveness.tex` and
`papers/kl-projections/sections/sec-dual-convergence.tex`.

Paper targets:
- Equation `eq:variation-seminorm`;
- Proposition `prop:topical-nonexpansive`;
- Definition `def:block-seminorm` and the quotient norm `eq:quotient-norm`.

Intended theorem/definition names:
- `variationSeminorm`, `variationSeminorm_add_const`;
- `variationSeminorm_nonexpansive_of_topical`;
- `blockQuotientSeminorm` (future generic version);
- `signedBlockQuotientSeminorm` (future paired-balance version).

Current status:
- the finite-dimensional variation machinery already lives in
  `FlowSinkhorn.KLProjection.Variation`;
- a first paired two-block quotient prototype already lives in
  `FlowSinkhorn.KLProjection.BlockQuotient`.

Future role:
this file should become the canonical import for every result that only depends on quotient
geometry and variation-seminorm non-expansiveness, before any application-specific formula enters.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace Setup

/-! ## Quotient geometry under translation shifts -/

/--
Translation equivariance lets one factor out a common constant shift before taking differences.

This is a convenient quotient-level rewrite for later dual/primal estimates:
the action on a pair of inputs depends only on their difference.
-/
theorem translationEquivariant_sub_shift
    {ι : Type*}
    (T : (ι → ℝ) → (ι → ℝ))
    (htrans : TranslationEquivariant T)
    (x y : ι → ℝ) (c : ℝ) :
    T (fun i => x i + c) - T (fun i => y i + c) = T x - T y := by
  funext i
  simp [htrans x c, htrans y c]

/--
Two independent translation shifts on the inputs expand to a constant-offset output difference.
-/
theorem translationEquivariant_sub_two_shifts
    {ι : Type*}
    (T : (ι → ℝ) → (ι → ℝ))
    (htrans : TranslationEquivariant T)
    (x y : ι → ℝ) (c₁ c₂ : ℝ) :
    T (fun i => x i + c₁) - T (fun i => y i + c₂) =
      fun i => (T x - T y) i + (c₁ - c₂) := by
  funext i
  have hx : T (fun j => x j + c₁) i = T x i + c₁ := by
    simpa using congrArg (fun f => f i) (htrans x c₁)
  have hy : T (fun j => y j + c₂) i = T y i + c₂ := by
    simpa using congrArg (fun f => f i) (htrans y c₂)
  calc
    (T (fun j => x j + c₁) - T (fun j => y j + c₂)) i
        = (T x i + c₁) - (T y i + c₂) := by simp [Pi.sub_apply, hx, hy]
    _ = (T x i - T y i) + (c₁ - c₂) := by ring
    _ = (T x - T y) i + (c₁ - c₂) := by rfl

/--
Topical maps remain non-expansive after a common translation of both inputs.

This is the form most useful when a downstream argument first normalizes iterates by a
common gauge shift and only then compares them through the variation seminorm.
-/
theorem variationSeminorm_nonexpansive_of_topical_shifted
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (T : (ι → ℝ) → (ι → ℝ))
    (hmono : Monotone T)
    (htrans : TranslationEquivariant T)
    (x y : ι → ℝ) (c : ℝ) :
    variationSeminorm (T (fun i => x i + c) - T (fun i => y i + c)) ≤
      variationSeminorm (x - y) := by
  simpa [translationEquivariant_sub_shift (T := T) htrans x y c] using
    variationSeminorm_nonexpansive_of_topical T hmono htrans x y

/-! ## Block-quotient seminorm -/

/--
Block-quotient seminorm for a single finite-dimensional real block.

Paper role: Definition `def:block-seminorm` (first block, constant-shift quotient direction).
This is the infimum of `‖x + c · 1‖_∞` over all scalar shifts `c`, which equals the variation
seminorm.
-/
noncomputable def blockQuotientSeminorm {ι : Type*} [Fintype ι] [Nonempty ι] (x : ι → ℝ) : ℝ :=
  sInf (Set.range fun c : ℝ => coordSupNorm (fun i => x i + c))

/--
The block-quotient seminorm equals the variation seminorm for single-block constant-shift quotients.

This confirms that quotienting by constant shifts in the ℓ∞ sense exactly recovers the
variation seminorm.
-/
theorem blockQuotientSeminorm_eq_variationSeminorm
    {ι : Type*} [Fintype ι] [Nonempty ι] (x : ι → ℝ) :
    blockQuotientSeminorm x = variationSeminorm x := by
  unfold blockQuotientSeminorm
  have hbdd : BddBelow (Set.range fun c : ℝ => coordSupNorm (fun i => x i + c)) :=
    ⟨0, by rintro r ⟨c, rfl⟩; exact coordSupNorm_nonneg _⟩
  apply le_antisymm
  · -- sInf ≤ variationSeminorm x: the centering shift achieves ≤ variationSeminorm x
    apply csInf_le_of_le hbdd ⟨centeringShift x, rfl⟩
    -- coordSupNorm (x + centeringShift x) ≤ variationSeminorm x
    unfold coordSupNorm
    apply Finset.sup'_le
    intro i _
    exact abs_add_centeringShift_le_variationSeminorm x i
  · -- variationSeminorm x ≤ sInf: it is a lower bound for all c
    apply le_csInf (Set.range_nonempty _)
    rintro r ⟨c, rfl⟩
    apply variationSeminorm_le_of_forall_abs_add_const_le x (c := c)
    intro i
    exact abs_le_coordSupNorm (fun i => x i + c) i

/-- blockQuotientSeminorm is nonneg. -/
theorem blockQuotientSeminorm_nonneg
    {ι : Type*} [Fintype ι] [Nonempty ι] (x : ι → ℝ) :
    0 ≤ blockQuotientSeminorm x := by
  rw [blockQuotientSeminorm_eq_variationSeminorm]
  exact variationSeminorm_nonneg x

/-- blockQuotientSeminorm is invariant under constant shifts. -/
theorem blockQuotientSeminorm_add_const
    {ι : Type*} [Fintype ι] [Nonempty ι] (x : ι → ℝ) (d : ℝ) :
    blockQuotientSeminorm (fun i => x i + d) = blockQuotientSeminorm x := by
  simp only [blockQuotientSeminorm_eq_variationSeminorm]
  exact variationSeminorm_add_const x d

/-! ## Mathlib Seminorm instance for block-quotient -/

/--
`blockQuotientSeminorm` as a Mathlib `Seminorm ℝ (ι → ℝ)`.

This registers Definition `def:block-seminorm` as a genuine Mathlib seminorm instance.
Since `blockQuotientSeminorm = variationSeminorm` (proved in
`blockQuotientSeminorm_eq_variationSeminorm`), we transport the Seminorm structure
from `variationSeminormAsSeminorm` via `Seminorm.of`.
-/
noncomputable def blockQuotientSeminormAsSeminorm
    {ι : Type*} [Fintype ι] [Nonempty ι] : Seminorm ℝ (ι → ℝ) :=
  Seminorm.of blockQuotientSeminorm
    (fun x y => by
      rw [blockQuotientSeminorm_eq_variationSeminorm,
          blockQuotientSeminorm_eq_variationSeminorm,
          blockQuotientSeminorm_eq_variationSeminorm]
      exact variationSeminorm_add x y)
    (fun a x => by
      rw [blockQuotientSeminorm_eq_variationSeminorm,
          blockQuotientSeminorm_eq_variationSeminorm]
      rw [variationSeminorm_smul]
      simp [Real.norm_eq_abs])

/-! ## Utility lemmas -/

/--
The variation seminorm is bounded above by the coordinatewise sup-norm.

Since every coordinate satisfies `|x i| ≤ coordSupNorm x`, the centering shift `c = 0`
already witnesses the infimum: `coordSupNorm x` is itself a valid radius for the unshifted
vector, and `variationSeminorm` is the infimum of all such radii over shifts.
-/
theorem variationSeminorm_le_coordSupNorm
    {ι : Type*} [Fintype ι] [Nonempty ι] (x : ι → ℝ) :
    variationSeminorm x ≤ coordSupNorm x :=
  variationSeminorm_le_of_forall_abs_add_const_le x (c := 0) (by simp [abs_le_coordSupNorm x])

/-- The block-quotient seminorm is sub-commutative: `bqS(x - y) = bqS(y - x)`. -/
theorem blockQuotientSeminorm_sub_comm
    {ι : Type*} [Fintype ι] [Nonempty ι] (x y : ι → ℝ) :
    blockQuotientSeminorm (x - y) = blockQuotientSeminorm (y - x) := by
  simp only [blockQuotientSeminorm_eq_variationSeminorm]
  exact variationSeminorm_sub_comm x y

/-- The block-quotient seminorm of zero is zero. -/
theorem blockQuotientSeminorm_zero
    {ι : Type*} [Fintype ι] [Nonempty ι] :
    blockQuotientSeminorm (0 : ι → ℝ) = 0 := by
  simp only [blockQuotientSeminorm_eq_variationSeminorm]
  exact variationSeminorm_zero

/-- The block-quotient seminorm of a negation: `bqS(-x) = bqS(x)`. -/
theorem blockQuotientSeminorm_neg
    {ι : Type*} [Fintype ι] [Nonempty ι] (x : ι → ℝ) :
    blockQuotientSeminorm (-x) = blockQuotientSeminorm x := by
  simp only [blockQuotientSeminorm_eq_variationSeminorm]
  exact variationSeminorm_neg x

/--
The block-quotient seminorm satisfies the sub-additivity axiom directly.

This is the triangle inequality transported from `variationSeminorm_add`.
-/
theorem blockQuotientSeminorm_add_le
    {ι : Type*} [Fintype ι] [Nonempty ι] (x y : ι → ℝ) :
    blockQuotientSeminorm (x + y) ≤ blockQuotientSeminorm x + blockQuotientSeminorm y := by
  simp only [blockQuotientSeminorm_eq_variationSeminorm]
  exact variationSeminorm_add x y

/--
Positive homogeneity of the block-quotient seminorm.
-/
theorem blockQuotientSeminorm_smul
    {ι : Type*} [Fintype ι] [Nonempty ι] (a : ℝ) (x : ι → ℝ) :
    blockQuotientSeminorm (a • x) = |a| * blockQuotientSeminorm x := by
  simp [blockQuotientSeminorm_eq_variationSeminorm, variationSeminorm_smul]

/--
Triangle inequality in difference form for the block-quotient seminorm.
-/
theorem blockQuotientSeminorm_sub_le_add
    {ι : Type*} [Fintype ι] [Nonempty ι] (x y z : ι → ℝ) :
    blockQuotientSeminorm (x - z) ≤
      blockQuotientSeminorm (x - y) + blockQuotientSeminorm (y - z) := by
  have hsum : (x - y) + (y - z) = x - z := by
    funext i
    change (x i - y i) + (y i - z i) = x i - z i
    ring
  calc
    blockQuotientSeminorm (x - z)
        = blockQuotientSeminorm ((x - y) + (y - z)) := by simp [hsum]
    _ ≤ blockQuotientSeminorm (x - y) + blockQuotientSeminorm (y - z) :=
      blockQuotientSeminorm_add_le (x - y) (y - z)

/--
The `blockQuotientSeminormAsSeminorm` equals `variationSeminormAsSeminorm` as Mathlib Seminorm
instances.

This is the canonical identity that lets applications work interchangeably with either name:
`blockQuotientSeminormAsSeminorm = variationSeminormAsSeminorm`.
-/
theorem blockQuotientSeminormAsSeminorm_eq_variationSeminormAsSeminorm
    {ι : Type*} [Fintype ι] [Nonempty ι] :
    @blockQuotientSeminormAsSeminorm ι _ _ = @variationSeminormAsSeminorm ι _ _ := by
  ext x
  simp only [blockQuotientSeminormAsSeminorm, variationSeminormAsSeminorm, Seminorm.of]
  exact blockQuotientSeminorm_eq_variationSeminorm x

/--
The variation seminorm at a zero input is zero.
This is a short paper-facing alias connecting `variationSeminorm_zero` to the concrete
`u₀ = 0` condition used in orbit bound hypotheses.
-/
theorem variationSeminorm_zero_fn
    {ι : Type*} [Fintype ι] [Nonempty ι] :
    variationSeminorm (fun _ : ι => (0 : ℝ)) = 0 :=
  variationSeminorm_zero

/--
Topical maps remain non-expansive after a common translation, using the bundled `IsTopical`
predicate.

This is the `IsTopical`-facing variant of `variationSeminorm_nonexpansive_of_topical_shifted`.
-/
theorem variationSeminorm_nonexpansive_of_isTopical_shifted
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (T : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical T)
    (x y : ι → ℝ) (c : ℝ) :
    variationSeminorm (T (fun i => x i + c) - T (fun i => y i + c)) ≤
      variationSeminorm (x - y) :=
  variationSeminorm_nonexpansive_of_topical_shifted T hT.mono hT.trans x y c

/--
Topical non-expansiveness with two independent gauge shifts on the inputs.

Applications often compare representatives normalized with different constants; this
statement removes that asymmetry at the variation-seminorm level.
-/
theorem variationSeminorm_nonexpansive_of_isTopical_shifted_two_consts
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (T : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical T)
    (x y : ι → ℝ) (c₁ c₂ : ℝ) :
    variationSeminorm (T (fun i => x i + c₁) - T (fun i => y i + c₂)) ≤
      variationSeminorm (x - y) := by
  have hbase :
      variationSeminorm (T x - T y) ≤ variationSeminorm (x - y) :=
    variationSeminorm_nonexpansive_of_topical T hT.mono hT.trans x y
  have hshift :
      T (fun i => x i + c₁) - T (fun i => y i + c₂) =
        fun i => (T x - T y) i + (c₁ - c₂) := by
    funext i
    have hx : T (fun j => x j + c₁) i = T x i + c₁ := by
      simpa using congrArg (fun f => f i) (hT.trans x c₁)
    have hy : T (fun j => y j + c₂) i = T y i + c₂ := by
      simpa using congrArg (fun f => f i) (hT.trans y c₂)
    calc
      (T (fun j => x j + c₁) - T (fun j => y j + c₂)) i
          = (T x i + c₁) - (T y i + c₂) := by simp [Pi.sub_apply, hx, hy]
      _ = (T x i - T y i) + (c₁ - c₂) := by ring
      _ = (T x - T y) i + (c₁ - c₂) := by rfl
  calc
    variationSeminorm (T (fun i => x i + c₁) - T (fun i => y i + c₂))
        = variationSeminorm (fun i => (T x - T y) i + (c₁ - c₂)) := by
          simp [hshift]
    _ = variationSeminorm (T x - T y) := by
          simpa using variationSeminorm_add_const (T x - T y) (c₁ - c₂)
    _ ≤ variationSeminorm (x - y) := hbase

/--
Shifted non-expansiveness for iterates of a topical map.
-/
theorem variationSeminorm_nonexpansive_iterate_of_isTopical_shifted
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    (k : ℕ) (x y : ι → ℝ) (c : ℝ) :
    variationSeminorm ((T^[k]) (fun i => x i + c) - (T^[k]) (fun i => y i + c)) ≤
      variationSeminorm (x - y) :=
  variationSeminorm_nonexpansive_of_isTopical_shifted (T := T^[k]) (isTopical_iterate hT k) x y c

/--
Common-shift translation rewrite for iterates of a topical map.
-/
theorem translationEquivariant_sub_shift_iterate_of_isTopical
    {ι : Type*}
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    (k : ℕ) (x y : ι → ℝ) (c : ℝ) :
    (T^[k]) (fun i => x i + c) - (T^[k]) (fun i => y i + c) =
      (T^[k]) x - (T^[k]) y :=
  translationEquivariant_sub_shift (T := T^[k]) (isTopical_iterate hT k).trans x y c

/--
Two-shift translation rewrite for iterates of a topical map.
-/
theorem translationEquivariant_sub_two_shifts_iterate_of_isTopical
    {ι : Type*}
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    (k : ℕ) (x y : ι → ℝ) (c₁ c₂ : ℝ) :
    (T^[k]) (fun i => x i + c₁) - (T^[k]) (fun i => y i + c₂) =
      fun i => ((T^[k]) x - (T^[k]) y) i + (c₁ - c₂) :=
  translationEquivariant_sub_two_shifts
    (T := T^[k]) (isTopical_iterate hT k).trans x y c₁ c₂

/--
Successor-index common-shift non-expansiveness for iterates of a topical map.
-/
theorem variationSeminorm_nonexpansive_iterate_of_isTopical_shifted_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    (k : ℕ) (x y : ι → ℝ) (c : ℝ) :
    variationSeminorm ((T^[k + 1]) (fun i => x i + c) - (T^[k + 1]) (fun i => y i + c)) ≤
      variationSeminorm (x - y) := by
  simpa [Nat.succ_eq_add_one] using
    variationSeminorm_nonexpansive_iterate_of_isTopical_shifted (hT := hT) (k + 1) x y c

/--
`of_le_index` convenience wrapper for common-shift shifted non-expansiveness
of topical iterates.
-/
theorem variationSeminorm_nonexpansive_iterate_of_isTopical_shifted_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    (k n : ℕ) (_hk : k ≤ n) (x y : ι → ℝ) (c : ℝ) :
    variationSeminorm ((T^[n]) (fun i => x i + c) - (T^[n]) (fun i => y i + c)) ≤
      variationSeminorm (x - y) :=
  variationSeminorm_nonexpansive_iterate_of_isTopical_shifted (hT := hT) n x y c

/--
Two-constant shifted non-expansiveness for iterates of a topical map.
-/
theorem variationSeminorm_nonexpansive_iterate_of_isTopical_shifted_two_consts
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    (k : ℕ) (x y : ι → ℝ) (c₁ c₂ : ℝ) :
    variationSeminorm ((T^[k]) (fun i => x i + c₁) - (T^[k]) (fun i => y i + c₂)) ≤
      variationSeminorm (x - y) :=
  variationSeminorm_nonexpansive_of_isTopical_shifted_two_consts
    (T := T^[k]) (isTopical_iterate hT k) x y c₁ c₂

/--
Successor-index two-constant shifted non-expansiveness for iterates of a topical map.
-/
theorem variationSeminorm_nonexpansive_iterate_of_isTopical_shifted_two_consts_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    (k : ℕ) (x y : ι → ℝ) (c₁ c₂ : ℝ) :
    variationSeminorm ((T^[k + 1]) (fun i => x i + c₁) - (T^[k + 1]) (fun i => y i + c₂)) ≤
      variationSeminorm (x - y) := by
  simpa [Nat.succ_eq_add_one] using
    variationSeminorm_nonexpansive_iterate_of_isTopical_shifted_two_consts
      (hT := hT) (k + 1) x y c₁ c₂

/--
`of_le_index` convenience wrapper for two-constant shifted non-expansiveness
of topical iterates.
-/
theorem variationSeminorm_nonexpansive_iterate_of_isTopical_shifted_two_consts_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    (k n : ℕ) (_hk : k ≤ n) (x y : ι → ℝ) (c₁ c₂ : ℝ) :
    variationSeminorm ((T^[n]) (fun i => x i + c₁) - (T^[n]) (fun i => y i + c₂)) ≤
      variationSeminorm (x - y) :=
  variationSeminorm_nonexpansive_iterate_of_isTopical_shifted_two_consts
    (hT := hT) n x y c₁ c₂

/--
Block-quotient nonexpansiveness for topical maps.

This is the quotient-geometric restatement of topical nonexpansiveness:
if `T` is monotone and translation-equivariant, then it contracts the
block-quotient seminorm on differences.
-/
theorem blockQuotientSeminorm_nonexpansive_of_topical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (T : (ι → ℝ) → (ι → ℝ))
    (hmono : Monotone T)
    (htrans : TranslationEquivariant T)
    (x y : ι → ℝ) :
    blockQuotientSeminorm (T x - T y) ≤ blockQuotientSeminorm (x - y) := by
  simpa [blockQuotientSeminorm_eq_variationSeminorm] using
    variationSeminorm_nonexpansive_of_topical T hmono htrans x y

/--
Block-quotient nonexpansiveness in bundled `IsTopical` form.

This wrapper removes the need for applications to unpack monotonicity and
translation-equivariance explicitly.
-/
theorem blockQuotientSeminorm_nonexpansive_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (T : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical T)
    (x y : ι → ℝ) :
    blockQuotientSeminorm (T x - T y) ≤ blockQuotientSeminorm (x - y) :=
  blockQuotientSeminorm_nonexpansive_of_topical T hT.mono hT.trans x y

/--
Shifted-input block-quotient nonexpansiveness for topical maps.

Common gauge shifts can be removed before applying the quotient seminorm
contraction estimate.
-/
theorem blockQuotientSeminorm_nonexpansive_of_isTopical_shifted
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (T : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical T)
    (x y : ι → ℝ) (c : ℝ) :
    blockQuotientSeminorm (T (fun i => x i + c) - T (fun i => y i + c)) ≤
      blockQuotientSeminorm (x - y) := by
  simpa [blockQuotientSeminorm_eq_variationSeminorm] using
    variationSeminorm_nonexpansive_of_isTopical_shifted T hT x y c

/--
Block-quotient non-expansiveness with two independent gauge shifts.
-/
theorem blockQuotientSeminorm_nonexpansive_of_isTopical_shifted_two_consts
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (T : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical T)
    (x y : ι → ℝ) (c₁ c₂ : ℝ) :
    blockQuotientSeminorm (T (fun i => x i + c₁) - T (fun i => y i + c₂)) ≤
      blockQuotientSeminorm (x - y) := by
  simpa [blockQuotientSeminorm_eq_variationSeminorm] using
    variationSeminorm_nonexpansive_of_isTopical_shifted_two_consts T hT x y c₁ c₂

/--
Shifted block-quotient non-expansiveness for iterates of a topical map.
-/
theorem blockQuotientSeminorm_nonexpansive_iterate_of_isTopical_shifted_two_consts
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    (k : ℕ) (x y : ι → ℝ) (c₁ c₂ : ℝ) :
    blockQuotientSeminorm ((T^[k]) (fun i => x i + c₁) - (T^[k]) (fun i => y i + c₂)) ≤
      blockQuotientSeminorm (x - y) := by
  simpa [blockQuotientSeminorm_eq_variationSeminorm] using
    variationSeminorm_nonexpansive_of_isTopical_shifted_two_consts
      (T := T^[k]) (isTopical_iterate hT k) x y c₁ c₂

/--
`IsTopical` bridge to `SeminormNonexpansive` for `blockQuotientSeminormAsSeminorm`.

This lets downstream lemmas written for abstract seminorm nonexpansiveness
consume topical hypotheses directly in block-quotient form.
-/
theorem SeminormNonexpansive_blockQuotientSeminormAsSeminorm_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T) :
    SeminormNonexpansive blockQuotientSeminormAsSeminorm T := by
  intro x y
  change blockQuotientSeminorm (T x - T y) ≤ blockQuotientSeminorm (x - y)
  exact blockQuotientSeminorm_nonexpansive_of_isTopical T hT x y

/--
Iterates of an `IsTopical` map are nonexpansive for `blockQuotientSeminormAsSeminorm`.

This is the iterate-level bridge needed by downstream generic orbit lemmas phrased in
`SeminormNonexpansive`.
-/
theorem SeminormNonexpansive_blockQuotientSeminormAsSeminorm_iterate_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    (k : ℕ) :
    SeminormNonexpansive blockQuotientSeminormAsSeminorm (T^[k]) :=
  SeminormNonexpansive_iterate blockQuotientSeminormAsSeminorm T
    (SeminormNonexpansive_blockQuotientSeminormAsSeminorm_of_isTopical (hT := hT)) k

/--
Unshifted block-quotient nonexpansiveness for iterates of an `IsTopical` map.
-/
theorem blockQuotientSeminorm_nonexpansive_iterate_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    (k : ℕ) (x y : ι → ℝ) :
    blockQuotientSeminorm ((T^[k]) x - (T^[k]) y) ≤ blockQuotientSeminorm (x - y) := by
  simpa using
    (SeminormNonexpansive_blockQuotientSeminormAsSeminorm_iterate_of_isTopical
      (hT := hT) k) x y

/--
Common-shift block-quotient nonexpansiveness for iterates of an `IsTopical` map.
-/
theorem blockQuotientSeminorm_nonexpansive_iterate_of_isTopical_shifted
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    (k : ℕ) (x y : ι → ℝ) (c : ℝ) :
    blockQuotientSeminorm ((T^[k]) (fun i => x i + c) - (T^[k]) (fun i => y i + c)) ≤
      blockQuotientSeminorm (x - y) := by
  simpa [blockQuotientSeminorm_eq_variationSeminorm] using
    variationSeminorm_nonexpansive_iterate_of_isTopical_shifted (hT := hT) k x y c

/--
Successor-index common-shift block-quotient nonexpansiveness for iterates of an `IsTopical` map.
-/
theorem blockQuotientSeminorm_nonexpansive_iterate_of_isTopical_shifted_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    (k : ℕ) (x y : ι → ℝ) (c : ℝ) :
    blockQuotientSeminorm ((T^[k + 1]) (fun i => x i + c) - (T^[k + 1]) (fun i => y i + c)) ≤
      blockQuotientSeminorm (x - y) := by
  simpa [Nat.succ_eq_add_one] using
    blockQuotientSeminorm_nonexpansive_iterate_of_isTopical_shifted (hT := hT) (k + 1) x y c

/--
`of_le_index` convenience wrapper for two-constant shifted iterate nonexpansiveness
in block-quotient form.
-/
theorem blockQuotientSeminorm_nonexpansive_iterate_of_isTopical_shifted_two_consts_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    (k n : ℕ) (_hk : k ≤ n) (x y : ι → ℝ) (c₁ c₂ : ℝ) :
    blockQuotientSeminorm ((T^[n]) (fun i => x i + c₁) - (T^[n]) (fun i => y i + c₂)) ≤
      blockQuotientSeminorm (x - y) :=
  blockQuotientSeminorm_nonexpansive_iterate_of_isTopical_shifted_two_consts
    (hT := hT) n x y c₁ c₂

/--
Successor-index two-constant shifted iterate nonexpansiveness in block-quotient form.
-/
theorem blockQuotientSeminorm_nonexpansive_iterate_of_isTopical_shifted_two_consts_succ
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {T : (ι → ℝ) → (ι → ℝ)} (hT : IsTopical T)
    (k : ℕ) (x y : ι → ℝ) (c₁ c₂ : ℝ) :
    blockQuotientSeminorm ((T^[k + 1]) (fun i => x i + c₁) - (T^[k + 1]) (fun i => y i + c₂)) ≤
      blockQuotientSeminorm (x - y) := by
  simpa [Nat.succ_eq_add_one] using
    blockQuotientSeminorm_nonexpansive_iterate_of_isTopical_shifted_two_consts
      (hT := hT) (k + 1) x y c₁ c₂

end Setup
end KLProjection
end FlowSinkhorn
