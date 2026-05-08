import FlowSinkhorn.KLProjection.Applications.OT.HGamma

/-!
# `κ` for balanced optimal transport

This module is reserved for Proposition `prop:kappa_OT` from
`papers/kl-projections/sections/sec-sinkhorn.tex`.

Intended theorem names:
- `ot_kappa_eq_one`;
- `ot_signedPair_split_bound`.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace Applications
namespace OT

/--
Compatibility bridge for callers that already have the scalar normalization `κ = 1`.

This is not the paper-facing certification of Proposition `prop:kappa_OT`.  The concrete proof is
`ot_kappa_eq_one` below, via `ot_kappa_one_concrete`, and the statement map points there.  Keep this
small bridge only for older application wrappers that work with an abstract scalar `kappa`.
-/
theorem ot_kappa_eq_one_of_assumption
    {kappa : ℝ}
    (hkappa : kappa = 1) :
    kappa = 1 :=
  hkappa

/--
Signed-pair split bound specialized to balanced OT normalization.

This wrapper exposes the import-level API expected by the paper section: once a split quantity is
bounded by `\kappa` and `\kappa = 1`, we get the normalized bound by transitivity.
-/
theorem ot_signedPair_split_bound
    {splitTerm kappa : ℝ}
    (hsplit : splitTerm ≤ kappa)
    (hkappa : kappa = 1) :
    splitTerm ≤ 1 := by
  simpa [hkappa] using hsplit

/--
Canonical `H_γ/κ` budget rewrite for balanced OT.
-/
theorem ot_hGammaKappaBudget_eq_of_kappa_eq_one
    {kappa cost gamma hGamma : ℝ}
    (hkappa : kappa = 1) :
    PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma = cost + gamma * hGamma := by
  simp [PrimalDualBounds.hGammaKappaBudget, hkappa]

/--
Scaled split control transferred to the canonical `H_γ/κ` budget under `κ = 1`.
-/
theorem ot_signedPair_scaled_bound_of_kappa_eq_one
    {splitTerm kappa cost gamma hGamma : ℝ}
    (hsplit : splitTerm ≤ kappa)
    (hkappa : kappa = 1)
    (hnonneg : 0 ≤ cost + gamma * hGamma) :
    splitTerm * (cost + gamma * hGamma) ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma := by
  have hs1 : splitTerm ≤ (1 : ℝ) :=
    ot_signedPair_split_bound hsplit hkappa
  have hscaled : splitTerm * (cost + gamma * hGamma) ≤ 1 * (cost + gamma * hGamma) :=
    mul_le_mul_of_nonneg_right hs1 hnonneg
  simpa [PrimalDualBounds.hGammaKappaBudget, hkappa] using hscaled

/--
`H_γ` upper bounds transfer to normalized budgets in balanced OT.
-/
theorem ot_hGammaBudget_le_of_hGamma_bound_and_kappa_eq_one
    {kappa cost gamma hGamma hGammaUpper : ℝ}
    (hgamma : 0 ≤ gamma)
    (hH : hGamma ≤ hGammaUpper)
    (hkappa : kappa = 1) :
    PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤
      cost + gamma * hGammaUpper := by
  have hkappa_nonneg : 0 ≤ kappa := by simp [hkappa]
  have hbudget :
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤
        PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGammaUpper :=
    ot_HGamma_bound hkappa_nonneg hgamma hH
  calc
    PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma
        ≤ PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGammaUpper := hbudget
    _ = cost + gamma * hGammaUpper := by
          simp [PrimalDualBounds.hGammaKappaBudget, hkappa]

/--
Key inequality behind Proposition `prop:kappa_OT`.

For any Y : ι₁ × ι₂ → ℝ representable as α_i + β_j, fixing any column j₀,
the difference α_i - α_{i'} can be read off as Y(i,j₀) - Y(i',j₀), which gives:

  |α_i - α_{i'}| ≤ 2 * sup_{p : ι₁ × ι₂} |Y p|

This is the estimate showing that κ ≤ 1 in the balanced OT setting.
-/
theorem ot_kappa_rowSpan_bound
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (alpha : ι₁ → ℝ) (beta : ι₂ → ℝ) (j₀ : ι₂)
    (Y : ι₁ × ι₂ → ℝ)
    (hY : ∀ i j, alpha i + beta j = Y (i, j)) :
    ∀ i i' : ι₁,
      |alpha i - alpha i'| ≤
        2 * Finset.univ.sup' Finset.univ_nonempty (fun p : ι₁ × ι₂ => |Y p|) := by
  intro i i'
  -- Step 1: rewrite difference using the representation
  have hrepr : alpha i - alpha i' = Y (i, j₀) - Y (i', j₀) := by
    have h1 := hY i j₀
    have h2 := hY i' j₀
    linarith
  -- Step 2: introduce the sup value for convenience
  set S := Finset.univ.sup' Finset.univ_nonempty (fun p : ι₁ × ι₂ => |Y p|) with hS_def
  -- Step 3: each |Y p| is bounded by S
  have hle : ∀ p : ι₁ × ι₂, |Y p| ≤ S := by
    intro p
    exact Finset.le_sup' (fun p : ι₁ × ι₂ => |Y p|) (Finset.mem_univ p)
  -- Step 4: combine using triangle inequality and the sup bound
  rw [hrepr]
  -- |a - b| ≤ |a| + |b| by triangle inequality applied to a + (-b)
  have htri : |Y (i, j₀) - Y (i', j₀)| ≤ |Y (i, j₀)| + |Y (i', j₀)| := by
    rcases abs_cases (Y (i, j₀) - Y (i', j₀)) with ⟨h, _⟩ | ⟨h, _⟩ <;>
    rcases abs_cases (Y (i, j₀)) with ⟨h2, _⟩ | ⟨h2, _⟩ <;>
    rcases abs_cases (Y (i', j₀)) with ⟨h3, _⟩ | ⟨h3, _⟩ <;>
    linarith
  calc |Y (i, j₀) - Y (i', j₀)|
      ≤ |Y (i, j₀)| + |Y (i', j₀)| := htri
      _ ≤ S + S := add_le_add (hle _) (hle _)
      _ = 2 * S := by ring

/--
Concrete OT κ-bound: the variation seminorm of the row-potential `α` is bounded by the
sup-norm of `Y` when `Y = α ⊕ β` (i.e. `Y(i,j) = α i + β j`).

This is the key quantitative step behind Proposition `prop:kappa_OT`:
the best centred representative of `α` fits inside the ball of radius `‖Y‖_∞`,
proving that `κ ≤ 1` in the balanced OT setting.
-/
theorem ot_variationSeminorm_le_supNorm
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (alpha : ι₁ → ℝ) (beta : ι₂ → ℝ) (j₀ : ι₂)
    (Y : ι₁ × ι₂ → ℝ)
    (hY : ∀ i j, alpha i + beta j = Y (i, j)) :
    variationSeminorm alpha ≤
        Finset.univ.sup' Finset.univ_nonempty (fun p : ι₁ × ι₂ => |Y p|) := by
  set S := Finset.univ.sup' Finset.univ_nonempty (fun p : ι₁ × ι₂ => |Y p|) with hS_def
  -- Step 1: for all i, i', alpha i - alpha i' ≤ 2 * S
  have key : ∀ i i' : ι₁, alpha i - alpha i' ≤ 2 * S := fun i i' => by
    exact (abs_le.mp (ot_kappa_rowSpan_bound alpha beta j₀ Y hY i i')).2
  -- Step 2: get a witness j_min achieving the minimum of alpha
  obtain ⟨j_min, _, hj_min⟩ :=
    Finset.exists_min_image Finset.univ alpha Finset.univ_nonempty
  -- Step 3: coordMin alpha = alpha j_min
  have hinf : coordMin alpha = alpha j_min := by
    unfold coordMin
    apply le_antisymm
    · exact Finset.inf'_le (s := Finset.univ) (f := alpha) (Finset.mem_univ j_min)
    · exact Finset.le_inf' _ _ (fun i _ => hj_min i (Finset.mem_univ i))
  -- Step 4: coordMax alpha - coordMin alpha ≤ 2 * S
  have hspan : coordMax alpha - coordMin alpha ≤ 2 * S := by
    rw [hinf]
    unfold coordMax
    apply sub_le_iff_le_add.mpr
    apply Finset.sup'_le Finset.univ_nonempty
    intro i _
    linarith [key i j_min]
  -- Step 5: variationSeminorm alpha = (coordMax alpha - coordMin alpha) / 2 ≤ S
  unfold variationSeminorm oscillation
  linarith

/--
Existential form of the OT κ ≤ 1 result (Proposition `prop:kappa_OT`).

For any separable decomposition `Y(i,j) = α i + β j`, there exists a shift `c`
such that the centred row-potential `α + c` satisfies `‖α + c‖_∞ ≤ ‖Y‖_∞`.

This is the direct witness construction: `c = centeringShift alpha` (defined in
`Variation.lean` as `-(coordMax alpha + coordMin alpha)/2`).
Together with `ot_variationSeminorm_le_supNorm`, it completes the blueprint for κ ≤ 1.
-/
theorem ot_centered_representative_in_supNorm_ball
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (alpha : ι₁ → ℝ) (beta : ι₂ → ℝ) (j₀ : ι₂)
    (Y : ι₁ × ι₂ → ℝ)
    (hY : ∀ i j, alpha i + beta j = Y (i, j)) :
    ∃ c : ℝ, ∀ i : ι₁,
      |alpha i + c| ≤
        Finset.univ.sup' Finset.univ_nonempty (fun p : ι₁ × ι₂ => |Y p|) := by
  refine ⟨centeringShift alpha, fun i => ?_⟩
  -- |alpha i + centeringShift alpha| ≤ variationSeminorm alpha ≤ sup |Y p|
  calc |alpha i + centeringShift alpha|
      ≤ variationSeminorm alpha :=
          abs_add_centeringShift_le_variationSeminorm alpha i
    _ ≤ Finset.univ.sup' Finset.univ_nonempty (fun p : ι₁ × ι₂ => |Y p|) :=
          ot_variationSeminorm_le_supNorm alpha beta j₀ Y hY

/--
Constructive split for Proposition `prop:kappa_OT`.

For any separable decomposition `Y(i,j) = α i + β j`, there exist `w₁ : ι₁ → ℝ` and
`w₂ : ι₂ → ℝ` such that:
1. `w₁ i + w₂ j = Y(i,j)` for all `i, j`  (same decomposition, re-centred);
2. `|w₁ i| ≤ ‖Y‖_∞` for all `i`           (row component fits in the sup-norm ball).

Witness: `w₁ = fun i => alpha i + centeringShift alpha`,
`w₂ = fun j => beta j - centeringShift alpha`.
-/
theorem ot_kappa_concrete_split
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (alpha : ι₁ → ℝ) (beta : ι₂ → ℝ) (j₀ : ι₂)
    (Y : ι₁ × ι₂ → ℝ)
    (hY : ∀ i j, alpha i + beta j = Y (i, j)) :
    ∃ w₁ : ι₁ → ℝ, ∃ w₂ : ι₂ → ℝ,
      (∀ i j, w₁ i + w₂ j = Y (i, j)) ∧
      ∀ i, |w₁ i| ≤
        Finset.univ.sup' Finset.univ_nonempty (fun p : ι₁ × ι₂ => |Y p|) := by
  -- Witness: shift alpha by centeringShift, adjust beta by the same amount
  set c := centeringShift alpha with hc_def
  refine ⟨fun i => alpha i + c, fun j => beta j - c, ?_, ?_⟩
  · -- Decomposition is preserved: (α i + c) + (β j - c) = α i + β j = Y(i,j)
    intro i j
    have := hY i j
    linarith
  · -- ‖w₁‖_∞ ≤ ‖Y‖_∞ via abs_add_centeringShift_le_variationSeminorm and
    -- ot_variationSeminorm_le_supNorm
    intro i
    exact calc |alpha i + c|
        ≤ variationSeminorm alpha :=
            abs_add_centeringShift_le_variationSeminorm alpha i
      _ ≤ Finset.univ.sup' Finset.univ_nonempty (fun p : ι₁ × ι₂ => |Y p|) :=
            ot_variationSeminorm_le_supNorm alpha beta j₀ Y hY

/--
The OT κ ≤ 1 result stated using `coordSupNorm`.

For any separable `Y = α ⊕ β`, there exist `w₁ : ι₁ → ℝ` and `w₂ : ι₂ → ℝ`
with the same decomposition and `coordSupNorm w₁ ≤ coordSupNorm Y`.

This is the statement that the column kappa constant equals 1:
the infimum of `coordSupNorm w₁ / coordSupNorm Y` over all valid decompositions
is at most 1.
-/
theorem ot_kappa_coordSupNorm_le
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (alpha : ι₁ → ℝ) (beta : ι₂ → ℝ) (j₀ : ι₂)
    (Y : ι₁ × ι₂ → ℝ) (hY : ∀ i j, alpha i + beta j = Y (i, j)) :
    ∃ w₁ : ι₁ → ℝ, ∃ w₂ : ι₂ → ℝ,
      (∀ i j, w₁ i + w₂ j = Y (i, j)) ∧
      coordSupNorm w₁ ≤ coordSupNorm Y := by
    obtain ⟨w₁, w₂, hsplit, hbound⟩ :=
      ot_kappa_concrete_split alpha beta j₀ Y hY
    refine ⟨w₁, w₂, hsplit, ?_⟩
    unfold coordSupNorm
    apply Finset.sup'_le Finset.univ_nonempty
    intro i _
    exact hbound i

/--
Concrete OT orbit bound from the split-potential κ = 1 result.

If the dual update `Psi` is non-expansive w.r.t. `variationSeminormAsSeminorm`,
and the fixed-point `alphaStar` satisfies `variationSeminorm alphaStar ≤ coordSupNorm Y`,
then for any starting point `alpha0` and any iteration count `k`:
  `variationSeminorm (Psi^[k] alpha0) ≤ variationSeminorm alpha0 + 2 * coordSupNorm Y`.

This is the concrete seminorm orbit bound for OT Sinkhorn from
Proposition `prop:uniform_iter_final`.
-/
theorem ot_variationSeminorm_iterateBound
    {ι₁ : Type*} [Fintype ι₁] [Nonempty ι₁]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar alpha0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {ι₂ : Type*} [Fintype ι₂] [Nonempty ι₂]
    {Y : ι₁ × ι₂ → ℝ}
    (hbound : variationSeminorm alphaStar ≤ coordSupNorm Y)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) alpha0) ≤
      variationSeminorm alpha0 + 2 * coordSupNorm Y := by
  have hiter :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint
      variationSeminormAsSeminorm Psi hPsi (uStar := alphaStar) (u0 := alpha0) hfix k
  have hiter' : variationSeminorm ((Psi^[k]) alpha0) ≤
      variationSeminorm alpha0 + 2 * variationSeminorm alphaStar := hiter
  linarith

/--
OT κ = 1 consequence: the variation seminorm of the optimal row potential is bounded
by the sup-norm of Y.

This is the clean form of Proposition `prop:kappa_OT` for direct application in orbit bounds:
given any separable decomposition `Y(i,j) = α i + β j`, the variation seminorm of `α`
is controlled by `coordSupNorm Y`. Combined with `ot_variationSeminorm_iterateBound`,
this gives the concrete orbit bound for OT Sinkhorn.
-/
theorem ot_kappa_one_concrete
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (alpha : ι₁ → ℝ) (beta : ι₂ → ℝ) (j₀ : ι₂)
    (Y : ι₁ × ι₂ → ℝ)
    (hY : ∀ i j, alpha i + beta j = Y (i, j)) :
    variationSeminorm alpha ≤ coordSupNorm Y :=
  ot_variationSeminorm_le_supNorm alpha beta j₀ Y hY

/--
Concrete paper-facing `\kappa = 1` normalization for balanced OT.

Under the separable decomposition `Y(i,j) = α_i + β_j`, if `‖Y‖_∞ ≤ 1` then
`variationSeminorm α ≤ 1`.
-/
theorem ot_kappa_eq_one
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (alpha : ι₁ → ℝ) (beta : ι₂ → ℝ) (j₀ : ι₂)
    (Y : ι₁ × ι₂ → ℝ)
    (hY : ∀ i j, alpha i + beta j = Y (i, j))
    (hY_unit : coordSupNorm Y ≤ 1) :
    variationSeminorm alpha ≤ 1 := by
  exact (ot_kappa_one_concrete alpha beta j₀ Y hY).trans hY_unit

/--
Bridge lemma: transfer the concrete OT `κ = 1` seminorm control to an abstract
paper-level `κ` budget parameter.
-/
theorem ot_variationSeminorm_le_kappa_of_kappa_one_concrete
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (alpha : ι₁ → ℝ) (beta : ι₂ → ℝ) (j₀ : ι₂)
    (Y : ι₁ × ι₂ → ℝ)
    (hY : ∀ i j, alpha i + beta j = Y (i, j))
    {kappa : ℝ}
    (hYkappa : coordSupNorm Y ≤ kappa) :
    variationSeminorm alpha ≤ kappa := by
  exact le_trans (ot_kappa_one_concrete alpha beta j₀ Y hY) hYkappa

/--
Bridge lemma: if a split term is controlled by the concrete OT seminorm witness,
then it satisfies the paper-facing normalized split bound (`κ = 1`).
-/
theorem ot_signedPair_split_bound_of_kappa_one_concrete
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (alpha : ι₁ → ℝ) (beta : ι₂ → ℝ) (j₀ : ι₂)
    (Y : ι₁ × ι₂ → ℝ)
    (hY : ∀ i j, alpha i + beta j = Y (i, j))
    {splitTerm kappa : ℝ}
    (hsplit : splitTerm ≤ variationSeminorm alpha)
    (hYkappa : coordSupNorm Y ≤ kappa)
    (hkappa : kappa = 1) :
    splitTerm ≤ 1 := by
  have hsplit_kappa : splitTerm ≤ kappa := by
    exact le_trans hsplit
      (ot_variationSeminorm_le_kappa_of_kappa_one_concrete alpha beta j₀ Y hY hYkappa)
  exact ot_signedPair_split_bound hsplit_kappa hkappa

/--
Bridge lemma: concrete OT `κ = 1` control upgraded to the scaled
`H_γ/κ` paper-facing budget inequality.
-/
theorem ot_signedPair_scaled_bound_of_kappa_one_concrete
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (alpha : ι₁ → ℝ) (beta : ι₂ → ℝ) (j₀ : ι₂)
    (Y : ι₁ × ι₂ → ℝ)
    (hY : ∀ i j, alpha i + beta j = Y (i, j))
    {splitTerm kappa cost gamma hGamma : ℝ}
    (hsplit : splitTerm ≤ variationSeminorm alpha)
    (hYkappa : coordSupNorm Y ≤ kappa)
    (hkappa : kappa = 1)
    (hnonneg : 0 ≤ cost + gamma * hGamma) :
    splitTerm * (cost + gamma * hGamma) ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma := by
  have hsplit_kappa : splitTerm ≤ kappa := by
    exact le_trans hsplit
      (ot_variationSeminorm_le_kappa_of_kappa_one_concrete alpha beta j₀ Y hY hYkappa)
  exact ot_signedPair_scaled_bound_of_kappa_eq_one hsplit_kappa hkappa hnonneg

/--
Separable OT witness upgraded directly to a `κ = 1` HGamma budget bound.

This is the transitivity step used repeatedly in complexity arguments:
`variationSeminorm α ≤ coordSupNorm Y ≤ hGammaKappaBudget ...`.
-/
theorem ot_variationSeminorm_le_hGammaBudget_of_kappa_one_concrete
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (alpha : ι₁ → ℝ) (beta : ι₂ → ℝ) (j₀ : ι₂)
    (Y : ι₁ × ι₂ → ℝ)
    (hY : ∀ i j, alpha i + beta j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hbudget : coordSupNorm Y ≤
      PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma)) :
    variationSeminorm alpha ≤
      PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) := by
  exact (ot_kappa_one_concrete alpha beta j₀ Y hY).trans hbudget

/--
Split-term bridge to the `κ = 1` HGamma budget from a separable OT witness.

If a split term is controlled by `variationSeminorm α`, this theorem turns
separable decomposition + coordSupNorm budget into a budget-ready inequality.
-/
theorem ot_signedPair_split_le_hGammaBudget_of_kappa_one_concrete
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (alpha : ι₁ → ℝ) (beta : ι₂ → ℝ) (j₀ : ι₂)
    (Y : ι₁ × ι₂ → ℝ)
    (hY : ∀ i j, alpha i + beta j = Y (i, j))
    {splitTerm gamma min_b C_max : ℝ}
    (hsplit : splitTerm ≤ variationSeminorm alpha)
    (hbudget : coordSupNorm Y ≤
      PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma)) :
    splitTerm ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) := by
  exact hsplit.trans
    (ot_variationSeminorm_le_hGammaBudget_of_kappa_one_concrete
      alpha beta j₀ Y hY hbudget)

/--
Separable OT witness to HGamma budget using the explicit OT formula constant.

This converts the concrete assumption `coordSupNorm Y ≤ C_max + γ|log(min_b)| + 2*C_max`
into the canonical budget inequality.
-/
theorem ot_variationSeminorm_le_hGammaBudget_explicit_formula_of_kappa_one_concrete
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (alpha : ι₁ → ℝ) (beta : ι₂ → ℝ) (j₀ : ι₂)
    (Y : ι₁ × ι₂ → ℝ)
    (hY : ∀ i j, alpha i + beta j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_explicit : coordSupNorm Y ≤ C_max + gamma * |Real.log min_b| + 2 * C_max) :
    variationSeminorm alpha ≤
      PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) := by
  rw [ot_hGammaBudget_explicit_formula hgamma hmin_b hC_max]
  exact (ot_kappa_one_concrete alpha beta j₀ Y hY).trans hbudget_explicit

/--
Split-term to HGamma budget using separable decomposition and explicit OT constants.

This is the direct complexity-facing bridge from a split hypothesis
`splitTerm ≤ variationSeminorm α` to the canonical HGamma budget inequality.
-/
theorem ot_signedPair_split_le_hGammaBudget_explicit_formula_of_kappa_one_concrete
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (alpha : ι₁ → ℝ) (beta : ι₂ → ℝ) (j₀ : ι₂)
    (Y : ι₁ × ι₂ → ℝ)
    (hY : ∀ i j, alpha i + beta j = Y (i, j))
    {splitTerm gamma min_b C_max : ℝ}
    (hsplit : splitTerm ≤ variationSeminorm alpha)
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_explicit : coordSupNorm Y ≤ C_max + gamma * |Real.log min_b| + 2 * C_max) :
    splitTerm ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) := by
  exact hsplit.trans
    (ot_variationSeminorm_le_hGammaBudget_explicit_formula_of_kappa_one_concrete
      alpha beta j₀ Y hY hgamma hmin_b hC_max hbudget_explicit)

/--
Separable OT witness to the explicit budget constant
`3 * C_max + γ * |log(min_b)|`.
-/
theorem ot_variationSeminorm_le_explicit_threeCmax_of_kappa_one_concrete
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (alpha : ι₁ → ℝ) (beta : ι₂ → ℝ) (j₀ : ι₂)
    (Y : ι₁ × ι₂ → ℝ)
    (hY : ∀ i j, alpha i + beta j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|) :
    variationSeminorm alpha ≤ 3 * C_max + gamma * |Real.log min_b| := by
  exact (ot_kappa_one_concrete alpha beta j₀ Y hY).trans hbudget_three

/--
Split-term bridge to the explicit budget constant
`3 * C_max + γ * |log(min_b)|`.
-/
theorem ot_signedPair_split_le_explicit_threeCmax_of_kappa_one_concrete
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (alpha : ι₁ → ℝ) (beta : ι₂ → ℝ) (j₀ : ι₂)
    (Y : ι₁ × ι₂ → ℝ)
    (hY : ∀ i j, alpha i + beta j = Y (i, j))
    {splitTerm gamma min_b C_max : ℝ}
    (hsplit : splitTerm ≤ variationSeminorm alpha)
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|) :
    splitTerm ≤ 3 * C_max + gamma * |Real.log min_b| := by
  exact hsplit.trans
    (ot_variationSeminorm_le_explicit_threeCmax_of_kappa_one_concrete
      alpha beta j₀ Y hY hbudget_three)

/--
Separable OT witness to canonical HGamma budget using the normalized explicit constant
`3 * C_max + γ * |log(min_b)|`.
-/
theorem ot_variationSeminorm_le_hGammaBudget_explicit_threeCmax_of_kappa_one_concrete
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (alpha : ι₁ → ℝ) (beta : ι₂ → ℝ) (j₀ : ι₂)
    (Y : ι₁ × ι₂ → ℝ)
    (hY : ∀ i j, alpha i + beta j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|) :
    variationSeminorm alpha ≤
      PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) := by
  rw [ot_hGammaBudget_explicit_formula_threeCmax hgamma hmin_b hC_max]
  exact (ot_kappa_one_concrete alpha beta j₀ Y hY).trans hbudget_three

/--
Split-term to canonical HGamma budget via the normalized explicit constant
`3 * C_max + γ * |log(min_b)|`.
-/
theorem ot_signedPair_split_le_hGammaBudget_explicit_threeCmax_of_kappa_one_concrete
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (alpha : ι₁ → ℝ) (beta : ι₂ → ℝ) (j₀ : ι₂)
    (Y : ι₁ × ι₂ → ℝ)
    (hY : ∀ i j, alpha i + beta j = Y (i, j))
    {splitTerm gamma min_b C_max : ℝ}
    (hsplit : splitTerm ≤ variationSeminorm alpha)
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|) :
    splitTerm ≤ PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) := by
  exact hsplit.trans
    (ot_variationSeminorm_le_hGammaBudget_explicit_threeCmax_of_kappa_one_concrete
      alpha beta j₀ Y hY hgamma hmin_b hC_max hbudget_three)

/--
Concrete OT orbit bound: the variation seminorm of any iterate is controlled by
the variation seminorm of the starting point plus twice the sup-norm of Y.

This directly connects `prop:kappa_OT` (`κ = 1`) to the concrete orbit bound
`variationSeminorm (Psi^[k] u₀) ≤ variationSeminorm u₀ + 2 * coordSupNorm Y`
under the assumption that the fixed point `alphaStar` has a separable decomposition.
-/
theorem ot_concrete_orbit_bound_from_split
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar alpha0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) alpha0) ≤
      variationSeminorm alpha0 + 2 * coordSupNorm Y :=
  ot_variationSeminorm_iterateBound Psi hPsi hfix
    (ot_kappa_one_concrete alphaStar betaStar j₀ Y hY) k

/--
Concrete OT orbit bound upgraded to the canonical `H_γ` budget under a separable witness.

This is the orbit-level counterpart of
`ot_variationSeminorm_le_hGammaBudget_of_kappa_one_concrete`:
it directly packages
`variationSeminorm ((Psi^[k]) alpha0) ≤ variationSeminorm alpha0 + 2 * coordSupNorm Y`
with a `coordSupNorm Y` budget certificate.
-/
theorem ot_orbit_bound_le_hGammaBudget_of_kappa_one_concrete
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar alpha0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hbudget : coordSupNorm Y ≤
      PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma))
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) alpha0) ≤
      variationSeminorm alpha0 +
        2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
          (|Real.log min_b| + 2 * C_max / gamma) := by
  have horbit := ot_concrete_orbit_bound_from_split
    (ι₁ := ι₁) (ι₂ := ι₂) (alpha0 := alpha0) Psi hPsi hfix j₀ hY k
  have hscale :
      2 * coordSupNorm Y ≤
        2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
          (|Real.log min_b| + 2 * C_max / gamma) :=
    mul_le_mul_of_nonneg_left hbudget (by positivity)
  have hadd :
      variationSeminorm alpha0 + 2 * coordSupNorm Y ≤
        variationSeminorm alpha0 +
          2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
            (|Real.log min_b| + 2 * C_max / gamma) := by
    simpa [add_comm, add_left_comm, add_assoc] using
      add_le_add_left hscale (variationSeminorm alpha0)
  exact horbit.trans hadd

/--
Concrete OT orbit bound from separable decomposition using the explicit `H_γ` formula.

This reduces the caller-side boilerplate by allowing a direct explicit-constant assumption
`coordSupNorm Y ≤ C_max + γ|log(min_b)| + 2*C_max`.
-/
theorem ot_orbit_bound_le_hGammaBudget_explicit_formula_of_kappa_one_concrete
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar alpha0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_explicit : coordSupNorm Y ≤ C_max + gamma * |Real.log min_b| + 2 * C_max)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) alpha0) ≤
      variationSeminorm alpha0 +
        2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
          (|Real.log min_b| + 2 * C_max / gamma) := by
  have hbudget : coordSupNorm Y ≤
      PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) := by
    rw [ot_hGammaBudget_explicit_formula hgamma hmin_b hC_max]
    exact hbudget_explicit
  exact ot_orbit_bound_le_hGammaBudget_of_kappa_one_concrete
    (ι₁ := ι₁) (ι₂ := ι₂) Psi hPsi hfix j₀ hY hbudget k

/--
Concrete OT orbit bound from separable decomposition in explicit `U_max` form.

From `coordSupNorm Y ≤ 3*C_max + γ*|log(min_b)|`, this returns the complexity-ready constant
`6*C_max + 2*γ*|log(min_b)|`.
-/
theorem ot_orbit_bound_le_explicit_Umax_of_kappa_one_concrete
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar alpha0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) alpha0) ≤
      variationSeminorm alpha0 + (6 * C_max + 2 * gamma * |Real.log min_b|) := by
  have horbit := ot_concrete_orbit_bound_from_split
    (ι₁ := ι₁) (ι₂ := ι₂) (alpha0 := alpha0) Psi hPsi hfix j₀ hY k
  have hscale : 2 * coordSupNorm Y ≤ 2 * (3 * C_max + gamma * |Real.log min_b|) :=
    mul_le_mul_of_nonneg_left hbudget_three (by positivity)
  calc
    variationSeminorm ((Psi^[k]) alpha0)
        ≤ variationSeminorm alpha0 + 2 * coordSupNorm Y := horbit
    _ ≤ variationSeminorm alpha0 + 2 * (3 * C_max + gamma * |Real.log min_b|) :=
      by
        simpa [add_comm, add_left_comm, add_assoc] using
          add_le_add_left hscale (variationSeminorm alpha0)
    _ = variationSeminorm alpha0 + (6 * C_max + 2 * gamma * |Real.log min_b|) := by ring

/--
Zero-base variant of the explicit `U_max` orbit bound from separable decomposition.
-/
theorem ot_orbit_bound_le_explicit_Umax_zeroBase_of_kappa_one_concrete
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar alpha0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hzero : variationSeminorm alpha0 = 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) alpha0) ≤ 6 * C_max + 2 * gamma * |Real.log min_b| := by
  have horbit := ot_orbit_bound_le_explicit_Umax_of_kappa_one_concrete
    (ι₁ := ι₁) (ι₂ := ι₂) (alpha0 := alpha0) Psi hPsi hfix j₀ hY hbudget_three k
  rw [hzero, zero_add] at horbit
  exact horbit

/--
Successor-step convenience form of the explicit `U_max` orbit bound from separable decomposition.
-/
theorem ot_orbit_bound_le_explicit_Umax_succ_of_kappa_one_concrete
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar alpha0 : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (n : ℕ) :
    variationSeminorm ((Psi^[n + 1]) alpha0) ≤
      variationSeminorm alpha0 + (6 * C_max + 2 * gamma * |Real.log min_b|) :=
  ot_orbit_bound_le_explicit_Umax_of_kappa_one_concrete
    (ι₁ := ι₁) (ι₂ := ι₂) (alpha0 := alpha0) Psi hPsi hfix j₀ hY hbudget_three (n + 1)

/--
Zero-function basepoint variant of the explicit `U_max` orbit bound.
-/
theorem ot_orbit_bound_le_explicit_Umax_zeroFn_of_kappa_one_concrete
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) (fun _ : ι₁ => (0 : ℝ))) ≤
      6 * C_max + 2 * gamma * |Real.log min_b| :=
  ot_orbit_bound_le_explicit_Umax_zeroBase_of_kappa_one_concrete
    (ι₁ := ι₁) (ι₂ := ι₂) (alpha0 := fun _ : ι₁ => (0 : ℝ))
    Psi hPsi hfix j₀ hY hbudget_three
    (by simpa using (variationSeminorm_zero (ι := ι₁))) k

/--
Successor-step zero-function variant of the explicit `U_max` orbit bound.
-/
theorem ot_orbit_bound_le_explicit_Umax_zeroFn_succ_of_kappa_one_concrete
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (n : ℕ) :
    variationSeminorm ((Psi^[n + 1]) (fun _ : ι₁ => (0 : ℝ))) ≤
      6 * C_max + 2 * gamma * |Real.log min_b| :=
  ot_orbit_bound_le_explicit_Umax_zeroFn_of_kappa_one_concrete
    (ι₁ := ι₁) (ι₂ := ι₂) Psi hPsi hfix j₀ hY hbudget_three (n + 1)

/--
Zero-function orbit bound under the explicit `H_γ`-formula assumption.
-/
theorem ot_orbit_bound_le_hGammaBudget_explicit_formula_zeroFn_of_kappa_one_concrete
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_explicit : coordSupNorm Y ≤ C_max + gamma * |Real.log min_b| + 2 * C_max)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) (fun _ : ι₁ => (0 : ℝ))) ≤
      2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) := by
  have horbit := ot_orbit_bound_le_hGammaBudget_explicit_formula_of_kappa_one_concrete
    (ι₁ := ι₁) (ι₂ := ι₂) (alpha0 := fun _ : ι₁ => (0 : ℝ))
    Psi hPsi hfix j₀ hY hgamma hmin_b hC_max hbudget_explicit k
  have hzero : variationSeminorm (fun _ : ι₁ => (0 : ℝ)) = 0 := by
    simpa using (variationSeminorm_zero (ι := ι₁))
  simpa [hzero, zero_add] using horbit

/--
Successor-step zero-function orbit bound under the explicit `H_γ`-formula assumption.
-/
theorem ot_orbit_bound_le_hGammaBudget_explicit_formula_zeroFn_succ_of_kappa_one_concrete
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_explicit : coordSupNorm Y ≤ C_max + gamma * |Real.log min_b| + 2 * C_max)
    (n : ℕ) :
    variationSeminorm ((Psi^[n + 1]) (fun _ : ι₁ => (0 : ℝ))) ≤
      2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) :=
  ot_orbit_bound_le_hGammaBudget_explicit_formula_zeroFn_of_kappa_one_concrete
    (ι₁ := ι₁) (ι₂ := ι₂) Psi hPsi hfix j₀ hY
    hgamma hmin_b hC_max hbudget_explicit (n + 1)

/--
Upper-constant bridge for the explicit `H_γ`-formula zero-function orbit bound.
-/
theorem
    ot_orbit_bound_le_hGammaBudget_explicit_formula_zeroFn_of_kappa_one_concrete_of_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max U : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_explicit : coordSupNorm Y ≤ C_max + gamma * |Real.log min_b| + 2 * C_max)
    (hU : 2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma) ≤ U)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) (fun _ : ι₁ => (0 : ℝ))) ≤ U :=
  (ot_orbit_bound_le_hGammaBudget_explicit_formula_zeroFn_of_kappa_one_concrete
    (ι₁ := ι₁) (ι₂ := ι₂) Psi hPsi hfix j₀ hY
    hgamma hmin_b hC_max hbudget_explicit k).trans hU

/--
Successor-step upper-constant bridge for the explicit `H_γ`-formula zero-function orbit bound.
-/
theorem
    ot_orbit_bound_le_hGammaBudget_explicit_formula_zeroFn_succ_of_kappa_one_concrete_of_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max U : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_explicit : coordSupNorm Y ≤ C_max + gamma * |Real.log min_b| + 2 * C_max)
    (hU : 2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma) ≤ U)
    (n : ℕ) :
    variationSeminorm ((Psi^[n + 1]) (fun _ : ι₁ => (0 : ℝ))) ≤ U :=
  (ot_orbit_bound_le_hGammaBudget_explicit_formula_zeroFn_succ_of_kappa_one_concrete
    (ι₁ := ι₁) (ι₂ := ι₂) Psi hPsi hfix j₀ hY
    hgamma hmin_b hC_max hbudget_explicit n).trans hU

/--
Upper-constant bridge for the explicit-`U_max` zero-function orbit bound.
-/
theorem ot_orbit_bound_le_explicit_Umax_zeroFn_of_kappa_one_concrete_of_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max U : ℝ}
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ U)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) (fun _ : ι₁ => (0 : ℝ))) ≤ U :=
  (ot_orbit_bound_le_explicit_Umax_zeroFn_of_kappa_one_concrete
    (ι₁ := ι₁) (ι₂ := ι₂) Psi hPsi hfix j₀ hY hbudget_three k).trans hU

/--
Successor-step upper-constant bridge for the explicit-`U_max` zero-function orbit bound.
-/
theorem ot_orbit_bound_le_explicit_Umax_zeroFn_succ_of_kappa_one_concrete_of_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max U : ℝ}
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ U)
    (n : ℕ) :
    variationSeminorm ((Psi^[n + 1]) (fun _ : ι₁ => (0 : ℝ))) ≤ U :=
  (ot_orbit_bound_le_explicit_Umax_zeroFn_succ_of_kappa_one_concrete
    (ι₁ := ι₁) (ι₂ := ι₂) Psi hPsi hfix j₀ hY hbudget_three n).trans hU

/--
Chained-upper-constant bridge for the explicit-`U_max` zero-function orbit bound.
-/
theorem ot_orbit_bound_le_explicit_Umax_zeroFn_of_kappa_one_concrete_of_upperConstant_of_le_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max U V : ℝ}
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ U)
    (hUV : U ≤ V)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) (fun _ : ι₁ => (0 : ℝ))) ≤ V :=
  (ot_orbit_bound_le_explicit_Umax_zeroFn_of_kappa_one_concrete_of_upperConstant
    (ι₁ := ι₁) (ι₂ := ι₂) Psi hPsi hfix j₀ hY hbudget_three hU k).trans hUV

/--
Chained-upper-constant bridge for the explicit `H_γ`-formula zero-function orbit bound.
-/
theorem
    ot_orbit_bound_le_hGammaBudget_explicit_formula_zeroFn_of_kappa_one_concrete_of_upperConstant_of_le_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max U V : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_explicit : coordSupNorm Y ≤ C_max + gamma * |Real.log min_b| + 2 * C_max)
    (hU : 2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma) ≤ U)
    (hUV : U ≤ V)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) (fun _ : ι₁ => (0 : ℝ))) ≤ V :=
  (ot_orbit_bound_le_hGammaBudget_explicit_formula_zeroFn_of_kappa_one_concrete_of_upperConstant
    (ι₁ := ι₁) (ι₂ := ι₂) Psi hPsi hfix j₀ hY
    hgamma hmin_b hC_max hbudget_explicit hU k).trans hUV

/--
Explicit-`U_max` zero-function orbit bound evaluated at a ceiling index.
-/
theorem ot_orbit_bound_explicit_Umax_zeroFn_at_ceil_index
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max eps : ℝ}
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|) :
    variationSeminorm
        ((Psi^[Nat.ceil ((6 * C_max + 2 * gamma * |Real.log min_b|) / eps)])
          (fun _ : ι₁ => (0 : ℝ))) ≤
      6 * C_max + 2 * gamma * |Real.log min_b| :=
  ot_orbit_bound_le_explicit_Umax_zeroFn_of_kappa_one_concrete
    (ι₁ := ι₁) (ι₂ := ι₂) Psi hPsi hfix j₀ hY hbudget_three
    (Nat.ceil ((6 * C_max + 2 * gamma * |Real.log min_b|) / eps))

/--
Upper-constant ceiling-index wrapper for the explicit-`U_max` zero-function orbit bound.
-/
theorem ot_orbit_bound_explicit_Umax_zeroFn_at_ceil_index_of_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max eps U : ℝ}
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ U) :
    variationSeminorm
        ((Psi^[Nat.ceil ((6 * C_max + 2 * gamma * |Real.log min_b|) / eps)])
          (fun _ : ι₁ => (0 : ℝ))) ≤
      U :=
  (ot_orbit_bound_explicit_Umax_zeroFn_at_ceil_index
    (ι₁ := ι₁) (ι₂ := ι₂) Psi hPsi hfix j₀ hY hbudget_three).trans hU

/--
Explicit `H_γ`-budget zero-function orbit bound evaluated at a ceiling index.
-/
theorem ot_orbit_bound_hGammaBudget_zeroFn_at_ceil_index
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max eps : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_explicit : coordSupNorm Y ≤ C_max + gamma * |Real.log min_b| + 2 * C_max) :
    variationSeminorm
        ((Psi^[Nat.ceil
          ((2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
            (|Real.log min_b| + 2 * C_max / gamma)) / eps)])
          (fun _ : ι₁ => (0 : ℝ))) ≤
      2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) :=
  ot_orbit_bound_le_hGammaBudget_explicit_formula_zeroFn_of_kappa_one_concrete
    (ι₁ := ι₁) (ι₂ := ι₂) Psi hPsi hfix j₀ hY
    hgamma hmin_b hC_max hbudget_explicit
    (Nat.ceil ((2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma)) / eps))

/--
Upper-constant ceiling-index wrapper for the explicit `H_γ`-budget zero-function orbit bound.
-/
theorem ot_orbit_bound_hGammaBudget_zeroFn_at_ceil_index_of_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max eps U : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_explicit : coordSupNorm Y ≤ C_max + gamma * |Real.log min_b| + 2 * C_max)
    (hU : 2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma) ≤ U) :
    variationSeminorm
        ((Psi^[Nat.ceil
          ((2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
            (|Real.log min_b| + 2 * C_max / gamma)) / eps)])
          (fun _ : ι₁ => (0 : ℝ))) ≤
      U :=
  (ot_orbit_bound_hGammaBudget_zeroFn_at_ceil_index
    (ι₁ := ι₁) (ι₂ := ι₂) Psi hPsi hfix j₀ hY
    hgamma hmin_b hC_max hbudget_explicit).trans hU

/--
Successor-index wrapper for explicit-`U_max` zero-function orbit bound at ceiling index.
-/
theorem ot_orbit_bound_explicit_Umax_zeroFn_at_ceil_index_succ
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max eps : ℝ}
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|) :
    variationSeminorm
        ((Psi^[Nat.ceil ((6 * C_max + 2 * gamma * |Real.log min_b|) / eps) + 1])
          (fun _ : ι₁ => (0 : ℝ))) ≤
      6 * C_max + 2 * gamma * |Real.log min_b| :=
  ot_orbit_bound_le_explicit_Umax_zeroFn_of_kappa_one_concrete
    (ι₁ := ι₁) (ι₂ := ι₂) Psi hPsi hfix j₀ hY hbudget_three
    (Nat.ceil ((6 * C_max + 2 * gamma * |Real.log min_b|) / eps) + 1)

/--
Successor-index wrapper for explicit `H_γ`-budget zero-function orbit bound
at ceiling index.
-/
theorem ot_orbit_bound_hGammaBudget_zeroFn_at_ceil_index_succ
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max eps : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_explicit : coordSupNorm Y ≤ C_max + gamma * |Real.log min_b| + 2 * C_max) :
    variationSeminorm
        ((Psi^[Nat.ceil
          ((2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
            (|Real.log min_b| + 2 * C_max / gamma)) / eps) + 1])
          (fun _ : ι₁ => (0 : ℝ))) ≤
      2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma) :=
  ot_orbit_bound_le_hGammaBudget_explicit_formula_zeroFn_of_kappa_one_concrete
    (ι₁ := ι₁) (ι₂ := ι₂) Psi hPsi hfix j₀ hY
    hgamma hmin_b hC_max hbudget_explicit
    (Nat.ceil
      ((2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
        (|Real.log min_b| + 2 * C_max / gamma)) / eps) + 1)

/--
Successor-index upper-constant wrapper for explicit `H_γ`-budget zero-function orbit bound
at ceiling index.
-/
theorem ot_orbit_bound_hGammaBudget_zeroFn_at_ceil_index_succ_of_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max eps U : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_explicit : coordSupNorm Y ≤ C_max + gamma * |Real.log min_b| + 2 * C_max)
    (hU : 2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma) ≤ U) :
    variationSeminorm
        ((Psi^[Nat.ceil
          ((2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
            (|Real.log min_b| + 2 * C_max / gamma)) / eps) + 1])
          (fun _ : ι₁ => (0 : ℝ))) ≤
      U :=
  (ot_orbit_bound_hGammaBudget_zeroFn_at_ceil_index_succ
    (ι₁ := ι₁) (ι₂ := ι₂) Psi hPsi hfix j₀ hY
    hgamma hmin_b hC_max hbudget_explicit).trans hU

/--
Successor-index upper-constant wrapper for explicit-`U_max` zero-function orbit bound
at ceiling index.
-/
theorem ot_orbit_bound_explicit_Umax_zeroFn_at_ceil_index_succ_of_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max eps U : ℝ}
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ U) :
    variationSeminorm
        ((Psi^[Nat.ceil ((6 * C_max + 2 * gamma * |Real.log min_b|) / eps) + 1])
          (fun _ : ι₁ => (0 : ℝ))) ≤
      U :=
  (ot_orbit_bound_explicit_Umax_zeroFn_at_ceil_index_succ
    (ι₁ := ι₁) (ι₂ := ι₂) Psi hPsi hfix j₀ hY hbudget_three).trans hU

/--
Chained-upper-constant bridge for explicit-`U_max` zero-function orbit bound at ceiling index.
-/
theorem ot_orbit_bound_explicit_Umax_zeroFn_at_ceil_index_of_upperConstant_of_le_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max eps U V : ℝ}
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ U)
    (hUV : U ≤ V) :
    variationSeminorm
        ((Psi^[Nat.ceil ((6 * C_max + 2 * gamma * |Real.log min_b|) / eps)])
          (fun _ : ι₁ => (0 : ℝ))) ≤
      V :=
  (ot_orbit_bound_explicit_Umax_zeroFn_at_ceil_index_of_upperConstant
    (ι₁ := ι₁) (ι₂ := ι₂) Psi hPsi hfix j₀ hY hbudget_three hU).trans hUV

/--
Chained-upper-constant bridge for successor-index explicit `H_γ`-budget zero-function
orbit bound at ceiling index.
-/
theorem ot_orbit_bound_hGammaBudget_zeroFn_at_ceil_index_succ_of_upperConstant_of_le_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max eps U V : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_explicit : coordSupNorm Y ≤ C_max + gamma * |Real.log min_b| + 2 * C_max)
    (hU : 2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma) ≤ U)
    (hUV : U ≤ V) :
    variationSeminorm
        ((Psi^[Nat.ceil
          ((2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
            (|Real.log min_b| + 2 * C_max / gamma)) / eps) + 1])
          (fun _ : ι₁ => (0 : ℝ))) ≤
      V :=
  (ot_orbit_bound_hGammaBudget_zeroFn_at_ceil_index_succ_of_upperConstant
    (ι₁ := ι₁) (ι₂ := ι₂) Psi hPsi hfix j₀ hY
    hgamma hmin_b hC_max hbudget_explicit hU).trans hUV

/--
Chained-upper-constant bridge for successor-index explicit-`U_max` zero-function
orbit bound at ceiling index.
-/
theorem ot_orbit_bound_explicit_Umax_zeroFn_at_ceil_index_succ_of_upperConstant_of_le_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max eps U V : ℝ}
    (hbudget_three : coordSupNorm Y ≤ 3 * C_max + gamma * |Real.log min_b|)
    (hU : 6 * C_max + 2 * gamma * |Real.log min_b| ≤ U)
    (hUV : U ≤ V) :
    variationSeminorm
        ((Psi^[Nat.ceil ((6 * C_max + 2 * gamma * |Real.log min_b|) / eps) + 1])
          (fun _ : ι₁ => (0 : ℝ))) ≤
      V :=
  (ot_orbit_bound_explicit_Umax_zeroFn_at_ceil_index_succ_of_upperConstant
    (ι₁ := ι₁) (ι₂ := ι₂) Psi hPsi hfix j₀ hY hbudget_three hU).trans hUV

/--
Chained-upper-constant bridge for explicit `H_γ`-budget zero-function
orbit bound at ceiling index.
-/
theorem ot_orbit_bound_hGammaBudget_zeroFn_at_ceil_index_of_upperConstant_of_le_upperConstant
    {ι₁ ι₂ : Type*} [Fintype ι₁] [Nonempty ι₁] [Fintype ι₂] [Nonempty ι₂]
    (Psi : (ι₁ → ℝ) → (ι₁ → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {alphaStar : ι₁ → ℝ} (hfix : Psi alphaStar = alphaStar)
    {betaStar : ι₂ → ℝ} (j₀ : ι₂)
    {Y : ι₁ × ι₂ → ℝ}
    (hY : ∀ i j, alphaStar i + betaStar j = Y (i, j))
    {gamma min_b C_max eps U V : ℝ}
    (hgamma : 0 < gamma) (hmin_b : 0 < min_b) (hC_max : 0 ≤ C_max)
    (hbudget_explicit : coordSupNorm Y ≤ C_max + gamma * |Real.log min_b| + 2 * C_max)
    (hU : 2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
      (|Real.log min_b| + 2 * C_max / gamma) ≤ U)
    (hUV : U ≤ V) :
    variationSeminorm
        ((Psi^[Nat.ceil
          ((2 * PrimalDualBounds.hGammaKappaBudget 1 C_max gamma
            (|Real.log min_b| + 2 * C_max / gamma)) / eps)])
          (fun _ : ι₁ => (0 : ℝ))) ≤
      V :=
  (ot_orbit_bound_hGammaBudget_zeroFn_at_ceil_index_of_upperConstant
    (ι₁ := ι₁) (ι₂ := ι₂) Psi hPsi hfix j₀ hY
    hgamma hmin_b hC_max hbudget_explicit hU).trans hUV

end OT
end Applications
end KLProjection
end FlowSinkhorn
