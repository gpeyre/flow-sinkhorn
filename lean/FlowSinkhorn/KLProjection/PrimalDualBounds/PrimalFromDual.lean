import FlowSinkhorn.KLProjection.Setup
import FlowSinkhorn.KLProjection.PrimalDualBounds.Vocabulary
import Mathlib.Order.Monotone.Basic

/-!
# Primal bound from dual orbit radius

This module provides the monotone transfer lemma: if a primal quantity `X` is
controlled at a dual radius `d` (via a monotone function `Xbound`), and `d ≤ U`,
then `X ≤ Xbound U`.

## Note on reuse

This is a pure order-theoretic fact, independent of the specific problem. It can be
used by any application where primal feasibility is certified via a dual radius bound.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace PrimalDualBounds

open scoped BigOperators

/--
Abstract transport of a primal confinement estimate along a dual-radius bound.

This captures the generic blueprint shape:
if one has a primal control function `Xbound` known at a dual quantity `d`, then any upper bound
`d ≤ U` propagates the primal estimate to `U` by monotonicity.
-/
theorem primalBound_fromDualBound
    {d U X : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ U) :
    X ≤ Xbound U :=
  hprimal_at_d.trans (hmono hdual)

/--
Paper-facing alias with the upper dual quantity named as a budget.

This is the same transfer principle as `primalBound_fromDualBound`, but the statement shape is
often closer to the way Section 4 applications are written.
-/
theorem primalBound_fromDualBudget
    {d budget X : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hdual : d ≤ budget) :
    X ≤ Xbound budget := by
  calc
    X ≤ Xbound d := hprimal_at_d
    _ ≤ Xbound budget := hmono hdual

/--
Finite pairing bound used in the proof of Proposition `prop:mass-bound-block`.

If `|v_i| <= B` coordinatewise and the `l1` mass of `b` is bounded by `mass_b`, then the
plain pairing `sum_i b_i v_i` is bounded above by `mass_b * B`.
-/
theorem finite_pairing_le_l1_mul_bound
    {idx : Type*} [Fintype idx]
    {b v : idx → ℝ} {B mass_b : ℝ}
    (hB : 0 ≤ B)
    (hv : ∀ i : idx, |v i| ≤ B)
    (hmass : (∑ i : idx, |b i|) ≤ mass_b) :
    (∑ i : idx, b i * v i) ≤ mass_b * B := by
  have hterm : ∀ i : idx, |b i * v i| ≤ |b i| * B := by
    intro i
    calc
      |b i * v i| = |b i| * |v i| := by rw [abs_mul]
      _ ≤ |b i| * B := mul_le_mul_of_nonneg_left (hv i) (abs_nonneg (b i))
  calc
    (∑ i : idx, b i * v i)
        ≤ |∑ i : idx, b i * v i| := le_abs_self _
    _ ≤ ∑ i : idx, |b i * v i| := by
        simpa using
          (Finset.abs_sum_le_sum_abs
            (fun i : idx => b i * v i) (Finset.univ : Finset idx))
    _ ≤ ∑ i : idx, |b i| * B := Finset.sum_le_sum fun i _hi => hterm i
    _ = (∑ i : idx, |b i|) * B := by rw [Finset.sum_mul]
    _ ≤ mass_b * B := mul_le_mul_of_nonneg_right hmass hB

/--
Kernel-shift pairing estimate from the proof of Proposition `prop:mass-bound-block`.

The paper replaces `u` by a representative `u+h` whose coordinates are bounded by the quotient
radius.  If the shift is orthogonal to `b`, the pairing is unchanged and the finite `l1`/`linf`
bound controls it by `mass_b * radius`.
-/
theorem shiftedPairing_le_l1_mul_radius
    {idx : Type*} [Fintype idx]
    {b u shift : idx → ℝ} {dualPair mass_b radius : ℝ}
    (hradius : 0 ≤ radius)
    (hpair_eq : dualPair = ∑ i : idx, b i * u i)
    (hshift_orth : (∑ i : idx, b i * shift i) = 0)
    (hshift_bound : ∀ i : idx, |u i + shift i| ≤ radius)
    (hmass : (∑ i : idx, |b i|) ≤ mass_b) :
    dualPair ≤ mass_b * radius := by
  have hshifted :
      (∑ i : idx, b i * (u i + shift i)) = (∑ i : idx, b i * u i) := by
    calc
      (∑ i : idx, b i * (u i + shift i))
          = (∑ i : idx, b i * u i) + (∑ i : idx, b i * shift i) := by
            simp [mul_add, Finset.sum_add_distrib]
      _ = (∑ i : idx, b i * u i) := by rw [hshift_orth, add_zero]
  have hpair_shifted :
      dualPair = ∑ i : idx, b i * (u i + shift i) :=
    hpair_eq.trans hshifted.symm
  rw [hpair_shifted]
  exact finite_pairing_le_l1_mul_bound
    (B := radius) (mass_b := mass_b) hradius hshift_bound hmass

/--
Nonnegativity of an `l1` mass upper bound.

If `mass_b` is an upper bound for the finite `l1` mass `sum_i |b_i|`, then `mass_b` is
nonnegative.  This keeps paper-facing mass-bound statements from carrying a redundant
`0 <= mass_b` hypothesis.
-/
theorem massBound_nonneg_of_l1_bound
    {idx : Type*} [Fintype idx]
    {b : idx → ℝ} {mass_b : ℝ}
    (hmass : (∑ i : idx, |b i|) ≤ mass_b) :
    0 ≤ mass_b := by
  have hsum_nonneg : 0 ≤ (∑ i : idx, |b i|) := by
    exact Finset.sum_nonneg fun i _hi => abs_nonneg (b i)
  exact hsum_nonneg.trans hmass

/--
Nonnegativity of a finite `linf` radius.

On a nonempty finite block, any scalar that bounds all absolute coordinate values must be
nonnegative.  This removes a redundant radius-sign hypothesis from the paper-facing
mass-bound endpoint: the sign follows from the displayed representative bound itself.
-/
theorem radius_nonneg_of_abs_bound
    {idx : Type*} [Nonempty idx]
    {v : idx → ℝ} {radius : ℝ}
    (hbound : ∀ i : idx, |v i| ≤ radius) :
    0 ≤ radius := by
  let i : idx := Classical.choice (inferInstance : Nonempty idx)
  exact (abs_nonneg (v i)).trans (hbound i)

/--
Paper-facing explicit mass bound for Proposition `prop:mass-bound-block`.

The theorem isolates the algebraic step used in the proposition.  If the current primal mass is
bounded by the paper's affine envelope evaluated at the current dual radius, and all dual radii
are uniformly bounded by `Umax`, then every primal iterate is bounded by the displayed constant
`mass_b * Umax / gamma + d * exp (-Cmin / gamma)`.
-/
theorem primalMassBound_explicit_from_dualRadius
    {xMass dualRadius : ℕ → ℝ}
    {mass_b Umax gamma d Cmin : ℝ}
    (hgamma : 0 < gamma)
    (hmass_b : 0 ≤ mass_b)
    (hdual : ∀ k : ℕ, dualRadius k ≤ Umax)
    (hpointwise : ∀ k : ℕ,
      xMass k ≤ mass_b * dualRadius k / gamma + d * Real.exp (-Cmin / gamma)) :
    ∀ k : ℕ,
      xMass k ≤ mass_b * Umax / gamma + d * Real.exp (-Cmin / gamma) := by
  intro k
  have hscale : mass_b * dualRadius k / gamma ≤ mass_b * Umax / gamma := by
    have hmul : mass_b * dualRadius k ≤ mass_b * Umax :=
      mul_le_mul_of_nonneg_left (hdual k) hmass_b
    have hinv : 0 ≤ gamma⁻¹ := inv_nonneg.mpr (le_of_lt hgamma)
    have hmul_inv : (mass_b * dualRadius k) * gamma⁻¹ ≤
        (mass_b * Umax) * gamma⁻¹ :=
      mul_le_mul_of_nonneg_right hmul hinv
    simpa [div_eq_mul_inv] using hmul_inv
  exact (hpointwise k).trans (add_le_add hscale le_rfl)

/--
Coordinate-envelope version of Proposition `prop:mass-bound-block`.

This is closer to the proof used in the paper than `primalMassBound_explicit_from_dualRadius`:
instead of assuming the already-aggregated estimate on `xMass`, it starts from a coordinatewise
envelope `x k i <= envelope k i + exp(-Cmin/gamma)`.  Lean sums this envelope over the finite
coordinate set, uses the certified dual-radius contribution
`sum_i envelope(k,i) <= ||b||₁ * dualRadius(k) / gamma`, and then transports the radius through the
uniform bound `dualRadius(k) <= Umax`.
-/
theorem primalMassBound_explicit_from_coordinateEnvelope
    {coord : Type*} [Fintype coord]
    {x envelope : ℕ → coord → ℝ}
    {dualRadius : ℕ → ℝ}
    {mass_b Umax gamma d Cmin : ℝ}
    (hgamma : 0 < gamma)
    (hmass_b : 0 ≤ mass_b)
    (hcard : (Fintype.card coord : ℝ) ≤ d)
    (hdual : ∀ k : ℕ, dualRadius k ≤ Umax)
    (henvelope : ∀ k : ℕ,
      (∑ i : coord, envelope k i) ≤ mass_b * dualRadius k / gamma)
    (hcoord : ∀ k : ℕ, ∀ i : coord,
      x k i ≤ envelope k i + Real.exp (-Cmin / gamma)) :
    ∀ k : ℕ,
      (∑ i : coord, x k i) ≤
        mass_b * Umax / gamma + d * Real.exp (-Cmin / gamma) := by
  intro k
  let tail : ℝ := Real.exp (-Cmin / gamma)
  have hsum_coord :
      (∑ i : coord, x k i) ≤
        (∑ i : coord, (envelope k i + tail)) := by
    simpa [tail] using Finset.sum_le_sum fun i _hi => hcoord k i
  have hsum_split :
      (∑ i : coord, (envelope k i + tail)) =
        (∑ i : coord, envelope k i) + (Fintype.card coord : ℝ) * tail := by
    simp [Finset.sum_add_distrib, tail]
  have hscale : mass_b * dualRadius k / gamma ≤ mass_b * Umax / gamma := by
    have hmul : mass_b * dualRadius k ≤ mass_b * Umax :=
      mul_le_mul_of_nonneg_left (hdual k) hmass_b
    have hinv : 0 ≤ gamma⁻¹ := inv_nonneg.mpr (le_of_lt hgamma)
    have hmul_inv :
        (mass_b * dualRadius k) * gamma⁻¹ ≤ (mass_b * Umax) * gamma⁻¹ :=
      mul_le_mul_of_nonneg_right hmul hinv
    simpa [div_eq_mul_inv] using hmul_inv
  have htail_nonneg : 0 ≤ tail := le_of_lt (Real.exp_pos _)
  have htail_card : (Fintype.card coord : ℝ) * tail ≤ d * tail :=
    mul_le_mul_of_nonneg_right hcard htail_nonneg
  calc
    (∑ i : coord, x k i)
        ≤ (∑ i : coord, (envelope k i + tail)) := hsum_coord
    _ = (∑ i : coord, envelope k i) + (Fintype.card coord : ℝ) * tail := hsum_split
    _ ≤ mass_b * dualRadius k / gamma + (Fintype.card coord : ℝ) * tail :=
        add_le_add (henvelope k) le_rfl
    _ ≤ mass_b * Umax / gamma + d * tail := add_le_add hscale htail_card
    _ = mass_b * Umax / gamma + d * Real.exp (-Cmin / gamma) := rfl

/--
Cost-floor exponential-tail version of Proposition `prop:mass-bound-block`.

Compared to `primalMassBound_explicit_from_coordinateEnvelope`, this theorem does not take the
per-coordinate tail bound `z_i exp(-C_i / gamma) <= exp(-Cmin / gamma)` as an opaque premise.
It derives that tail estimate from `z_i <= 1`, the cost floor `Cmin <= C_i`, and monotonicity of
the exponential under `gamma > 0`, before invoking the finite coordinate-envelope summation.
-/
theorem primalMassBound_explicit_from_exponentialTail
    {coord : Type*} [Fintype coord]
    {x envelope : ℕ → coord → ℝ}
    {dualRadius : ℕ → ℝ}
    {z C : coord → ℝ}
    {mass_b Umax gamma d Cmin : ℝ}
    (hgamma : 0 < gamma)
    (hmass_b : 0 ≤ mass_b)
    (hcard : (Fintype.card coord : ℝ) ≤ d)
    (hz_le_one : ∀ i : coord, z i ≤ 1)
    (hCmin : ∀ i : coord, Cmin ≤ C i)
    (hdual : ∀ k : ℕ, dualRadius k ≤ Umax)
    (henvelope : ∀ k : ℕ,
      (∑ i : coord, envelope k i) ≤ mass_b * dualRadius k / gamma)
    (hcoord : ∀ k : ℕ, ∀ i : coord,
      x k i ≤ envelope k i + z i * Real.exp (-C i / gamma)) :
    ∀ k : ℕ,
      (∑ i : coord, x k i) ≤
        mass_b * Umax / gamma + d * Real.exp (-Cmin / gamma) := by
  have htail : ∀ i : coord,
      z i * Real.exp (-C i / gamma) ≤ Real.exp (-Cmin / gamma) := by
    intro i
    have hneg : -C i ≤ -Cmin := by linarith [hCmin i]
    have hdiv : -C i / gamma ≤ -Cmin / gamma := by
      have hinv_nonneg : 0 ≤ gamma⁻¹ := inv_nonneg.mpr (le_of_lt hgamma)
      have hmul := mul_le_mul_of_nonneg_right hneg hinv_nonneg
      simpa [div_eq_mul_inv] using hmul
    have hexp : Real.exp (-C i / gamma) ≤ Real.exp (-Cmin / gamma) :=
      Real.exp_le_exp.mpr hdiv
    have hzexp :
        z i * Real.exp (-C i / gamma) ≤ 1 * Real.exp (-C i / gamma) :=
      mul_le_mul_of_nonneg_right (hz_le_one i) (le_of_lt (Real.exp_pos _))
    calc
      z i * Real.exp (-C i / gamma)
          ≤ 1 * Real.exp (-C i / gamma) := hzexp
      _ = Real.exp (-C i / gamma) := by ring
      _ ≤ Real.exp (-Cmin / gamma) := hexp
  have hcoord_tail : ∀ k : ℕ, ∀ i : coord,
      x k i ≤ envelope k i + Real.exp (-Cmin / gamma) := by
    intro k i
    exact (hcoord k i).trans (add_le_add_right (htail i) (envelope k i))
  exact primalMassBound_explicit_from_coordinateEnvelope
    hgamma hmass_b hcard hdual henvelope hcoord_tail

/--
Normalized-reference version of Proposition `prop:mass-bound-block`.

This strengthens `primalMassBound_explicit_from_exponentialTail`: the paper's reference weights
are normalized nonnegative masses, so the coordinate bound `z_i <= 1` should not be an opaque
input.  Lean derives it from `0 <= z_i` and `sum_i z_i <= 1`, then applies the cost-floor
exponential-tail theorem.
-/
theorem primalMassBound_explicit_from_normalizedExponentialTail
    {coord : Type*} [Fintype coord]
    {x envelope : ℕ → coord → ℝ}
    {dualRadius : ℕ → ℝ}
    {z C : coord → ℝ}
    {mass_b Umax gamma d Cmin : ℝ}
    (hgamma : 0 < gamma)
    (hmass_b : 0 ≤ mass_b)
    (hcard : (Fintype.card coord : ℝ) ≤ d)
    (hz_nonneg : ∀ i : coord, 0 ≤ z i)
    (hz_mass : (∑ i : coord, z i) ≤ 1)
    (hCmin : ∀ i : coord, Cmin ≤ C i)
    (hdual : ∀ k : ℕ, dualRadius k ≤ Umax)
    (henvelope : ∀ k : ℕ,
      (∑ i : coord, envelope k i) ≤ mass_b * dualRadius k / gamma)
    (hcoord : ∀ k : ℕ, ∀ i : coord,
      x k i ≤ envelope k i + z i * Real.exp (-C i / gamma)) :
    ∀ k : ℕ,
      (∑ i : coord, x k i) ≤
        mass_b * Umax / gamma + d * Real.exp (-Cmin / gamma) := by
  classical
  have hz_le_one : ∀ i : coord, z i ≤ 1 := by
    intro i
    have hsingle : z i ≤ (∑ j : coord, z j) := by
      simpa using
        (Finset.single_le_sum
          (s := Finset.univ) (f := z)
          (by intro j _hj; exact hz_nonneg j)
          (by simp : i ∈ (Finset.univ : Finset coord)))
    exact hsingle.trans hz_mass
  exact primalMassBound_explicit_from_exponentialTail
    hgamma hmass_b hcard hz_le_one hCmin hdual henvelope hcoord

/--
Share-envelope version of Proposition `prop:mass-bound-block`.

This endpoint removes the already-summed envelope hypothesis from
`primalMassBound_explicit_from_normalizedExponentialTail`.  Instead it assumes a coordinate share
`share(k,i)` whose finite sum is bounded by the dual radius, and it defines the coordinate envelope
as `(||b||₁/γ) * share(k,i)`.  Lean then proves the aggregate envelope bound by summing the shares
before applying the normalized cost-floor theorem.
-/
theorem primalMassBound_explicit_from_normalizedShareEnvelope
    {coord : Type*} [Fintype coord]
    {x share : ℕ → coord → ℝ}
    {dualRadius : ℕ → ℝ}
    {z C : coord → ℝ}
    {mass_b Umax gamma d Cmin : ℝ}
    (hgamma : 0 < gamma)
    (hmass_b : 0 ≤ mass_b)
    (hcard : (Fintype.card coord : ℝ) ≤ d)
    (hz_nonneg : ∀ i : coord, 0 ≤ z i)
    (hz_mass : (∑ i : coord, z i) ≤ 1)
    (hCmin : ∀ i : coord, Cmin ≤ C i)
    (hdual : ∀ k : ℕ, dualRadius k ≤ Umax)
    (hshare : ∀ k : ℕ, (∑ i : coord, share k i) ≤ dualRadius k)
    (hcoord : ∀ k : ℕ, ∀ i : coord,
      x k i ≤ (mass_b / gamma) * share k i + z i * Real.exp (-C i / gamma)) :
    ∀ k : ℕ,
      (∑ i : coord, x k i) ≤
        mass_b * Umax / gamma + d * Real.exp (-Cmin / gamma) := by
  have hfactor_nonneg : 0 ≤ mass_b / gamma := by
    exact div_nonneg hmass_b (le_of_lt hgamma)
  have henvelope : ∀ k : ℕ,
      (∑ i : coord, (mass_b / gamma) * share k i) ≤
        mass_b * dualRadius k / gamma := by
    intro k
    have hmul :
        (mass_b / gamma) * (∑ i : coord, share k i) ≤
          (mass_b / gamma) * dualRadius k :=
      mul_le_mul_of_nonneg_left (hshare k) hfactor_nonneg
    have hsum :
        (∑ i : coord, (mass_b / gamma) * share k i) =
          (mass_b / gamma) * (∑ i : coord, share k i) := by
      simp [Finset.mul_sum]
    calc
      (∑ i : coord, (mass_b / gamma) * share k i)
          = (mass_b / gamma) * (∑ i : coord, share k i) := hsum
      _ ≤ (mass_b / gamma) * dualRadius k := hmul
      _ = mass_b * dualRadius k / gamma := by ring
  exact primalMassBound_explicit_from_normalizedExponentialTail
    (x := x)
    (envelope := fun k i => (mass_b / gamma) * share k i)
    (dualRadius := dualRadius)
    (z := z)
    (C := C)
    (hgamma := hgamma)
    (hmass_b := hmass_b)
    (hcard := hcard)
    (hz_nonneg := hz_nonneg)
    (hz_mass := hz_mass)
    (hCmin := hCmin)
    (hdual := hdual)
    (henvelope := henvelope)
    (hcoord := hcoord)

/--
Dual-ascent certificate version of Proposition `prop:mass-bound-block`.

This is the scalar argument used in the paper proof.  The primal mass is recovered from the
dual identity
`xMass k = (dualPair k - dualValue k) / gamma`, the dual value is nondecreasing from the
zero-start value, the pairing with `b` is controlled by the block quotient radius, and the
zero-start dual value has the explicit cost-floor lower bound.  These four certificates imply the
displayed uniform mass estimate.
-/
theorem primalMassBound_explicit_from_dualAscentCertificate
    {xMass dualValue dualPair dualRadius : ℕ → ℝ}
    {mass_b Umax gamma d Cmin : ℝ}
    (hgamma : 0 < gamma)
    (hmass_b : 0 ≤ mass_b)
    (hdual : ∀ k : ℕ, dualRadius k ≤ Umax)
    (hpair : ∀ k : ℕ, dualPair k ≤ mass_b * dualRadius k)
    (hascent : ∀ k : ℕ, dualValue 0 ≤ dualValue k)
    (hmass_identity : ∀ k : ℕ, xMass k = (dualPair k - dualValue k) / gamma)
    (hzero : -gamma * d * Real.exp (-Cmin / gamma) ≤ dualValue 0) :
    ∀ k : ℕ,
      xMass k ≤ mass_b * Umax / gamma + d * Real.exp (-Cmin / gamma) := by
  intro k
  let tail : ℝ := Real.exp (-Cmin / gamma)
  have hpairU : dualPair k ≤ mass_b * Umax :=
    (hpair k).trans (mul_le_mul_of_nonneg_left (hdual k) hmass_b)
  have hnegValue : -dualValue k ≤ -dualValue 0 :=
    neg_le_neg (hascent k)
  have hnum :
      dualPair k - dualValue k ≤ mass_b * Umax - dualValue 0 := by
    simpa [sub_eq_add_neg] using add_le_add hpairU hnegValue
  have hdiv :
      (dualPair k - dualValue k) / gamma ≤
        (mass_b * Umax - dualValue 0) / gamma :=
    div_le_div_of_nonneg_right hnum (le_of_lt hgamma)
  have hzero_neg : -dualValue 0 ≤ gamma * d * tail := by
    linarith [hzero]
  have hnum_zero :
      mass_b * Umax - dualValue 0 ≤ mass_b * Umax + gamma * d * tail := by
    linarith
  have hdiv_zero :
      (mass_b * Umax - dualValue 0) / gamma ≤
        (mass_b * Umax + gamma * d * tail) / gamma :=
    div_le_div_of_nonneg_right hnum_zero (le_of_lt hgamma)
  have hsplit :
      (mass_b * Umax + gamma * d * tail) / gamma =
        mass_b * Umax / gamma + d * tail := by
    field_simp [hgamma.ne']
  calc
    xMass k = (dualPair k - dualValue k) / gamma := hmass_identity k
    _ ≤ (mass_b * Umax - dualValue 0) / gamma := hdiv
    _ ≤ (mass_b * Umax + gamma * d * tail) / gamma := hdiv_zero
    _ = mass_b * Umax / gamma + d * Real.exp (-Cmin / gamma) := by
      simpa [tail] using hsplit

/--
Dual-ascent mass bound with the zero-start cost floor derived internally.

This strengthens `primalMassBound_explicit_from_dualAscentCertificate`: instead of assuming the
zero-start lower bound on the dual value, it assumes the concrete zero-start formula
`dualValue 0 = -γ * sum_i exp(-C_i/γ)`.  Lean derives the paper's cost-floor estimate from
`Cmin <= C_i`, `card coord <= d`, and monotonicity of the exponential.
-/
theorem primalMassBound_explicit_from_dualAscent_zeroStartCostFloor
    {coord : Type*} [Fintype coord]
    {xMass dualValue dualPair dualRadius : ℕ → ℝ}
    {C : coord → ℝ}
    {mass_b Umax gamma d Cmin : ℝ}
    (hgamma : 0 < gamma)
    (hmass_b : 0 ≤ mass_b)
    (hcard : (Fintype.card coord : ℝ) ≤ d)
    (hCmin : ∀ i : coord, Cmin ≤ C i)
    (hdual : ∀ k : ℕ, dualRadius k ≤ Umax)
    (hpair : ∀ k : ℕ, dualPair k ≤ mass_b * dualRadius k)
    (hascent : ∀ k : ℕ, dualValue 0 ≤ dualValue k)
    (hmass_identity : ∀ k : ℕ, xMass k = (dualPair k - dualValue k) / gamma)
    (hzero_eq : dualValue 0 = -gamma * (∑ i : coord, Real.exp (-C i / gamma))) :
    ∀ k : ℕ,
      xMass k ≤ mass_b * Umax / gamma + d * Real.exp (-Cmin / gamma) := by
  let tail : ℝ := Real.exp (-Cmin / gamma)
  have htail_coord : ∀ i : coord, Real.exp (-C i / gamma) ≤ tail := by
    intro i
    have hneg : -C i ≤ -Cmin := by linarith [hCmin i]
    have hdiv : -C i / gamma ≤ -Cmin / gamma := by
      have hinv_nonneg : 0 ≤ gamma⁻¹ := inv_nonneg.mpr (le_of_lt hgamma)
      have hmul := mul_le_mul_of_nonneg_right hneg hinv_nonneg
      simpa [div_eq_mul_inv] using hmul
    simpa [tail] using Real.exp_le_exp.mpr hdiv
  have hsum_tail :
      (∑ i : coord, Real.exp (-C i / gamma)) ≤
        (Fintype.card coord : ℝ) * tail := by
    calc
      (∑ i : coord, Real.exp (-C i / gamma))
          ≤ ∑ _i : coord, tail := Finset.sum_le_sum fun i _hi => htail_coord i
      _ = (Fintype.card coord : ℝ) * tail := by simp [tail]
  have htail_nonneg : 0 ≤ tail := le_of_lt (Real.exp_pos _)
  have hsum_bound :
      (∑ i : coord, Real.exp (-C i / gamma)) ≤ d * tail :=
    hsum_tail.trans (mul_le_mul_of_nonneg_right hcard htail_nonneg)
  have hmul :
      gamma * (∑ i : coord, Real.exp (-C i / gamma)) ≤ gamma * (d * tail) :=
    mul_le_mul_of_nonneg_left hsum_bound (le_of_lt hgamma)
  have hzero_lower : -gamma * d * tail ≤ dualValue 0 := by
    rw [hzero_eq]
    have hneg := neg_le_neg hmul
    simpa [mul_assoc, mul_left_comm, mul_comm] using hneg
  exact primalMassBound_explicit_from_dualAscentCertificate
    (xMass := xMass)
    (dualValue := dualValue)
    (dualPair := dualPair)
    (dualRadius := dualRadius)
    (mass_b := mass_b)
    (Umax := Umax)
    (gamma := gamma)
    (d := d)
    (Cmin := Cmin)
    hgamma hmass_b hdual hpair hascent hmass_identity hzero_lower

/--
Dual-ascent mass bound from the paper's dual-value formula.

This strengthens `primalMassBound_explicit_from_dualAscent_zeroStartCostFloor`: instead of
assuming the rearranged mass identity, it assumes the direct dual formula
`dualValue k = dualPair k - gamma * xMass k`, corresponding to
`F_gamma(u^(k)) = <b,u^(k)> - gamma * ||x^(k)||_1` in the manuscript.  Lean derives the
division identity using `gamma > 0`, then applies the zero-start cost-floor theorem.
-/
theorem primalMassBound_explicit_from_dualFormula_zeroStartCostFloor
    {coord : Type*} [Fintype coord]
    {xMass dualValue dualPair dualRadius : ℕ → ℝ}
    {C : coord → ℝ}
    {mass_b Umax gamma d Cmin : ℝ}
    (hgamma : 0 < gamma)
    (hmass_b : 0 ≤ mass_b)
    (hcard : (Fintype.card coord : ℝ) ≤ d)
    (hCmin : ∀ i : coord, Cmin ≤ C i)
    (hdual : ∀ k : ℕ, dualRadius k ≤ Umax)
    (hpair : ∀ k : ℕ, dualPair k ≤ mass_b * dualRadius k)
    (hascent : ∀ k : ℕ, dualValue 0 ≤ dualValue k)
    (hdual_formula : ∀ k : ℕ, dualValue k = dualPair k - gamma * xMass k)
    (hzero_eq : dualValue 0 = -gamma * (∑ i : coord, Real.exp (-C i / gamma))) :
    ∀ k : ℕ,
      xMass k ≤ mass_b * Umax / gamma + d * Real.exp (-Cmin / gamma) := by
  have hmass_identity : ∀ k : ℕ, xMass k = (dualPair k - dualValue k) / gamma := by
    intro k
    have hnum : gamma * xMass k = dualPair k - dualValue k := by
      rw [hdual_formula k]
      ring
    calc
      xMass k = (gamma * xMass k) / gamma := by
        field_simp [hgamma.ne']
      _ = (dualPair k - dualValue k) / gamma := by
        rw [hnum]
  exact primalMassBound_explicit_from_dualAscent_zeroStartCostFloor
    (xMass := xMass)
    (dualValue := dualValue)
    (dualPair := dualPair)
    (dualRadius := dualRadius)
    (C := C)
    (mass_b := mass_b)
    (Umax := Umax)
    (gamma := gamma)
    (d := d)
    (Cmin := Cmin)
    hgamma hmass_b hcard hCmin hdual hpair hascent hmass_identity hzero_eq

/--
Dual-ascent mass bound with the quotient-radius pairing estimate derived internally.

This is closer to the proof of Proposition `prop:mass-bound-block`: the theorem now assumes a
finite dual potential `u k`, a gauge shift `shift k`, and the orthogonality and coordinate bound
certificates
`sum_j b_j shift_j = 0` and `|u_j + shift_j| <= dualRadius(k)`.  Lean derives
`dualPair(k) <= ||b||_1 * dualRadius(k)` before applying the scalar mass-bound theorem.
-/
theorem primalMassBound_explicit_from_dualFormula_shiftedPairing_zeroStartCostFloor
    {coord : Type*} [Fintype coord]
    {pot : Type*} [Fintype pot]
    {xMass dualValue dualPair dualRadius : ℕ → ℝ}
    {C : coord → ℝ}
    {b : pot → ℝ} {u shift : ℕ → pot → ℝ}
    {mass_b Umax gamma d Cmin : ℝ}
    (hgamma : 0 < gamma)
    (hmass_b_nonneg : 0 ≤ mass_b)
    (hcard : (Fintype.card coord : ℝ) ≤ d)
    (hCmin : ∀ i : coord, Cmin ≤ C i)
    (hdual : ∀ k : ℕ, dualRadius k ≤ Umax)
    (hradius_nonneg : ∀ k : ℕ, 0 ≤ dualRadius k)
    (hmass_b_l1 : (∑ j : pot, |b j|) ≤ mass_b)
    (hpair_eq : ∀ k : ℕ, dualPair k = ∑ j : pot, b j * u k j)
    (hshift_orth : ∀ k : ℕ, (∑ j : pot, b j * shift k j) = 0)
    (hshift_bound : ∀ k : ℕ, ∀ j : pot, |u k j + shift k j| ≤ dualRadius k)
    (hascent : ∀ k : ℕ, dualValue 0 ≤ dualValue k)
    (hdual_formula : ∀ k : ℕ, dualValue k = dualPair k - gamma * xMass k)
    (hzero_eq : dualValue 0 = -gamma * (∑ i : coord, Real.exp (-C i / gamma))) :
    ∀ k : ℕ,
      xMass k ≤ mass_b * Umax / gamma + d * Real.exp (-Cmin / gamma) := by
  have hpair : ∀ k : ℕ, dualPair k ≤ mass_b * dualRadius k := by
    intro k
    exact shiftedPairing_le_l1_mul_radius
      (b := b)
      (u := u k)
      (shift := shift k)
      (dualPair := dualPair k)
      (mass_b := mass_b)
      (radius := dualRadius k)
      (hradius_nonneg k)
      (hpair_eq k)
      (hshift_orth k)
      (hshift_bound k)
      hmass_b_l1
  exact primalMassBound_explicit_from_dualFormula_zeroStartCostFloor
    (xMass := xMass)
    (dualValue := dualValue)
    (dualPair := dualPair)
    (dualRadius := dualRadius)
    (C := C)
    (mass_b := mass_b)
    (Umax := Umax)
    (gamma := gamma)
    (d := d)
    (Cmin := Cmin)
    hgamma hmass_b_nonneg hcard hCmin hdual hpair hascent hdual_formula hzero_eq

/--
Finite monotonicity induction used in Proposition `prop:mass-bound-block`.

The paper proves dual ascent one step at a time.  This lemma converts the per-step certificate
`dualValue k <= dualValue (k+1)` into the global lower bound `dualValue 0 <= dualValue k` used by
the scalar mass-bound algebra.
-/
theorem dualValue_zero_le_of_stepAscent
    {dualValue : ℕ → ℝ}
    (hstep : ∀ k : ℕ, dualValue k ≤ dualValue (k + 1)) :
    ∀ k : ℕ, dualValue 0 ≤ dualValue k := by
  intro k
  induction k with
  | zero =>
      exact le_rfl
  | succ k ih =>
      exact ih.trans (by simpa [Nat.succ_eq_add_one] using hstep k)

/--
Step-ascent version of Proposition `prop:mass-bound-block`.

Compared with `primalMassBound_explicit_from_dualFormula_shiftedPairing_zeroStartCostFloor`, this
is closer to the iterative proof in the paper: it assumes only the per-step dual-ascent inequality
and derives the global comparison with the zero iterate internally.
-/
theorem primalMassBound_explicit_from_dualFormula_shiftedPairing_stepAscent_zeroStartCostFloor
    {coord : Type*} [Fintype coord]
    {pot : Type*} [Fintype pot]
    {xMass dualValue dualPair dualRadius : ℕ → ℝ}
    {C : coord → ℝ}
    {b : pot → ℝ} {u shift : ℕ → pot → ℝ}
    {mass_b Umax gamma d Cmin : ℝ}
    (hgamma : 0 < gamma)
    (hmass_b_nonneg : 0 ≤ mass_b)
    (hcard : (Fintype.card coord : ℝ) ≤ d)
    (hCmin : ∀ i : coord, Cmin ≤ C i)
    (hdual : ∀ k : ℕ, dualRadius k ≤ Umax)
    (hradius_nonneg : ∀ k : ℕ, 0 ≤ dualRadius k)
    (hmass_b_l1 : (∑ j : pot, |b j|) ≤ mass_b)
    (hpair_eq : ∀ k : ℕ, dualPair k = ∑ j : pot, b j * u k j)
    (hshift_orth : ∀ k : ℕ, (∑ j : pot, b j * shift k j) = 0)
    (hshift_bound : ∀ k : ℕ, ∀ j : pot, |u k j + shift k j| ≤ dualRadius k)
    (hstep_ascent : ∀ k : ℕ, dualValue k ≤ dualValue (k + 1))
    (hdual_formula : ∀ k : ℕ, dualValue k = dualPair k - gamma * xMass k)
    (hzero_eq : dualValue 0 = -gamma * (∑ i : coord, Real.exp (-C i / gamma))) :
    ∀ k : ℕ,
      xMass k ≤ mass_b * Umax / gamma + d * Real.exp (-Cmin / gamma) := by
  have hascent : ∀ k : ℕ, dualValue 0 ≤ dualValue k :=
    dualValue_zero_le_of_stepAscent (dualValue := dualValue) hstep_ascent
  exact primalMassBound_explicit_from_dualFormula_shiftedPairing_zeroStartCostFloor
    (xMass := xMass)
    (dualValue := dualValue)
    (dualPair := dualPair)
    (dualRadius := dualRadius)
    (C := C)
    (b := b)
    (u := u)
    (shift := shift)
    (mass_b := mass_b)
    (Umax := Umax)
    (gamma := gamma)
    (d := d)
    (Cmin := Cmin)
    hgamma hmass_b_nonneg hcard hCmin hdual hradius_nonneg hmass_b_l1 hpair_eq
    hshift_orth hshift_bound hascent hdual_formula hzero_eq

/--
Displayed-objective version of Proposition `prop:mass-bound-block`.

This endpoint removes the auxiliary `dualValue` sequence from the paper-facing statement.  The
dual objective value is the displayed formula `dualPair k - gamma * xMass k`, so per-step ascent
and the zero-start identity are stated directly for that expression.
-/
theorem primalMassBound_from_displayedDualObjective_stepAscent
    {coord : Type*} [Fintype coord]
    {pot : Type*} [Fintype pot] [Nonempty pot]
    {xMass dualPair dualRadius : ℕ → ℝ}
    {C : coord → ℝ}
    {b : pot → ℝ} {u shift : ℕ → pot → ℝ}
    {mass_b Umax gamma d Cmin : ℝ}
    (hgamma : 0 < gamma)
    (hcard : (Fintype.card coord : ℝ) ≤ d)
    (hCmin : ∀ i : coord, Cmin ≤ C i)
    (hdual : ∀ k : ℕ, dualRadius k ≤ Umax)
    (hmass_b_l1 : (∑ j : pot, |b j|) ≤ mass_b)
    (hpair_eq : ∀ k : ℕ, dualPair k = ∑ j : pot, b j * u k j)
    (hshift_orth : ∀ k : ℕ, (∑ j : pot, b j * shift k j) = 0)
    (hshift_bound : ∀ k : ℕ, ∀ j : pot, |u k j + shift k j| ≤ dualRadius k)
    (hstep_ascent : ∀ k : ℕ,
      dualPair k - gamma * xMass k ≤ dualPair (k + 1) - gamma * xMass (k + 1))
    (hzero_eq :
      dualPair 0 - gamma * xMass 0 =
        -gamma * (∑ i : coord, Real.exp (-C i / gamma))) :
    ∀ k : ℕ,
      xMass k ≤ mass_b * Umax / gamma + d * Real.exp (-Cmin / gamma) := by
  have hmass_b_nonneg : 0 ≤ mass_b :=
    massBound_nonneg_of_l1_bound (b := b) hmass_b_l1
  have hradius_nonneg : ∀ k : ℕ, 0 ≤ dualRadius k := by
    intro k
    exact
      radius_nonneg_of_abs_bound
        (v := fun j : pot => u k j + shift k j)
        (radius := dualRadius k)
        (hshift_bound k)
  exact
    primalMassBound_explicit_from_dualFormula_shiftedPairing_stepAscent_zeroStartCostFloor
      (xMass := xMass)
      (dualValue := fun k => dualPair k - gamma * xMass k)
      (dualPair := dualPair)
      (dualRadius := dualRadius)
      (C := C)
      (b := b)
      (u := u)
      (shift := shift)
      (mass_b := mass_b)
      (Umax := Umax)
      (gamma := gamma)
      (d := d)
      (Cmin := Cmin)
      hgamma hmass_b_nonneg hcard hCmin hdual hradius_nonneg hmass_b_l1 hpair_eq
      hshift_orth hshift_bound hstep_ascent (by intro k; rfl) hzero_eq

/--
Finite-pairing displayed-objective version of Proposition `prop:mass-bound-block`.

This is the current paper-facing endpoint.  It removes both auxiliary objective containers:
there is no abstract `dualValue` sequence and no separate `dualPair` sequence.  The ascent and
zero-start hypotheses are stated directly for the finite displayed objective
`sum_j b_j u_k(j) - gamma * xMass(k)`.
-/
theorem primalMassBound_from_displayedFinitePairing_stepAscent
    {coord : Type*} [Fintype coord]
    {pot : Type*} [Fintype pot] [Nonempty pot]
    {xMass dualRadius : ℕ → ℝ}
    {C : coord → ℝ}
    {b : pot → ℝ} {u shift : ℕ → pot → ℝ}
    {mass_b Umax gamma d Cmin : ℝ}
    (hgamma : 0 < gamma)
    (hcard : (Fintype.card coord : ℝ) ≤ d)
    (hCmin : ∀ i : coord, Cmin ≤ C i)
    (hdual : ∀ k : ℕ, dualRadius k ≤ Umax)
    (hmass_b_l1 : (∑ j : pot, |b j|) ≤ mass_b)
    (hshift_orth : ∀ k : ℕ, (∑ j : pot, b j * shift k j) = 0)
    (hshift_bound : ∀ k : ℕ, ∀ j : pot, |u k j + shift k j| ≤ dualRadius k)
    (hstep_ascent : ∀ k : ℕ,
      (∑ j : pot, b j * u k j) - gamma * xMass k ≤
        (∑ j : pot, b j * u (k + 1) j) - gamma * xMass (k + 1))
    (hzero_eq :
      (∑ j : pot, b j * u 0 j) - gamma * xMass 0 =
        -gamma * (∑ i : coord, Real.exp (-C i / gamma))) :
    ∀ k : ℕ,
      xMass k ≤ mass_b * Umax / gamma + d * Real.exp (-Cmin / gamma) := by
  have hmass_b_nonneg : 0 ≤ mass_b :=
    massBound_nonneg_of_l1_bound (b := b) hmass_b_l1
  have hradius_nonneg : ∀ k : ℕ, 0 ≤ dualRadius k := by
    intro k
    exact
      radius_nonneg_of_abs_bound
        (v := fun j : pot => u k j + shift k j)
        (radius := dualRadius k)
        (hshift_bound k)
  exact
    primalMassBound_explicit_from_dualFormula_shiftedPairing_stepAscent_zeroStartCostFloor
      (xMass := xMass)
      (dualValue := fun k => (∑ j : pot, b j * u k j) - gamma * xMass k)
      (dualPair := fun k => ∑ j : pot, b j * u k j)
      (dualRadius := dualRadius)
      (C := C)
      (b := b)
      (u := u)
      (shift := shift)
      (mass_b := mass_b)
      (Umax := Umax)
      (gamma := gamma)
      (d := d)
      (Cmin := Cmin)
      hgamma hmass_b_nonneg hcard hCmin hdual hradius_nonneg hmass_b_l1
      (by intro k; rfl)
      hshift_orth hshift_bound hstep_ascent (by intro k; rfl) hzero_eq

/--
Exact-constant finite-pairing version of Proposition `prop:mass-bound-block`.

This endpoint uses the constants as they appear in the paper: the exact finite `l1` mass
`sum_j |b_j|` and the exact coordinate cardinality `card(coord)`.  Thus the paper-facing
statement no longer carries an abstract mass upper bound `mass_b` or an abstract dimension
upper bound `d`; Lean instantiates the previous scalar proof with these exact finite quantities.
-/
theorem primalMassBound_from_displayedFinitePairing_exactL1_card_stepAscent
    {coord : Type*} [Fintype coord]
    {pot : Type*} [Fintype pot] [Nonempty pot]
    {xMass dualRadius : ℕ → ℝ}
    {C : coord → ℝ}
    {b : pot → ℝ} {u shift : ℕ → pot → ℝ}
    {Umax gamma Cmin : ℝ}
    (hgamma : 0 < gamma)
    (hCmin : ∀ i : coord, Cmin ≤ C i)
    (hdual : ∀ k : ℕ, dualRadius k ≤ Umax)
    (hshift_orth : ∀ k : ℕ, (∑ j : pot, b j * shift k j) = 0)
    (hshift_bound : ∀ k : ℕ, ∀ j : pot, |u k j + shift k j| ≤ dualRadius k)
    (hstep_ascent : ∀ k : ℕ,
      (∑ j : pot, b j * u k j) - gamma * xMass k ≤
        (∑ j : pot, b j * u (k + 1) j) - gamma * xMass (k + 1))
    (hzero_eq :
      (∑ j : pot, b j * u 0 j) - gamma * xMass 0 =
        -gamma * (∑ i : coord, Real.exp (-C i / gamma))) :
    ∀ k : ℕ,
      xMass k ≤
        (∑ j : pot, |b j|) * Umax / gamma +
          (Fintype.card coord : ℝ) * Real.exp (-Cmin / gamma) := by
  let mass_b : ℝ := ∑ j : pot, |b j|
  have hmass_b_nonneg : 0 ≤ mass_b := by
    exact Finset.sum_nonneg fun j _hj => abs_nonneg (b j)
  have hradius_nonneg : ∀ k : ℕ, 0 ≤ dualRadius k := by
    intro k
    exact
      radius_nonneg_of_abs_bound
        (v := fun j : pot => u k j + shift k j)
        (radius := dualRadius k)
        (hshift_bound k)
  simpa [mass_b] using
    primalMassBound_explicit_from_dualFormula_shiftedPairing_stepAscent_zeroStartCostFloor
      (xMass := xMass)
      (dualValue := fun k => (∑ j : pot, b j * u k j) - gamma * xMass k)
      (dualPair := fun k => ∑ j : pot, b j * u k j)
      (dualRadius := dualRadius)
      (C := C)
      (b := b)
      (u := u)
      (shift := shift)
      (mass_b := mass_b)
      (Umax := Umax)
      (gamma := gamma)
      (d := (Fintype.card coord : ℝ))
      (Cmin := Cmin)
      hgamma hmass_b_nonneg le_rfl hCmin hdual hradius_nonneg le_rfl
      (by intro k; rfl)
      hshift_orth hshift_bound hstep_ascent (by intro k; rfl) hzero_eq

/--
Zero-start exact-constant version of Proposition `prop:mass-bound-block`.

This is a refinement of `primalMassBound_from_displayedFinitePairing_exactL1_card_stepAscent`.
Instead of assuming the zero-start objective identity as one opaque equation, it assumes the two
primitive zero-start facts used in the paper proof: the initial dual potential is zero and the
initial primal mass is the displayed finite Gibbs mass.  Lean derives the objective identity
`sum_j b_j u_0(j) - gamma * xMass(0) = -gamma * sum_i exp(-C_i/gamma)` internally.
-/
theorem primalMassBound_from_zeroStartFinitePairing_exactL1_card_stepAscent
    {coord : Type*} [Fintype coord]
    {pot : Type*} [Fintype pot] [Nonempty pot]
    {xMass dualRadius : ℕ → ℝ}
    {C : coord → ℝ}
    {b : pot → ℝ} {u shift : ℕ → pot → ℝ}
    {Umax gamma Cmin : ℝ}
    (hgamma : 0 < gamma)
    (hCmin : ∀ i : coord, Cmin ≤ C i)
    (hdual : ∀ k : ℕ, dualRadius k ≤ Umax)
    (hshift_orth : ∀ k : ℕ, (∑ j : pot, b j * shift k j) = 0)
    (hshift_bound : ∀ k : ℕ, ∀ j : pot, |u k j + shift k j| ≤ dualRadius k)
    (hstep_ascent : ∀ k : ℕ,
      (∑ j : pot, b j * u k j) - gamma * xMass k ≤
        (∑ j : pot, b j * u (k + 1) j) - gamma * xMass (k + 1))
    (hzero_start :
      (∀ j : pot, u 0 j = 0) ∧
        xMass 0 = ∑ i : coord, Real.exp (-C i / gamma)) :
    ∀ k : ℕ,
      xMass k ≤
        (∑ j : pot, |b j|) * Umax / gamma +
          (Fintype.card coord : ℝ) * Real.exp (-Cmin / gamma) := by
  have hpair_zero : (∑ j : pot, b j * u 0 j) = 0 := by
    simp [hzero_start.1]
  have hzero_eq :
      (∑ j : pot, b j * u 0 j) - gamma * xMass 0 =
        -gamma * (∑ i : coord, Real.exp (-C i / gamma)) := by
    rw [hpair_zero, hzero_start.2]
    ring
  exact
    primalMassBound_from_displayedFinitePairing_exactL1_card_stepAscent
      (xMass := xMass)
      (dualRadius := dualRadius)
      (C := C)
      (b := b)
      (u := u)
      (shift := shift)
      (Umax := Umax)
      (gamma := gamma)
      (Cmin := Cmin)
      hgamma hCmin hdual hshift_orth hshift_bound hstep_ascent hzero_eq

/--
Uniform-shift-bound exact-constant version of Proposition `prop:mass-bound-block`.

This is the current strongest paper-facing endpoint for Proposition 4.2.  It removes the
auxiliary sequence `dualRadius k` from the statement: the paper only needs the uniform
representative bound by `Umax`, so Lean instantiates the previous radius-sequence theorem with
the constant radius `dualRadius k = Umax`.
-/
theorem primalMassBound_from_zeroStartFinitePairing_exactL1_card_uniformShiftBound
    {coord : Type*} [Fintype coord]
    {pot : Type*} [Fintype pot] [Nonempty pot]
    {xMass : ℕ → ℝ}
    {C : coord → ℝ}
    {b : pot → ℝ} {u shift : ℕ → pot → ℝ}
    {Umax gamma Cmin : ℝ}
    (hgamma : 0 < gamma)
    (hCmin : ∀ i : coord, Cmin ≤ C i)
    (hshift_orth : ∀ k : ℕ, (∑ j : pot, b j * shift k j) = 0)
    (hshift_bound : ∀ k : ℕ, ∀ j : pot, |u k j + shift k j| ≤ Umax)
    (hstep_ascent : ∀ k : ℕ,
      (∑ j : pot, b j * u k j) - gamma * xMass k ≤
        (∑ j : pot, b j * u (k + 1) j) - gamma * xMass (k + 1))
    (hzero_start :
      (∀ j : pot, u 0 j = 0) ∧
        xMass 0 = ∑ i : coord, Real.exp (-C i / gamma)) :
    ∀ k : ℕ,
      xMass k ≤
        (∑ j : pot, |b j|) * Umax / gamma +
          (Fintype.card coord : ℝ) * Real.exp (-Cmin / gamma) := by
  exact
    primalMassBound_from_zeroStartFinitePairing_exactL1_card_stepAscent
      (xMass := xMass)
      (dualRadius := fun _ => Umax)
      (C := C)
      (b := b)
      (u := u)
      (shift := shift)
      (Umax := Umax)
      (gamma := gamma)
      (Cmin := Cmin)
      hgamma hCmin
      (by intro k; exact le_rfl)
      hshift_orth hshift_bound hstep_ascent hzero_start

/--
Existential-representative version of Proposition `prop:mass-bound-block`.

This endpoint matches the quotient-norm reading more closely than
`primalMassBound_from_zeroStartFinitePairing_exactL1_card_uniformShiftBound`: the paper needs a
bounded representative of each potential class, not an externally named global shift sequence.
Lean chooses the representative shifts from the existential certificate and then applies the
uniform-shift proof above.
-/
theorem primalMassBound_from_zeroStartFinitePairing_exactL1_card_shiftExists
    {coord : Type*} [Fintype coord]
    {pot : Type*} [Fintype pot] [Nonempty pot]
    {xMass : ℕ → ℝ}
    {C : coord → ℝ}
    {b : pot → ℝ} {u : ℕ → pot → ℝ}
    {Umax gamma Cmin : ℝ}
    (hgamma : 0 < gamma)
    (hCmin : ∀ i : coord, Cmin ≤ C i)
    (hshift_cert : ∀ k : ℕ, ∃ shift : pot → ℝ,
      (∑ j : pot, b j * shift j) = 0 ∧
        ∀ j : pot, |u k j + shift j| ≤ Umax)
    (hstep_ascent : ∀ k : ℕ,
      (∑ j : pot, b j * u k j) - gamma * xMass k ≤
        (∑ j : pot, b j * u (k + 1) j) - gamma * xMass (k + 1))
    (hzero_start :
      (∀ j : pot, u 0 j = 0) ∧
        xMass 0 = ∑ i : coord, Real.exp (-C i / gamma)) :
    ∀ k : ℕ,
      xMass k ≤
        (∑ j : pot, |b j|) * Umax / gamma +
          (Fintype.card coord : ℝ) * Real.exp (-Cmin / gamma) := by
  classical
  let shift : ℕ → pot → ℝ := fun k => Classical.choose (hshift_cert k)
  have hshift_orth : ∀ k : ℕ, (∑ j : pot, b j * shift k j) = 0 := by
    intro k
    simpa [shift] using (Classical.choose_spec (hshift_cert k)).1
  have hshift_bound : ∀ k : ℕ, ∀ j : pot, |u k j + shift k j| ≤ Umax := by
    intro k j
    simpa [shift] using (Classical.choose_spec (hshift_cert k)).2 j
  exact
    primalMassBound_from_zeroStartFinitePairing_exactL1_card_uniformShiftBound
      (xMass := xMass)
      (C := C)
      (b := b)
      (u := u)
      (shift := shift)
      (Umax := Umax)
      (gamma := gamma)
      (Cmin := Cmin)
      hgamma hCmin hshift_orth hshift_bound hstep_ascent hzero_start

/--
Vocabulary-certificate version of Proposition `prop:mass-bound-block`.

This endpoint exposes the assumptions with the same conceptual names as the paper:
`CostLowerBound`, `FiniteQuotientRadiusBound`, `DisplayedFinitePairingAscent`, and
`ZeroStartPrimalMass`.  Lean unfolds these proof-free statement predicates, chooses the finite
quotient representatives, derives the zero-start objective identity, and proves the mass estimate
with the exact constants from the paper.
-/
theorem primalMassBound_from_zeroStartFinitePairing_exactL1_card_quotientRadiusCertificate
    {coord : Type*} [Fintype coord]
    {pot : Type*} [Fintype pot] [Nonempty pot]
    {xMass : ℕ → ℝ}
    {C : coord → ℝ}
    {b : pot → ℝ} {u : ℕ → pot → ℝ}
    {Umax gamma Cmin : ℝ}
    (hgamma : 0 < gamma)
    (hCmin : CostLowerBound C Cmin)
    (hradius : FiniteQuotientRadiusBound b u Umax)
    (hascent : DisplayedFinitePairingAscent b u xMass gamma)
    (hzero_start : ZeroStartPrimalMass C u xMass gamma) :
    ∀ k : ℕ,
      xMass k ≤
        (∑ j : pot, |b j|) * Umax / gamma +
          (Fintype.card coord : ℝ) * Real.exp (-Cmin / gamma) := by
  classical
  let shift : ℕ → pot → ℝ := fun k => Classical.choose (hradius k)
  have hshift_orth : ∀ k : ℕ, (∑ j : pot, b j * shift k j) = 0 := by
    intro k
    simpa [FiniteQuotientRadiusBound, shift] using (Classical.choose_spec (hradius k)).1
  have hshift_bound : ∀ k : ℕ, ∀ j : pot, |u k j + shift k j| ≤ Umax := by
    intro k j
    simpa [FiniteQuotientRadiusBound, shift] using (Classical.choose_spec (hradius k)).2 j
  have hpair_zero : (∑ j : pot, b j * u 0 j) = 0 := by
    simp [hzero_start.1]
  have hzero_eq :
      (∑ j : pot, b j * u 0 j) - gamma * xMass 0 =
        -gamma * (∑ i : coord, Real.exp (-C i / gamma)) := by
    rw [hpair_zero, hzero_start.2]
    ring
  exact
    primalMassBound_from_displayedFinitePairing_exactL1_card_stepAscent
      (xMass := xMass)
      (dualRadius := fun _ => Umax)
      (C := C)
      (b := b)
      (u := u)
      (shift := shift)
      (Umax := Umax)
      (gamma := gamma)
      (Cmin := Cmin)
      hgamma hCmin
      (by intro k; exact le_rfl)
      hshift_orth hshift_bound hascent hzero_eq

/--
Structured-certificate version of Proposition `prop:mass-bound-block`.

This paper-facing endpoint exposes one named finite certificate instead of
separate raw hypotheses.  Lean unfolds the certificate and reuses the certified
quotient-radius proof above, so the remaining semantic bridge is exactly the
construction of `PrimalMassBoundBlockCertificate` from the concrete KL
block-update dynamics.
-/
theorem primalMassBound_from_blockCertificate
    {coord : Type*} [Fintype coord]
    {pot : Type*} [Fintype pot] [Nonempty pot]
    {xMass : ℕ → ℝ}
    {C : coord → ℝ}
    {b : pot → ℝ} {u : ℕ → pot → ℝ}
    {Umax gamma Cmin : ℝ}
    (hcert : PrimalMassBoundBlockCertificate xMass C b u Umax gamma Cmin) :
    ∀ k : ℕ,
      xMass k ≤
        (∑ j : pot, |b j|) * Umax / gamma +
          (Fintype.card coord : ℝ) * Real.exp (-Cmin / gamma) := by
  exact
    primalMassBound_from_zeroStartFinitePairing_exactL1_card_quotientRadiusCertificate
      (xMass := xMass)
      (C := C)
      (b := b)
      (u := u)
      (Umax := Umax)
      (gamma := gamma)
      (Cmin := Cmin)
      hcert.gamma_positive
      hcert.cost_lower_bound
      hcert.quotient_radius_bound
      hcert.displayed_pairing_ascent
      hcert.zero_start_primal_mass

/--
Paper-facing one-line corollary when the dual radius is already bounded by `U`.
-/
theorem primalMassBound_fromDualRadius
    {U X : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    (hprimal_at_U : X ≤ Xbound U) :
    X ≤ Xbound U :=
  primalBound_fromDualBound hmono hprimal_at_U (le_rfl : U ≤ U)

/--
Primal confinement from an orbit upper bound.

This is the single-iterate specialisation of the primal transfer principle:
if `X ≤ Xbound orbit_k` holds for the current iterate seminorm value `orbit_k`,
and `orbit_k ≤ U_max` holds by some uniform orbit bound, then `X ≤ Xbound U_max`.

This is the form in which `prop:mass-bound-block` is applied: one substitutes the uniform
orbit estimate `p u_k ≤ U_max` and monotonicity of `Xbound` promotes the per-iterate control
to the uniform orbit budget.
-/
theorem primalBound_from_dualOrbitBound
    {X U_max : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {orbit_k : ℝ}
    (horbit : orbit_k ≤ U_max)
    (hprimal : X ≤ Xbound orbit_k) :
    X ≤ Xbound U_max :=
  primalBound_fromDualBound hmono hprimal horbit

/--
Uniform primal confinement from a pointwise orbit bound.

If every value in a sequence `orbit` is bounded by `U_max`, and `X ≤ Xbound (orbit k)` holds
for some index `k`, then `X ≤ Xbound U_max` for that same index.

This packages the typical usage pattern in Section 4: one has a uniform bound
`∀ k, orbit k ≤ U_max` (from the non-expansive iterate estimate) and a primal estimate that
holds at each iterate norm value; the conclusion is the primal bound evaluated at the budget.
-/
theorem primalBound_all_iterates
    {X U_max : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {orbit : ℕ → ℝ}
    (horbit : ∀ k, orbit k ≤ U_max)
    {k : ℕ}
    (hprimal_k : X ≤ Xbound (orbit k)) :
    X ≤ Xbound U_max :=
  primalBound_from_dualOrbitBound hmono (horbit k) hprimal_k

/--
Primal confinement through two chained dual bounds.

If `X ≤ Xbound d`, `d ≤ U1`, and `U1 ≤ U2`, then `X ≤ Xbound U2`.

This three-step chain is common in applications where a dual radius is first bounded by an
intermediate quantity and then the intermediate quantity is bounded by the final budget constant.
-/
theorem primalBound_trans
    {d U1 U2 X : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    (hprimal_at_d : X ≤ Xbound d)
    (hd_U1 : d ≤ U1)
    (hU1_U2 : U1 ≤ U2) :
    X ≤ Xbound U2 :=
  primalBound_fromDualBound hmono hprimal_at_d (hd_U1.trans hU1_U2)

/--
Primal confinement from a pointwise dual `O(1/k)` bound.

If the iterate-level dual quantity `gap k` satisfies a rate bound
`gap k ≤ C / (k + 1)`, then any primal estimate stated at `gap k`
transfers to the explicit rate envelope `C / (k + 1)`.
-/
theorem primalBound_from_dualRateBound
    {X C : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {k : ℕ}
    (hprimal_k : X ≤ Xbound (gap k))
    (hrate_k : gap k ≤ C / (k + 1 : ℝ)) :
    X ≤ Xbound (C / (k + 1 : ℝ)) :=
  primalBound_fromDualBound hmono hprimal_k hrate_k

/--
Primal confinement from a pointwise dual rate plus a target threshold.

This is the direct optimization-ready bridge:
`gap k ≤ C/(k+1)` and `C/(k+1) ≤ eps` imply `X ≤ Xbound eps`
whenever `X ≤ Xbound (gap k)` holds.
-/
theorem primalBound_from_dualRate_threshold
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {k : ℕ}
    (hprimal_k : X ≤ Xbound (gap k))
    (hrate_k : gap k ≤ C / (k + 1 : ℝ))
    (hthreshold : C / (k + 1 : ℝ) ≤ eps) :
    X ≤ Xbound eps :=
  primalBound_trans hmono hprimal_k hrate_k hthreshold

/--
Primal confinement from a pointwise dual rate and ratio-form stopping rule.

If `C/eps ≤ k+1` (with `eps > 0`), this theorem derives the threshold
`C/(k+1) ≤ eps` and then applies `primalBound_from_dualRate_threshold`.
-/
theorem primalBound_from_dualRate_ratioBound
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {k : ℕ}
    (hprimal_k : X ≤ Xbound (gap k))
    (hrate_k : gap k ≤ C / (k + 1 : ℝ))
    (heps : 0 < eps)
    (hratio : C / eps ≤ (k + 1 : ℝ)) :
    X ≤ Xbound eps := by
  have hthreshold' : C ≤ (k + 1 : ℝ) * eps := (div_le_iff₀ heps).1 hratio
  have hthreshold : C / (k + 1 : ℝ) ≤ eps := by
    have hpos : 0 < (k + 1 : ℝ) := by
      exact_mod_cast Nat.succ_pos k
    have hthreshold'' : C ≤ eps * (k + 1 : ℝ) := by
      simpa [mul_comm] using hthreshold'
    exact (div_le_iff₀ hpos).2 hthreshold''
  exact primalBound_from_dualRate_threshold hmono hprimal_k hrate_k hthreshold

/--
Primal confinement from a pointwise dual rate and a natural-number ratio bound.

This is the `ℕ`-valued variant of `primalBound_from_dualRate_ratioBound`,
useful when iteration proofs provide `C/eps ≤ k+1` directly in naturals.
-/
theorem primalBound_from_dualRate_ratioBound_of_natBound
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {k : ℕ}
    (hprimal_k : X ≤ Xbound (gap k))
    (hrate_k : gap k ≤ C / (k + 1 : ℝ))
    (heps : 0 < eps)
    (hratio_nat : C / eps ≤ (k + 1 : ℕ)) :
    X ≤ Xbound eps := by
  have hratio : C / eps ≤ (k + 1 : ℝ) := by
    exact_mod_cast hratio_nat
  exact primalBound_from_dualRate_ratioBound
    (hmono := hmono) (hprimal_k := hprimal_k) (hrate_k := hrate_k) (heps := heps) hratio

/--
Primal confinement from a pointwise dual rate and ceiling-form stopping rule.

This matches paper-facing iteration complexity statements where one controls
`Nat.ceil (C/eps)` and needs a direct primal conclusion.
-/
theorem primalBound_from_dualRate_closedFormCeil
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {k : ℕ}
    (hprimal_k : X ≤ Xbound (gap k))
    (hrate_k : gap k ≤ C / (k + 1 : ℝ))
    (heps : 0 < eps)
    (hk : Nat.ceil (C / eps) ≤ k + 1) :
    X ≤ Xbound eps := by
  have hratio : C / eps ≤ (k + 1 : ℝ) := by
    simpa [Nat.cast_add, Nat.cast_one] using (Nat.le_of_ceil_le hk : C / eps ≤ (k + 1 : ℕ))
  exact primalBound_from_dualRate_ratioBound hmono hprimal_k hrate_k heps hratio

/--
Ceiling-form primal confinement through an intermediate natural-number budget.

This wrapper is convenient when complexity arguments first prove
`Nat.ceil (C/eps) ≤ N` and only afterwards relate `N ≤ k+1`.
-/
theorem primalBound_from_dualRate_closedFormCeil_of_natBound
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {k N : ℕ}
    (hprimal_k : X ≤ Xbound (gap k))
    (hrate_k : gap k ≤ C / (k + 1 : ℝ))
    (heps : 0 < eps)
    (hk_nat : Nat.ceil (C / eps) ≤ N)
    (hN : N ≤ k + 1) :
    X ≤ Xbound eps :=
  primalBound_from_dualRate_closedFormCeil
    hmono hprimal_k hrate_k heps (hk_nat.trans hN)

/--
Closed-form-ceil dual-rate transfer with an index comparison helper.

If the ceiling condition is known at index `m` and `m ≤ n`, this upgrades directly to `n`.
-/
theorem primalBound_from_dualRate_closedFormCeil_of_le_index
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n : ℕ}
    (hprimal_n : X ≤ Xbound (gap n))
    (hrate_n : gap n ≤ C / (n + 1 : ℝ))
    (heps : 0 < eps)
    (hk : Nat.ceil (C / eps) ≤ m + 1)
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hn : Nat.ceil (C / eps) ≤ n + 1 := by
    exact hk.trans (Nat.add_le_add_right hmn 1)
  exact primalBound_from_dualRate_closedFormCeil
    hmono hprimal_n hrate_n heps hn

/-! ### Successor-index dual-rate threshold bridges -/

/--
Successor-index dual-rate threshold transfer.

This is the `(k+1)` iterate convenience wrapper for
`primalBound_from_dualRate_threshold`.
-/
theorem primalBound_from_dualRate_threshold_succ
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {k : ℕ}
    (hprimal_k1 : X ≤ Xbound (gap (k + 1)))
    (hrate_k1 : gap (k + 1) ≤ C / ((k + 1 : ℝ) + 1))
    (hthreshold : C / ((k + 1 : ℝ) + 1) ≤ eps) :
    X ≤ Xbound eps := by
  exact primalBound_fromDualBound hmono hprimal_k1 (hrate_k1.trans hthreshold)

/--
Successor-index dual-rate ratio-form transfer.
-/
theorem primalBound_from_dualRate_ratioBound_succ
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {k : ℕ}
    (hprimal_k1 : X ≤ Xbound (gap (k + 1)))
    (hrate_k1 : gap (k + 1) ≤ C / ((k + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hratio : C / eps ≤ (k + 1 : ℝ) + 1) :
    X ≤ Xbound eps := by
  have hthreshold' : C ≤ ((k + 1 : ℝ) + 1) * eps := (div_le_iff₀ heps).1 hratio
  have hpos : 0 < ((k + 1 : ℝ) + 1) := by positivity
  have hthreshold : C / ((k + 1 : ℝ) + 1) ≤ eps := by
    have hthreshold'' : C ≤ eps * ((k + 1 : ℝ) + 1) := by
      simpa [mul_comm] using hthreshold'
    exact (div_le_iff₀ hpos).2 hthreshold''
  exact primalBound_from_dualRate_threshold_succ hmono hprimal_k1 hrate_k1 hthreshold

/--
Successor-index dual-rate ratio transfer from a natural-number bound.

This variant is convenient when complexity proofs provide the index bound in `ℕ`.
-/
theorem primalBound_from_dualRate_ratioBound_succ_of_natBound
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {k : ℕ}
    (hprimal_k1 : X ≤ Xbound (gap (k + 1)))
    (hrate_k1 : gap (k + 1) ≤ C / ((k + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hratio_nat : C / eps ≤ ((k + 1) + 1 : ℕ)) :
    X ≤ Xbound eps := by
  have hratio' : C / eps ≤ (((k + 1) + 1 : ℕ) : ℝ) := by
    exact_mod_cast hratio_nat
  have hratio : C / eps ≤ (k + 1 : ℝ) + 1 := by
    simpa [Nat.cast_add, Nat.cast_one] using hratio'
  exact primalBound_from_dualRate_ratioBound_succ
    (hmono := hmono) (hprimal_k1 := hprimal_k1) (hrate_k1 := hrate_k1) (heps := heps) hratio

/--
Successor-index dual-rate ceiling-form transfer.
-/
theorem primalBound_from_dualRate_closedFormCeil_succ
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {k : ℕ}
    (hprimal_k1 : X ≤ Xbound (gap (k + 1)))
    (hrate_k1 : gap (k + 1) ≤ C / ((k + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hk : Nat.ceil (C / eps) ≤ (k + 1) + 1) :
    X ≤ Xbound eps := by
  have hratio_nat : C / eps ≤ ((k + 1) + 1 : ℕ) :=
    Nat.le_of_ceil_le hk
  exact primalBound_from_dualRate_ratioBound_succ_of_natBound
    (hmono := hmono) (hprimal_k1 := hprimal_k1) (hrate_k1 := hrate_k1)
    (heps := heps) hratio_nat

/--
Successor-index closed-form-ceil dual-rate transfer via an intermediate natural bound.
-/
theorem primalBound_from_dualRate_closedFormCeil_succ_of_natBound
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {k N : ℕ}
    (hprimal_k1 : X ≤ Xbound (gap (k + 1)))
    (hrate_k1 : gap (k + 1) ≤ C / ((k + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hk_nat : Nat.ceil (C / eps) ≤ N)
    (hN : N ≤ (k + 1) + 1) :
    X ≤ Xbound eps :=
  primalBound_from_dualRate_closedFormCeil_succ
    hmono hprimal_k1 hrate_k1 heps (hk_nat.trans hN)

/-! ## Bridge from explicit `α*B/(n+1)` rate assumptions -/

/--
Primal confinement from an explicit dual rate envelope `α*B/(n+1)`.
-/
theorem primalBound_from_explicitRateBound
    {X alpha B : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n : ℕ}
    (hprimal_n : X ≤ Xbound (gap n))
    (hrate_n : gap n ≤ (alpha * B) / (n + 1 : ℝ)) :
    X ≤ Xbound ((alpha * B) / (n + 1 : ℝ)) :=
  primalBound_from_dualRateBound hmono hprimal_n hrate_n

/--
Explicit dual rate envelope plus budget threshold gives primal budget confinement.
-/
theorem primalBound_from_explicitRate_budget
    {X alpha B budget : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n : ℕ}
    (hprimal_n : X ≤ Xbound (gap n))
    (hrate_n : gap n ≤ (alpha * B) / (n + 1 : ℝ))
    (hbudget : (alpha * B) / (n + 1 : ℝ) ≤ budget) :
    X ≤ Xbound budget :=
  primalBound_from_dualRate_threshold hmono hprimal_n hrate_n hbudget

/--
Explicit dual rate envelope plus multiplicative threshold gives primal budget confinement.

This variant accepts the threshold in product form
`alpha * B ≤ budget * (n+1)`, often produced by algebraic manipulations in
complexity proofs before a division step.
-/
theorem primalBound_from_explicitRate_budget_of_mulThreshold
    {X alpha B budget : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n : ℕ}
    (hprimal_n : X ≤ Xbound (gap n))
    (hrate_n : gap n ≤ (alpha * B) / (n + 1 : ℝ))
    (hbudget_mul : alpha * B ≤ budget * (n + 1 : ℝ)) :
    X ≤ Xbound budget := by
  have hpos : 0 < (n + 1 : ℝ) := by
    exact_mod_cast Nat.succ_pos n
  have hbudget : (alpha * B) / (n + 1 : ℝ) ≤ budget := by
    exact (div_le_iff₀ hpos).2 hbudget_mul
  exact primalBound_from_explicitRate_budget hmono hprimal_n hrate_n hbudget

/--
Explicit dual rate envelope plus ratio-form stopping rule gives primal `eps`-confinement.
-/
theorem primalBound_from_explicitRate_ratioBound
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n : ℕ}
    (hprimal_n : X ≤ Xbound (gap n))
    (hrate_n : gap n ≤ (alpha * B) / (n + 1 : ℝ))
    (heps : 0 < eps)
    (hratio : (alpha * B) / eps ≤ (n + 1 : ℝ)) :
    X ≤ Xbound eps :=
  primalBound_from_dualRate_ratioBound hmono hprimal_n hrate_n heps hratio

/--
Explicit-rate ratio-form transfer from a natural-number bound.
-/
theorem primalBound_from_explicitRate_ratioBound_of_natBound
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n : ℕ}
    (hprimal_n : X ≤ Xbound (gap n))
    (hrate_n : gap n ≤ (alpha * B) / (n + 1 : ℝ))
    (heps : 0 < eps)
    (hratio_nat : (alpha * B) / eps ≤ (n + 1 : ℕ)) :
    X ≤ Xbound eps := by
  have hratio : (alpha * B) / eps ≤ (n + 1 : ℝ) := by
    exact_mod_cast hratio_nat
  exact primalBound_from_explicitRate_ratioBound
    (hmono := hmono) (hprimal_n := hprimal_n) (hrate_n := hrate_n) (heps := heps) hratio

/--
Successor-index explicit-rate budget transfer.

This is the `(n+1)` iterate convenience wrapper for
`primalBound_from_explicitRate_budget`.
-/
theorem primalBound_from_explicitRate_budget_succ
    {X alpha B budget : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ (alpha * B) / ((n + 1 : ℝ) + 1))
    (hbudget : (alpha * B) / ((n + 1 : ℝ) + 1) ≤ budget) :
    X ≤ Xbound budget := by
  exact primalBound_fromDualBound hmono hprimal_n1 (hrate_n1.trans hbudget)

/--
Successor-index explicit-rate ratio-form transfer.
-/
theorem primalBound_from_explicitRate_ratioBound_succ
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ (alpha * B) / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hratio : (alpha * B) / eps ≤ (n + 1 : ℝ) + 1) :
    X ≤ Xbound eps := by
  have hthreshold' : alpha * B ≤ eps * ((n + 1 : ℝ) + 1) := by
    nlinarith [(div_le_iff₀ heps).1 hratio]
  have hpos : 0 < ((n + 1 : ℝ) + 1) := by positivity
  have hthreshold : (alpha * B) / ((n + 1 : ℝ) + 1) ≤ eps := by
    have hthreshold'' : alpha * B ≤ eps * ((n + 1 : ℝ) + 1) := hthreshold'
    exact (div_le_iff₀ hpos).2 hthreshold''
  exact primalBound_fromDualBound hmono hprimal_n1 (hrate_n1.trans hthreshold)

/--
Successor-index explicit-rate ratio-form transfer from a natural-number bound.
-/
theorem primalBound_from_explicitRate_ratioBound_succ_of_natBound
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ (alpha * B) / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hratio_nat : (alpha * B) / eps ≤ ((n + 1) + 1 : ℕ)) :
    X ≤ Xbound eps := by
  have hratio' : (alpha * B) / eps ≤ (((n + 1) + 1 : ℕ) : ℝ) := by
    exact_mod_cast hratio_nat
  have hratio : (alpha * B) / eps ≤ (n + 1 : ℝ) + 1 := by
    simpa [Nat.cast_add, Nat.cast_one] using hratio'
  exact primalBound_from_explicitRate_ratioBound_succ
    (hmono := hmono) (hprimal_n1 := hprimal_n1) (hrate_n1 := hrate_n1) (heps := heps) hratio

/--
Explicit dual rate envelope plus ceiling-form stopping rule gives primal `eps`-confinement.
-/
theorem primalBound_from_explicitRate_closedFormCeil
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n : ℕ}
    (hprimal_n : X ≤ Xbound (gap n))
    (hrate_n : gap n ≤ (alpha * B) / (n + 1 : ℝ))
    (heps : 0 < eps)
    (hn : Nat.ceil ((alpha * B) / eps) ≤ n + 1) :
    X ≤ Xbound eps :=
  primalBound_from_dualRate_closedFormCeil hmono hprimal_n hrate_n heps hn

/--
Explicit-rate closed-form-ceil transfer via an intermediate natural bound.
-/
theorem primalBound_from_explicitRate_closedFormCeil_of_natBound
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n N : ℕ}
    (hprimal_n : X ≤ Xbound (gap n))
    (hrate_n : gap n ≤ (alpha * B) / (n + 1 : ℝ))
    (heps : 0 < eps)
    (hn_nat : Nat.ceil ((alpha * B) / eps) ≤ N)
    (hN : N ≤ n + 1) :
    X ≤ Xbound eps :=
  primalBound_from_explicitRate_closedFormCeil
    hmono hprimal_n hrate_n heps (hn_nat.trans hN)

/--
Explicit-rate closed-form-ceil transfer with an index comparison helper.
-/
theorem primalBound_from_explicitRate_closedFormCeil_of_le_index
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n : ℕ}
    (hprimal_n : X ≤ Xbound (gap n))
    (hrate_n : gap n ≤ (alpha * B) / (n + 1 : ℝ))
    (heps : 0 < eps)
    (hn : Nat.ceil ((alpha * B) / eps) ≤ m + 1)
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hn' : Nat.ceil ((alpha * B) / eps) ≤ n + 1 := by
    exact hn.trans (Nat.add_le_add_right hmn 1)
  exact primalBound_from_explicitRate_closedFormCeil
    hmono hprimal_n hrate_n heps hn'

/--
Successor-index explicit-rate ceiling-form transfer.

This is the `(n+1)` iterate convenience form of
`primalBound_from_explicitRate_closedFormCeil`.
-/
theorem primalBound_from_explicitRate_closedFormCeil_succ
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ (alpha * B) / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hn : Nat.ceil ((alpha * B) / eps) ≤ (n + 1) + 1) :
    X ≤ Xbound eps := by
  have hratio : (alpha * B) / eps ≤ (n + 1 : ℝ) + 1 := by
    have hratio_nat : (alpha * B) / eps ≤ ((n + 1) + 1 : ℕ) :=
      Nat.le_of_ceil_le hn
    exact_mod_cast hratio_nat
  exact primalBound_from_explicitRate_ratioBound_succ
    (hmono := hmono) (hprimal_n1 := hprimal_n1) (hrate_n1 := hrate_n1)
    (heps := heps) hratio

/--
Successor-index dual-rate closed-form-ceil transfer with an index comparison helper.

If the ceiling bound is known at successor index `m+1` and `m ≤ n`, this upgrades to `n+1`.
-/
theorem primalBound_from_dualRate_closedFormCeil_succ_of_le_index
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ C / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hk : Nat.ceil (C / eps) ≤ (m + 1) + 1)
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hmn_nat : (m + 1) + 1 ≤ (n + 1) + 1 := by
    exact Nat.add_le_add_right (Nat.add_le_add_right hmn 1) 1
  have hn : Nat.ceil (C / eps) ≤ (n + 1) + 1 := hk.trans hmn_nat
  exact primalBound_from_dualRate_closedFormCeil_succ
    hmono hprimal_n1 hrate_n1 heps hn

/--
Successor-index explicit-rate ratio transfer with an index comparison helper.

If the ratio bound holds at index `m+1` and `m ≤ n`, this upgrades directly to `n+1`.
-/
theorem primalBound_from_explicitRate_ratioBound_succ_of_le_index
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ (alpha * B) / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hratio_m : (alpha * B) / eps ≤ (m + 1 : ℝ) + 1)
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hmn_nat : (m + 1) + 1 ≤ (n + 1) + 1 := by
    exact Nat.add_le_add_right (Nat.add_le_add_right hmn 1) 1
  have hmn_real : (m + 1 : ℝ) + 1 ≤ (n + 1 : ℝ) + 1 := by
    exact_mod_cast hmn_nat
  have hratio_n : (alpha * B) / eps ≤ (n + 1 : ℝ) + 1 := hratio_m.trans hmn_real
  exact primalBound_from_explicitRate_ratioBound_succ
    hmono hprimal_n1 hrate_n1 heps hratio_n

/--
Successor-index explicit-rate closed-form-ceil transfer with an index comparison helper.
-/
theorem primalBound_from_explicitRate_closedFormCeil_succ_of_le_index
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ (alpha * B) / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hn : Nat.ceil ((alpha * B) / eps) ≤ (m + 1) + 1)
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hmn_nat : (m + 1) + 1 ≤ (n + 1) + 1 := by
    exact Nat.add_le_add_right (Nat.add_le_add_right hmn 1) 1
  have hn' : Nat.ceil ((alpha * B) / eps) ≤ (n + 1) + 1 := hn.trans hmn_nat
  exact primalBound_from_explicitRate_closedFormCeil_succ
    hmono hprimal_n1 hrate_n1 heps hn'

/--
Successor-index explicit-rate closed-form-ceil transfer via an intermediate natural bound.
-/
theorem primalBound_from_explicitRate_closedFormCeil_succ_of_natBound
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {n N : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ (alpha * B) / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hn_nat : Nat.ceil ((alpha * B) / eps) ≤ N)
    (hN : N ≤ (n + 1) + 1) :
    X ≤ Xbound eps :=
  primalBound_from_explicitRate_closedFormCeil_succ
    hmono hprimal_n1 hrate_n1 heps (hn_nat.trans hN)

/--
Successor-index dual-rate ratio-form transfer with an index comparison helper.
-/
theorem primalBound_from_dualRate_ratioBound_succ_of_le_index
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ C / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hratio_m : C / eps ≤ (m + 1 : ℝ) + 1)
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hmn_nat : (m + 1) + 1 ≤ (n + 1) + 1 := by
    exact Nat.add_le_add_right (Nat.add_le_add_right hmn 1) 1
  have hmn_real : (m + 1 : ℝ) + 1 ≤ (n + 1 : ℝ) + 1 := by
    exact_mod_cast hmn_nat
  have hratio_n : C / eps ≤ (n + 1 : ℝ) + 1 := hratio_m.trans hmn_real
  exact primalBound_from_dualRate_ratioBound_succ
    hmono hprimal_n1 hrate_n1 heps hratio_n

/--
Successor-index dual-rate ratio-form transfer from a natural-number bound
with an index comparison helper.
-/
theorem primalBound_from_dualRate_ratioBound_succ_of_natBound_of_le_index
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ C / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hratio_nat_m : C / eps ≤ ((m + 1) + 1 : ℕ))
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hratio_m' : C / eps ≤ (((m + 1) + 1 : ℕ) : ℝ) := by
    exact_mod_cast hratio_nat_m
  have hratio_m : C / eps ≤ (m + 1 : ℝ) + 1 := by
    simpa [Nat.cast_add, Nat.cast_one] using hratio_m'
  exact primalBound_from_dualRate_ratioBound_succ_of_le_index
    hmono hprimal_n1 hrate_n1 heps hratio_m hmn

/--
Successor-index dual-rate closed-form-ceil transfer from a natural-number bound
with an index comparison helper.
-/
theorem primalBound_from_dualRate_closedFormCeil_succ_of_natBound_of_le_index
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n N : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ C / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hk_nat : Nat.ceil (C / eps) ≤ N)
    (hN : N ≤ (m + 1) + 1)
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hk : Nat.ceil (C / eps) ≤ (m + 1) + 1 := hk_nat.trans hN
  exact primalBound_from_dualRate_closedFormCeil_succ_of_le_index
    hmono hprimal_n1 hrate_n1 heps hk hmn

/--
Successor-index explicit-rate closed-form-ceil transfer from a natural-number bound
with an index comparison helper.
-/
theorem primalBound_from_explicitRate_closedFormCeil_succ_of_natBound_of_le_index
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n N : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ (alpha * B) / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hn_nat : Nat.ceil ((alpha * B) / eps) ≤ N)
    (hN : N ≤ (m + 1) + 1)
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hn : Nat.ceil ((alpha * B) / eps) ≤ (m + 1) + 1 := hn_nat.trans hN
  exact primalBound_from_explicitRate_closedFormCeil_succ_of_le_index
    hmono hprimal_n1 hrate_n1 heps hn hmn

/--
Successor-index explicit-rate ratio-form transfer from a natural-number bound
with an index comparison helper.
-/
theorem primalBound_from_explicitRate_ratioBound_succ_of_natBound_of_le_index
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n : ℕ}
    (hprimal_n1 : X ≤ Xbound (gap (n + 1)))
    (hrate_n1 : gap (n + 1) ≤ (alpha * B) / ((n + 1 : ℝ) + 1))
    (heps : 0 < eps)
    (hratio_nat_m : (alpha * B) / eps ≤ ((m + 1) + 1 : ℕ))
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hratio_m' : (alpha * B) / eps ≤ (((m + 1) + 1 : ℕ) : ℝ) := by
    exact_mod_cast hratio_nat_m
  have hratio_m : (alpha * B) / eps ≤ (m + 1 : ℝ) + 1 := by
    simpa [Nat.cast_add, Nat.cast_one] using hratio_m'
  exact primalBound_from_explicitRate_ratioBound_succ_of_le_index
    hmono hprimal_n1 hrate_n1 heps hratio_m hmn

/--
Dual-rate closed-form-ceil transfer from a natural bound and an index comparison helper.
-/
theorem primalBound_from_dualRate_closedFormCeil_of_natBound_of_le_index
    {X C eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n N : ℕ}
    (hprimal_n : X ≤ Xbound (gap n))
    (hrate_n : gap n ≤ C / (n + 1 : ℝ))
    (heps : 0 < eps)
    (hk_nat : Nat.ceil (C / eps) ≤ N)
    (hN : N ≤ m + 1)
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hk : Nat.ceil (C / eps) ≤ m + 1 := hk_nat.trans hN
  exact primalBound_from_dualRate_closedFormCeil_of_le_index
    hmono hprimal_n hrate_n heps hk hmn

/--
Explicit-rate closed-form-ceil transfer from a natural bound and an index comparison helper.
-/
theorem primalBound_from_explicitRate_closedFormCeil_of_natBound_of_le_index
    {X alpha B eps : ℝ} {Xbound : ℝ → ℝ}
    (hmono : Monotone Xbound)
    {gap : ℕ → ℝ} {m n N : ℕ}
    (hprimal_n : X ≤ Xbound (gap n))
    (hrate_n : gap n ≤ (alpha * B) / (n + 1 : ℝ))
    (heps : 0 < eps)
    (hn_nat : Nat.ceil ((alpha * B) / eps) ≤ N)
    (hN : N ≤ m + 1)
    (hmn : m ≤ n) :
    X ≤ Xbound eps := by
  have hn : Nat.ceil ((alpha * B) / eps) ≤ m + 1 := hn_nat.trans hN
  exact primalBound_from_explicitRate_closedFormCeil_of_le_index
    hmono hprimal_n hrate_n heps hn hmn

end PrimalDualBounds
end KLProjection
end FlowSinkhorn
