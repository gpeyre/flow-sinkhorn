import FlowSinkhorn.KLProjection.PrimalDualBounds.Blueprint
import FlowSinkhorn.KLProjection.Variation
import Mathlib.Analysis.SpecialFunctions.Arsinh

/-!
# Closed-form graph-`W₁` block updates

This module is reserved for the closed-form block analysis from
`papers/kl-projections/sections/sec-w1-graphs.tex`.

Paper targets:
- Proposition `prop:explici-formula-proj`;
- Proposition `prop:flow-algo`;
- Proposition `prop:V1V2_closed_form_flow`;
- Proposition `prop:Psi2_closed_nonexp`.

Intended theorem names:
- `graphW1_projection_closedForm`;
- `graphW1_flowSinkhorn_update`;
- `graphW1_blockQuotient_closedForm`;
- `graphW1_Psi2_nonexpansive`.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace Applications
namespace GraphW1

open Function

/--
Helper scalar root used in Proposition `prop:explici-formula-proj`.
-/
noncomputable def graphW1_phi (t u : ℝ) : ℝ :=
  (Real.sqrt (t ^ 2 + 4 * u) - t) / 2

/--
`phi(t,u) = (sqrt(t^2+4u)-t)/2` solves `x^2 + t*x - u = 0` whenever the discriminant is
nonnegative.
-/
theorem graphW1_phi_is_quadratic_root
    (t u : ℝ)
    (hdisc : 0 ≤ t ^ 2 + 4 * u) :
    (graphW1_phi t u) ^ 2 + t * graphW1_phi t u - u = 0 := by
  unfold graphW1_phi
  have hsq : (Real.sqrt (t ^ 2 + 4 * u)) ^ 2 = t ^ 2 + 4 * u := Real.sq_sqrt hdisc
  nlinarith [hsq]

/--
Coordinate form of the `C1` projection equation from Proposition `prop:explici-formula-proj`.

For a fixed coordinate, with
`t = bDiff / hRow` and `u = hCol / hRow`, the proposed scaling
`s = phi(t,u)` solves
`(s^2) * hRow + s * bDiff - hCol = 0`.
-/
theorem graphW1_projection_C1_coordinate
    (bDiff hRow hCol : ℝ)
    (hrow_ne : hRow ≠ 0)
    (hdisc : 0 ≤ (bDiff / hRow) ^ 2 + 4 * (hCol / hRow)) :
    let s := graphW1_phi (bDiff / hRow) (hCol / hRow)
    (s ^ 2) * hRow + s * bDiff - hCol = 0 := by
  intro s
  have hroot :
      s ^ 2 + (bDiff / hRow) * s - (hCol / hRow) = 0 := by
    simpa [s] using
      graphW1_phi_is_quadratic_root (bDiff / hRow) (hCol / hRow) hdisc
  have hroot_mul : hRow * (s ^ 2 + (bDiff / hRow) * s - (hCol / hRow)) = 0 := by
    rw [hroot, mul_zero]
  have hrewrite :
      hRow * (s ^ 2 + (bDiff / hRow) * s - (hCol / hRow)) =
        (s ^ 2) * hRow + s * bDiff - hCol := by
    field_simp [hrow_ne]
  simpa [hrewrite] using hroot_mul

/--
Coordinate form of the `C2` projection equation from Proposition `prop:explici-formula-proj`.

The geometric-mean candidate `sqrt(f*g)` satisfies the product constraint and is nonnegative.
-/
theorem graphW1_projection_C2_coordinate
    (f g : ℝ)
    (hfg : 0 ≤ f * g) :
    let p := Real.sqrt (f * g)
    p ^ 2 = f * g ∧ 0 ≤ p := by
  intro p
  constructor
  · simpa [p] using (Real.sq_sqrt hfg)
  · simpa [p] using (Real.sqrt_nonneg (f * g))

/--
Closed-form characterization of the graph-`W₁` primal KL projections (Proposition
`prop:explici-formula-proj`) in coordinate form.

This theorem certifies the two explicit formulas used in the paper:
1. `C1` scaling satisfies the coordinate quadratic equation with `phi`;
2. `C2` projection is the geometric mean coordinatewise.
-/
theorem graphW1_projection_closedForm
    {ι : Type*}
    (bDiff hRow hCol : ι → ℝ)
    (f g : ι → ι → ℝ)
    (hrow_ne : ∀ i : ι, hRow i ≠ 0)
    (hdisc : ∀ i : ι, 0 ≤ (bDiff i / hRow i) ^ 2 + 4 * (hCol i / hRow i))
    (hfg : ∀ i j : ι, 0 ≤ f i j * g i j) :
    (∀ i : ι,
      let s := graphW1_phi (bDiff i / hRow i) (hCol i / hRow i)
      (s ^ 2) * hRow i + s * bDiff i - hCol i = 0) ∧
    (∀ i j : ι,
      let p := Real.sqrt (f i j * g i j)
      p ^ 2 = f i j * g i j ∧ 0 ≤ p) := by
  constructor
  · intro i
    exact graphW1_projection_C1_coordinate (bDiff i) (hRow i) (hCol i) (hrow_ne i) (hdisc i)
  · intro i j
    exact graphW1_projection_C2_coordinate (f i j) (g i j) (hfg i j)

/-!
## Diagnostic: paper-as-stated graph-`W₁` closed-form encoding

This block intentionally encodes algebraic consequences of the paper formulas as written
(`sec-w1-graphs.tex`, especially `eq:bregman-flow-form`), to test consistency before any
manuscript correction.
-/

/--
Correct normalization bridge for `eq:bregman-flow-form`:
with `s_i = exp(v_i / (2γ))`, the entrywise parametrization
`zC_{i,j} * exp((v_i - v_j)/(2γ))` is exactly `zC_{i,j} * (s_i / s_j)`.
-/
theorem graphW1_bregman_flow_form_halfGamma_scaling
    (zC vi vj gamma : ℝ) :
    zC * Real.exp ((vi - vj) / (2 * gamma)) =
      zC * (Real.exp (vi / (2 * gamma)) / Real.exp (vj / (2 * gamma))) := by
  have hsplit : (vi - vj) / (2 * gamma) = vi / (2 * gamma) - vj / (2 * gamma) := by ring
  rw [hsplit, Real.exp_sub]

/--
If one uses the old normalization `s_i = exp(v_i / γ)`, then matching
`exp((v_i - v_j)/(2γ))` with `s_i / s_j` forces equal potentials.
This is kept as a legacy diagnostic check.
-/
theorem graphW1_paper_bregman_forms_force_equal_potentials
    {vi vj gamma : ℝ}
    (hgamma : gamma ≠ 0)
    (hexp :
      Real.exp ((vi - vj) / (2 * gamma)) =
        Real.exp ((vi - vj) / gamma)) :
    vi = vj := by
  have harg :
      (vi - vj) / (2 * gamma) = (vi - vj) / gamma :=
    Real.exp_injective hexp
  have harg' := harg
  field_simp [hgamma] at harg'
  linarith

/--
Concrete witness that the two exponential factors from
the old normalization (`s = exp(v/γ)`) are not equal in general.
-/
theorem graphW1_paper_bregman_forms_not_equal_in_general :
    Real.exp (((2 : ℝ) - 0) / (2 * (1 : ℝ))) ≠
      Real.exp (((2 : ℝ) - 0) / (1 : ℝ)) := by
  intro h
  have harg : (1 : ℝ) = 2 := by
    have : Real.exp (1 : ℝ) = Real.exp (2 : ℝ) := by
      simpa using h
    exact Real.exp_injective this
  norm_num at harg

/--
Closed-form Flow-Sinkhorn sweep update for graph-`W₁`.

Coordinate-wise certified form of Proposition `prop:flow-algo`.

Given:
1. the closed form of the intermediate ratio `q`,
2. the geometric-mean update `sNext = s * sqrt(q)`,
3. positivity hypotheses ensuring the square roots are on nonnegative arguments,
this yields the stable one-step scaling update
`sNext = sqrt((s/b) * (sqrt(r^2 + a*b) - r))`.
-/
theorem graphW1_flowSinkhorn_update
    (r a b s q sNext : ℝ)
    (hb : 0 < b)
    (hs : 0 < s)
    (hq :
      q = (Real.sqrt (r ^ 2 + a * b) - r) / (s * b))
    (hq_nonneg : 0 ≤ q)
    (hsNext : sNext = s * Real.sqrt q) :
    sNext = Real.sqrt ((s / b) * (Real.sqrt (r ^ 2 + a * b) - r)) := by
  have hs_nonneg : 0 ≤ s := le_of_lt hs
  have hy_nonneg : 0 ≤ s * Real.sqrt q := mul_nonneg hs_nonneg (Real.sqrt_nonneg q)
  have hsb_ne : s * b ≠ 0 := by nlinarith [hs, hb]
  have hsq :
      (s * Real.sqrt q) ^ 2 = (s / b) * (Real.sqrt (r ^ 2 + a * b) - r) := by
    calc
      (s * Real.sqrt q) ^ 2 = s ^ 2 * (Real.sqrt q) ^ 2 := by ring
      _ = s ^ 2 * q := by rw [Real.sq_sqrt hq_nonneg]
      _ = s ^ 2 * ((Real.sqrt (r ^ 2 + a * b) - r) / (s * b)) := by rw [hq]
      _ = (s / b) * (Real.sqrt (r ^ 2 + a * b) - r) := by
        field_simp [hsb_ne]
  have hrad :
      0 ≤ (s / b) * (Real.sqrt (r ^ 2 + a * b) - r) := by
    have hsq_nonneg : 0 ≤ (s * Real.sqrt q) ^ 2 := sq_nonneg _
    simpa [hsq] using hsq_nonneg
  have hsqrt_eq :
      Real.sqrt ((s / b) * (Real.sqrt (r ^ 2 + a * b) - r)) = s * Real.sqrt q := by
    exact (Real.sqrt_eq_iff_eq_sq hrad hy_nonneg).2 hsq.symm
  rw [hsNext]
  exact hsqrt_eq.symm

/--
Vector form of Proposition `prop:flow-algo` (Flow-Sinkhorn update in scaling variables).

Under the coordinatewise closed forms for the `C1` and `C2` steps, this gives the exact
one-sweep update written in the paper:
`sNext_i = sqrt((s_i / b_i) * (sqrt(r_i^2 + a_i*b_i) - r_i))`.
-/
theorem graphW1_flowSinkhorn_update_vector
    {ι : Type*}
    (r a b s q sNext : ι → ℝ)
    (hb : ∀ i : ι, 0 < b i)
    (hs : ∀ i : ι, 0 < s i)
    (hq :
      ∀ i : ι,
        q i = (Real.sqrt (r i ^ 2 + a i * b i) - r i) / (s i * b i))
    (hq_nonneg : ∀ i : ι, 0 ≤ q i)
    (hsNext : ∀ i : ι, sNext i = s i * Real.sqrt (q i)) :
    sNext =
      fun i =>
        Real.sqrt ((s i / b i) * (Real.sqrt (r i ^ 2 + a i * b i) - r i)) := by
  funext i
  exact graphW1_flowSinkhorn_update
    (r := r i) (a := a i) (b := b i) (s := s i) (q := q i) (sNext := sNext i)
    (hb i) (hs i) (hq i) (hq_nonneg i) (hsNext i)

/--
Paper-shaped form of Proposition `prop:flow-algo` with explicit `zC` matrix-vector terms.

This is the direct componentwise rewrite
`sNext_i = sqrt((s_i / (zC*(1/s))_i) * (sqrt(r_i^2 + (zC*s)_i * (zC*(1/s))_i) - r_i))`.
-/
theorem graphW1_flowSinkhorn_update_as_stated
    {ι : Type*} [Fintype ι]
    (zC : ι → ι → ℝ)
    (r s q sNext : ι → ℝ)
    (hb :
      ∀ i : ι,
        0 < Finset.univ.sum (fun j : ι => zC i j * (1 / s j)))
    (hs : ∀ i : ι, 0 < s i)
    (hq :
      ∀ i : ι,
        q i =
          (Real.sqrt
            (r i ^ 2 +
              (Finset.univ.sum (fun j : ι => zC i j * s j)) *
              (Finset.univ.sum (fun j : ι => zC i j * (1 / s j))) ) - r i) /
            (s i * (Finset.univ.sum (fun j : ι => zC i j * (1 / s j))))
      )
    (hq_nonneg : ∀ i : ι, 0 ≤ q i)
    (hsNext : ∀ i : ι, sNext i = s i * Real.sqrt (q i)) :
    sNext =
      fun i =>
        Real.sqrt (
          (s i / (Finset.univ.sum (fun j : ι => zC i j * (1 / s j)))) *
          (Real.sqrt
            (r i ^ 2 +
              (Finset.univ.sum (fun j : ι => zC i j * s j)) *
              (Finset.univ.sum (fun j : ι => zC i j * (1 / s j))) ) - r i)
        ) := by
  funext i
  exact graphW1_flowSinkhorn_update
    (r := r i)
    (a := Finset.univ.sum (fun j : ι => zC i j * s j))
    (b := Finset.univ.sum (fun j : ι => zC i j * (1 / s j)))
    (s := s i)
    (q := q i)
    (sNext := sNext i)
    (hb i)
    (hs i)
    (hq i)
    (hq_nonneg i)
    (hsNext i)

/--
Stable dual update derived from the code-form step.

This theorem encodes the sign-sensitive algebraic bridge:
if the implementation update is
`hNext = h/2 - (gamma/2) * m`
and `m` is written as
`(alphaMinus - alphaPlus)/(2*gamma) + arsinh(beta)`,
then (with `v = 2h`) the induced dual update is
`vNext = (1/2)v + (1/2)(alphaPlus - alphaMinus) - gamma*arsinh(beta)`.
-/
theorem graphW1_vUpdate_stable_correct_from_code
    (v h hNext m alphaPlus alphaMinus beta gamma : ℝ)
    (hgamma : gamma ≠ 0)
    (hv : v = 2 * h)
    (hhNext : hNext = h / 2 - gamma / 2 * m)
    (hm : m = (alphaMinus - alphaPlus) / (2 * gamma) + Real.arsinh beta) :
    2 * hNext =
      (1 / 2) * v + (1 / 2) * (alphaPlus - alphaMinus) - gamma * Real.arsinh beta := by
  rw [hhNext, hm, hv]
  field_simp [hgamma]
  ring

/--
Vector version of `graphW1_vUpdate_stable_correct_from_code`.
-/
theorem graphW1_vUpdate_stable_correct_from_code_vector
    {ι : Type*}
    (v h hNext m alphaPlus alphaMinus beta : ι → ℝ)
    (gamma : ℝ)
    (hgamma : gamma ≠ 0)
    (hv : ∀ i : ι, v i = 2 * h i)
    (hhNext : ∀ i : ι, hNext i = h i / 2 - gamma / 2 * m i)
    (hm :
      ∀ i : ι, m i = (alphaMinus i - alphaPlus i) / (2 * gamma) + Real.arsinh (beta i)) :
    (fun i : ι => 2 * hNext i) =
      (fun i : ι =>
        (1 / 2) * v i + (1 / 2) * (alphaPlus i - alphaMinus i) - gamma * Real.arsinh (beta i)) := by
  funext i
  exact graphW1_vUpdate_stable_correct_from_code
    (v := v i) (h := h i) (hNext := hNext i) (m := m i)
    (alphaPlus := alphaPlus i) (alphaMinus := alphaMinus i) (beta := beta i) (gamma := gamma)
    hgamma (hv i) (hhNext i) (hm i)

/--
Closed-form expression for the graph-`W₁` block quotient constant.

Concrete paper-facing statement of Proposition `prop:V1V2_closed_form_flow`.

It certifies that the quotient seminorm on both blocks (`V1` for vertex potentials, `V2` for
edge-flow tensors) is exactly the variation seminorm.
-/
theorem graphW1_blockQuotient_closedForm
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (v : ι → ℝ)
    (U : ι × ι → ℝ) :
    Setup.blockQuotientSeminorm v = variationSeminorm v ∧
      Setup.blockQuotientSeminorm U = variationSeminorm U := by
  exact ⟨Setup.blockQuotientSeminorm_eq_variationSeminorm v,
    Setup.blockQuotientSeminorm_eq_variationSeminorm U⟩

/--
Nonexpansiveness of the `Ψ₂` block map in the chosen seminorm.

Concrete paper-facing statement of Proposition `prop:Psi2_closed_nonexp`:
the closed-form graph-W₁ map `Ψ₂(v)_(i,j) = (v_j - v_i)/2` is non-expansive in variation seminorm.
-/
theorem graphW1_Psi2_nonexpansive
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (v : ι → ℝ) :
    variationSeminorm (fun p : ι × ι => (v p.2 - v p.1) / 2) ≤ variationSeminorm v := by
  apply variationSeminorm_le_of_shifted_supnorm_bound _ (c := 0)
  intro ⟨i, j⟩
  simp only [add_zero]
  have hkey : |v j - v i| ≤ oscillation v := by
    unfold oscillation
    rw [abs_le]
    constructor <;> linarith [le_coordMax v j, coordMin_le v i, coordMin_le v j, le_coordMax v i]
  calc
    |(v j - v i) / 2|
        = |v j - v i| / 2 := by rw [abs_div, abs_of_pos (by norm_num : (0 : ℝ) < 2)]
    _ ≤ oscillation v / 2 := by nlinarith
    _ = variationSeminorm v := rfl

/--
If `Ψ₁` and `Ψ₂` are seminorm-nonexpansive and the closed-form update agrees pointwise with
`Ψ₂ ∘ Ψ₁`, then the closed-form update is seminorm-nonexpansive.
-/
theorem graphW1_flowSinkhorn_closedForm_nonexpansive
    {𝕜 E : Type*}
    [NormedField 𝕜] [AddCommGroup E] [Module 𝕜 E]
    (p : Seminorm 𝕜 E)
    (Psi₁ Psi₂ closedFormUpdate : E → E)
    (hPsi₁ : SeminormNonexpansive p Psi₁)
    (hPsi₂ : SeminormNonexpansive p Psi₂)
    (hupdate : ∀ u : E, (Psi₂ ∘ Psi₁) u = closedFormUpdate u) :
    SeminormNonexpansive p closedFormUpdate := by
  intro x y
  calc
    p (closedFormUpdate x - closedFormUpdate y)
        = p ((Psi₂ ∘ Psi₁) x - (Psi₂ ∘ Psi₁) y) := by
          rw [hupdate x, hupdate y]
    _ ≤ p (Psi₁ x - Psi₁ y) := hPsi₂ (Psi₁ x) (Psi₁ y)
    _ ≤ p (x - y) := hPsi₁ x y

/--
Pointwise closed-form equality of one sweep propagates to every iterate of the sweep map.
-/
theorem graphW1_flowSinkhorn_update_iterate
    {E : Type*}
    (Psi₁ Psi₂ closedFormUpdate : E → E)
    (hupdate : ∀ u : E, (Psi₂ ∘ Psi₁) u = closedFormUpdate u) :
    ∀ k : ℕ, ∀ u : E, ((Psi₂ ∘ Psi₁)^[k]) u = (closedFormUpdate^[k]) u := by
  intro k
  induction k with
  | zero =>
      intro u
      simp
  | succ k ih =>
      intro u
      calc
        ((Psi₂ ∘ Psi₁)^[k + 1]) u = (Psi₂ ∘ Psi₁) (((Psi₂ ∘ Psi₁)^[k]) u) := by
          simp [Function.iterate_succ_apply']
        _ = (Psi₂ ∘ Psi₁) ((closedFormUpdate^[k]) u) := by
          rw [ih u]
        _ = closedFormUpdate ((closedFormUpdate^[k]) u) := by
          simpa using hupdate ((closedFormUpdate^[k]) u)
        _ = (closedFormUpdate^[k + 1]) u := by
          simp [Function.iterate_succ_apply']

/--
Every iterate of the closed-form graph-`W₁` update is seminorm-nonexpansive.

This composes:
1. blockwise nonexpansiveness of `Ψ₁` and `Ψ₂`,
2. pointwise closed-form identity for one sweep,
3. closure of `SeminormNonexpansive` under iteration.
-/
theorem graphW1_flowSinkhorn_closedForm_iterate_nonexpansive
    {𝕜 E : Type*}
    [NormedField 𝕜] [AddCommGroup E] [Module 𝕜 E]
    (p : Seminorm 𝕜 E)
    (Psi₁ Psi₂ closedFormUpdate : E → E)
    (hPsi₁ : SeminormNonexpansive p Psi₁)
    (hPsi₂ : SeminormNonexpansive p Psi₂)
    (hupdate : ∀ u : E, (Psi₂ ∘ Psi₁) u = closedFormUpdate u)
    (k : ℕ) :
    SeminormNonexpansive p (closedFormUpdate^[k]) := by
  have hclosed : SeminormNonexpansive p closedFormUpdate :=
    graphW1_flowSinkhorn_closedForm_nonexpansive
      p Psi₁ Psi₂ closedFormUpdate hPsi₁ hPsi₂ hupdate
  exact SeminormNonexpansive_iterate p closedFormUpdate hclosed k

/--
Uniform orbit bound for closed-form graph-`W₁` iterates near a fixed point.

This is the closed-form iterate analogue of the generic nonexpansive fixed-point estimate.
-/
theorem graphW1_flowSinkhorn_closedForm_iterate_bound
    {𝕜 E : Type*}
    [NormedField 𝕜] [AddCommGroup E] [Module 𝕜 E]
    (p : Seminorm 𝕜 E)
    (Psi₁ Psi₂ closedFormUpdate : E → E)
    (hPsi₁ : SeminormNonexpansive p Psi₁)
    (hPsi₂ : SeminormNonexpansive p Psi₂)
    (hupdate : ∀ u : E, (Psi₂ ∘ Psi₁) u = closedFormUpdate u)
    {uStar u0 : E}
    (hfix : closedFormUpdate uStar = uStar)
    (k : ℕ) :
    p ((closedFormUpdate^[k]) u0) ≤ p u0 + 2 * p uStar := by
  have hclosed : SeminormNonexpansive p closedFormUpdate :=
    graphW1_flowSinkhorn_closedForm_nonexpansive
      p Psi₁ Psi₂ closedFormUpdate hPsi₁ hPsi₂ hupdate
  exact seminorm_iterate_le_of_nonexpansive_fixedPoint
    p closedFormUpdate hclosed hfix k

/--
Uniform orbit bound for sweep iterates transported through the closed-form update identity.

This theorem is designed for downstream complexity arguments: one proves a fixed-point
bound in the closed-form model and directly obtains the same iterate bound for the sweep
`Ψ₂ ∘ Ψ₁`.
-/
theorem graphW1_flowSinkhorn_sweep_iterate_bound_via_closedForm
    {𝕜 E : Type*}
    [NormedField 𝕜] [AddCommGroup E] [Module 𝕜 E]
    (p : Seminorm 𝕜 E)
    (Psi₁ Psi₂ closedFormUpdate : E → E)
    (hPsi₁ : SeminormNonexpansive p Psi₁)
    (hPsi₂ : SeminormNonexpansive p Psi₂)
    (hupdate : ∀ u : E, (Psi₂ ∘ Psi₁) u = closedFormUpdate u)
    {uStar u0 : E}
    (hfix : closedFormUpdate uStar = uStar)
    (k : ℕ) :
    p (((Psi₂ ∘ Psi₁)^[k]) u0) ≤ p u0 + 2 * p uStar := by
  have hbound :
      p ((closedFormUpdate^[k]) u0) ≤ p u0 + 2 * p uStar :=
    graphW1_flowSinkhorn_closedForm_iterate_bound
      p Psi₁ Psi₂ closedFormUpdate hPsi₁ hPsi₂ hupdate hfix k
  have hiterEq :
      ((Psi₂ ∘ Psi₁)^[k]) u0 = (closedFormUpdate^[k]) u0 :=
    graphW1_flowSinkhorn_update_iterate Psi₁ Psi₂ closedFormUpdate hupdate k u0
  simpa [hiterEq] using hbound

/--
Closed-form sweep update is variation-seminorm nonexpansive from topical block witnesses.

This is the `IsTopical`-facing counterpart of
`graphW1_flowSinkhorn_closedForm_nonexpansive` specialized to
`variationSeminormAsSeminorm`.
-/
theorem graphW1_flowSinkhorn_closedForm_variationSeminorm_nonexpansive_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi₁ Psi₂ closedFormUpdate : (ι → ℝ) → (ι → ℝ))
    (hPsi₁ : IsTopical Psi₁)
    (hPsi₂ : IsTopical Psi₂)
    (hupdate : ∀ u : ι → ℝ, (Psi₂ ∘ Psi₁) u = closedFormUpdate u) :
    SeminormNonexpansive variationSeminormAsSeminorm closedFormUpdate := by
  exact
    graphW1_flowSinkhorn_closedForm_nonexpansive
      variationSeminormAsSeminorm Psi₁ Psi₂ closedFormUpdate
      (SeminormNonexpansive_variationSeminormAsSeminorm_of_isTopical hPsi₁)
      (SeminormNonexpansive_variationSeminormAsSeminorm_of_isTopical hPsi₂)
      hupdate

/--
Uniform iterate bound for the closed-form graph-W₁ update from topical block witnesses.

This packages:
1. `IsTopical` certificates for both block maps,
2. pointwise closed-form identity for one sweep,
3. fixed-point control at the base constant `B`.
-/
theorem graphW1_flowSinkhorn_closedForm_variationSeminorm_iterate_bound_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi₁ Psi₂ closedFormUpdate : (ι → ℝ) → (ι → ℝ))
    (hPsi₁ : IsTopical Psi₁)
    (hPsi₂ : IsTopical Psi₂)
    (hupdate : ∀ u : ι → ℝ, (Psi₂ ∘ Psi₁) u = closedFormUpdate u)
    {uStar u0 : ι → ℝ}
    (hfix : closedFormUpdate uStar = uStar)
    {B : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (k : ℕ) :
    variationSeminorm ((closedFormUpdate^[k]) u0) ≤ variationSeminorm u0 + 2 * B := by
  have hclosed :
      SeminormNonexpansive variationSeminormAsSeminorm closedFormUpdate :=
    graphW1_flowSinkhorn_closedForm_variationSeminorm_nonexpansive_of_isTopical
      Psi₁ Psi₂ closedFormUpdate hPsi₁ hPsi₂ hupdate
  simpa using
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
      variationSeminormAsSeminorm closedFormUpdate hclosed
      (uStar := uStar) (u0 := u0) hfix hbound k

/--
Uniform iterate bound for sweep iterates transported from the closed-form model.

This is the sweep-facing variant of
`graphW1_flowSinkhorn_closedForm_variationSeminorm_iterate_bound_of_isTopical`.
-/
theorem graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi₁ Psi₂ closedFormUpdate : (ι → ℝ) → (ι → ℝ))
    (hPsi₁ : IsTopical Psi₁)
    (hPsi₂ : IsTopical Psi₂)
    (hupdate : ∀ u : ι → ℝ, (Psi₂ ∘ Psi₁) u = closedFormUpdate u)
    {uStar u0 : ι → ℝ}
    (hfix : closedFormUpdate uStar = uStar)
    {B : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (k : ℕ) :
    variationSeminorm (((Psi₂ ∘ Psi₁)^[k]) u0) ≤ variationSeminorm u0 + 2 * B := by
  have hclosed :
      variationSeminorm ((closedFormUpdate^[k]) u0) ≤ variationSeminorm u0 + 2 * B :=
    graphW1_flowSinkhorn_closedForm_variationSeminorm_iterate_bound_of_isTopical
      Psi₁ Psi₂ closedFormUpdate hPsi₁ hPsi₂ hupdate hfix hbound k
  have hEq :
      ((Psi₂ ∘ Psi₁)^[k]) u0 = (closedFormUpdate^[k]) u0 :=
    graphW1_flowSinkhorn_update_iterate Psi₁ Psi₂ closedFormUpdate hupdate k u0
  simpa [hEq] using hclosed

/--
Zero-seed sweep iterate bound from topical block witnesses and closed-form identity.

If the initial point has zero variation seminorm, the base term disappears and one gets
`variationSeminorm (((Psi₂ ∘ Psi₁)^[k]) u0) ≤ 2 * B`.
-/
theorem graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_from_zero_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi₁ Psi₂ closedFormUpdate : (ι → ℝ) → (ι → ℝ))
    (hPsi₁ : IsTopical Psi₁)
    (hPsi₂ : IsTopical Psi₂)
    (hupdate : ∀ u : ι → ℝ, (Psi₂ ∘ Psi₁) u = closedFormUpdate u)
    {uStar u0 : ι → ℝ}
    (hfix : closedFormUpdate uStar = uStar)
    {B : ℝ}
    (hbound : variationSeminorm uStar ≤ B)
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm (((Psi₂ ∘ Psi₁)^[k]) u0) ≤ 2 * B := by
  have hbase :
      variationSeminorm (((Psi₂ ∘ Psi₁)^[k]) u0) ≤ variationSeminorm u0 + 2 * B :=
    graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_of_isTopical
      Psi₁ Psi₂ closedFormUpdate hPsi₁ hPsi₂ hupdate hfix hbound k
  rw [hzero, zero_add] at hbase
  exact hbase

/--
Closed-form rewrite of the graph-`W₁` budget at the two-diameter scale.

This exposes the constant used by two-step-path complexity bridges.
-/
theorem graphW1_hGammaBudget_twoDiam_closedForm
    {diam cost gamma hGamma : ℝ} :
    PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma hGamma =
      2 * diam * (cost + gamma * hGamma) := by
  simp [PrimalDualBounds.hGammaKappaBudget]

/--
Closed-form rewrite of the two-times budget constant at the two-diameter scale.
-/
theorem graphW1_twoTimes_hGammaBudget_twoDiam_closedForm
    {diam cost gamma hGamma : ℝ} :
    2 * PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma hGamma =
      4 * diam * (cost + gamma * hGamma) := by
  rw [graphW1_hGammaBudget_twoDiam_closedForm]
  ring

/--
Closed-form rewrite at the graph-`W₁` explicit scale `hGamma = log(n) / gamma`.
-/
theorem graphW1_hGammaBudget_twoDiam_logDivGamma_closedForm
    {diam cost gamma : ℝ} {n : ℕ}
    (hgamma : 0 < gamma) :
    PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma) =
      2 * diam * (cost + Real.log n) := by
  rw [graphW1_hGammaBudget_twoDiam_closedForm]
  have hg : gamma ≠ 0 := ne_of_gt hgamma
  have hmul : gamma * (Real.log n / gamma) = Real.log n := by
    field_simp [hg]
  simpa [hmul]

/--
Closed-form rewrite of the twice-budget constant at `hGamma = log(n) / gamma`.
-/
theorem graphW1_twoTimes_hGammaBudget_twoDiam_logDivGamma_closedForm
    {diam cost gamma : ℝ} {n : ℕ}
    (hgamma : 0 < gamma) :
    2 * PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma) =
      4 * diam * (cost + Real.log n) := by
  rw [graphW1_hGammaBudget_twoDiam_logDivGamma_closedForm (diam := diam) (cost := cost) hgamma]
  ring

/--
Two-step-path style sweep iterate bound with explicit diameter constant.

Starting from a fixed-point budget at `kappa`, if `kappa ≤ 2 * diam` and
`cost + gamma * hGamma ≥ 0`, the sweep iterate bound is upgraded to
`variationSeminorm u0 + 4 * diam * (cost + gamma * hGamma)`.
-/
theorem graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_twoDiam_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi₁ Psi₂ closedFormUpdate : (ι → ℝ) → (ι → ℝ))
    (hPsi₁ : IsTopical Psi₁)
    (hPsi₂ : IsTopical Psi₂)
    (hupdate : ∀ u : ι → ℝ, (Psi₂ ∘ Psi₁) u = closedFormUpdate u)
    {uStar u0 : ι → ℝ}
    (hfix : closedFormUpdate uStar = uStar)
    {kappa diam cost gamma hGamma : ℝ}
    (hbound : variationSeminorm uStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma)
    (hkappa : kappa ≤ 2 * diam)
    (hbase : 0 ≤ cost + gamma * hGamma)
    (k : ℕ) :
    variationSeminorm (((Psi₂ ∘ Psi₁)^[k]) u0) ≤
      variationSeminorm u0 + 4 * diam * (cost + gamma * hGamma) := by
  have hiter :
      variationSeminorm (((Psi₂ ∘ Psi₁)^[k]) u0) ≤
        variationSeminorm u0 +
          2 * PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma :=
    graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_of_isTopical
      Psi₁ Psi₂ closedFormUpdate hPsi₁ hPsi₂ hupdate hfix hbound k
  have hbudget :
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤
        2 * diam * (cost + gamma * hGamma) := by
    dsimp [PrimalDualBounds.hGammaKappaBudget]
    exact mul_le_mul_of_nonneg_right hkappa hbase
  have htwice :
      2 * PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤
        2 * (2 * diam * (cost + gamma * hGamma)) :=
    mul_le_mul_of_nonneg_left hbudget (by norm_num)
  have hiter' :
      variationSeminorm (((Psi₂ ∘ Psi₁)^[k]) u0) ≤
        variationSeminorm u0 + 2 * (2 * diam * (cost + gamma * hGamma)) :=
    hiter.trans (by
      have hsum := add_le_add_left htwice (variationSeminorm u0)
      simpa [add_assoc, add_left_comm, add_comm] using hsum)
  have hscale :
      variationSeminorm u0 + 2 * (2 * diam * (cost + gamma * hGamma)) =
        variationSeminorm u0 + 4 * diam * (cost + gamma * hGamma) := by
    ring_nf
  exact hscale ▸ hiter'

/--
Successor-index convenience form of
`graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_twoDiam_of_isTopical`.
-/
theorem graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_twoDiam_succ_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi₁ Psi₂ closedFormUpdate : (ι → ℝ) → (ι → ℝ))
    (hPsi₁ : IsTopical Psi₁)
    (hPsi₂ : IsTopical Psi₂)
    (hupdate : ∀ u : ι → ℝ, (Psi₂ ∘ Psi₁) u = closedFormUpdate u)
    {uStar u0 : ι → ℝ}
    (hfix : closedFormUpdate uStar = uStar)
    {kappa diam cost gamma hGamma : ℝ}
    (hbound : variationSeminorm uStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma)
    (hkappa : kappa ≤ 2 * diam)
    (hbase : 0 ≤ cost + gamma * hGamma)
    (k : ℕ) :
    variationSeminorm (((Psi₂ ∘ Psi₁)^[k + 1]) u0) ≤
      variationSeminorm u0 + 4 * diam * (cost + gamma * hGamma) := by
  simpa using
    graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_twoDiam_of_isTopical
      Psi₁ Psi₂ closedFormUpdate hPsi₁ hPsi₂ hupdate hfix hbound hkappa hbase (k + 1)

/--
Graph-`W₁` sweep iterate bound at the explicit `hGamma = log(n) / gamma` scale.

This is the closed-form bridge used by HGamma/Complexity layers:
`variationSeminorm ((Psi₂ ∘ Psi₁)^[k] u0) ≤ variationSeminorm u0 + 4 * diam * (cost + log n)`.
-/
theorem graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_twoDiam_logDivGamma_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi₁ Psi₂ closedFormUpdate : (ι → ℝ) → (ι → ℝ))
    (hPsi₁ : IsTopical Psi₁)
    (hPsi₂ : IsTopical Psi₂)
    (hupdate : ∀ u : ι → ℝ, (Psi₂ ∘ Psi₁) u = closedFormUpdate u)
    {uStar u0 : ι → ℝ}
    (hfix : closedFormUpdate uStar = uStar)
    {diam cost gamma : ℝ} {n : ℕ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm uStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (k : ℕ) :
    variationSeminorm (((Psi₂ ∘ Psi₁)^[k]) u0) ≤
      variationSeminorm u0 + 4 * diam * (cost + Real.log n) := by
  have hiter :
      variationSeminorm (((Psi₂ ∘ Psi₁)^[k]) u0) ≤
        variationSeminorm u0 +
          2 * PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma) :=
    graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_of_isTopical
      Psi₁ Psi₂ closedFormUpdate hPsi₁ hPsi₂ hupdate hfix hbound k
  rw [graphW1_twoTimes_hGammaBudget_twoDiam_logDivGamma_closedForm
    (diam := diam) (cost := cost) (n := n) hgamma] at hiter
  exact hiter

/--
Zero-seed explicit graph-`W₁` sweep iterate bound at `hGamma = log(n) / gamma`.
-/
theorem graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_twoDiam_logDivGamma_from_zero_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi₁ Psi₂ closedFormUpdate : (ι → ℝ) → (ι → ℝ))
    (hPsi₁ : IsTopical Psi₁)
    (hPsi₂ : IsTopical Psi₂)
    (hupdate : ∀ u : ι → ℝ, (Psi₂ ∘ Psi₁) u = closedFormUpdate u)
    {uStar u0 : ι → ℝ}
    (hfix : closedFormUpdate uStar = uStar)
    {diam cost gamma : ℝ} {n : ℕ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm uStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm (((Psi₂ ∘ Psi₁)^[k]) u0) ≤ 4 * diam * (cost + Real.log n) := by
  have hiter :
      variationSeminorm (((Psi₂ ∘ Psi₁)^[k]) u0) ≤
        variationSeminorm u0 + 4 * diam * (cost + Real.log n) :=
    graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_twoDiam_logDivGamma_of_isTopical
      Psi₁ Psi₂ closedFormUpdate hPsi₁ hPsi₂ hupdate hfix hgamma hbound k
  rw [hzero, zero_add] at hiter
  exact hiter

/--
Successor-index convenience form of the zero-seed explicit graph-`W₁` sweep bound.
-/
theorem graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_twoDiam_logDivGamma_from_zero_succ_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi₁ Psi₂ closedFormUpdate : (ι → ℝ) → (ι → ℝ))
    (hPsi₁ : IsTopical Psi₁)
    (hPsi₂ : IsTopical Psi₂)
    (hupdate : ∀ u : ι → ℝ, (Psi₂ ∘ Psi₁) u = closedFormUpdate u)
    {uStar u0 : ι → ℝ}
    (hfix : closedFormUpdate uStar = uStar)
    {diam cost gamma : ℝ} {n : ℕ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm uStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (hzero : variationSeminorm u0 = 0)
    (k : ℕ) :
    variationSeminorm (((Psi₂ ∘ Psi₁)^[k + 1]) u0) ≤ 4 * diam * (cost + Real.log n) := by
  simpa using
    graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_twoDiam_logDivGamma_from_zero_of_isTopical
      Psi₁ Psi₂ closedFormUpdate hPsi₁ hPsi₂ hupdate hfix hgamma hbound hzero (k + 1)

/--
Successor-index convenience form of the explicit graph-`W₁` sweep bound with base term.
-/
theorem graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_twoDiam_logDivGamma_succ_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi₁ Psi₂ closedFormUpdate : (ι → ℝ) → (ι → ℝ))
    (hPsi₁ : IsTopical Psi₁)
    (hPsi₂ : IsTopical Psi₂)
    (hupdate : ∀ u : ι → ℝ, (Psi₂ ∘ Psi₁) u = closedFormUpdate u)
    {uStar u0 : ι → ℝ}
    (hfix : closedFormUpdate uStar = uStar)
    {diam cost gamma : ℝ} {n : ℕ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm uStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (k : ℕ) :
    variationSeminorm (((Psi₂ ∘ Psi₁)^[k + 1]) u0) ≤
      variationSeminorm u0 + 4 * diam * (cost + Real.log n) := by
  simpa using
    graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_twoDiam_logDivGamma_of_isTopical
      Psi₁ Psi₂ closedFormUpdate hPsi₁ hPsi₂ hupdate hfix hgamma hbound (k + 1)

/--
Index-bounded convenience form of the explicit graph-`W₁` sweep bound with base term.
-/
theorem graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_twoDiam_logDivGamma_of_isTopical_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi₁ Psi₂ closedFormUpdate : (ι → ℝ) → (ι → ℝ))
    (hPsi₁ : IsTopical Psi₁)
    (hPsi₂ : IsTopical Psi₂)
    (hupdate : ∀ u : ι → ℝ, (Psi₂ ∘ Psi₁) u = closedFormUpdate u)
    {uStar u0 : ι → ℝ}
    (hfix : closedFormUpdate uStar = uStar)
    {diam cost gamma : ℝ} {n : ℕ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm uStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (k m : ℕ)
    (hmk : m ≤ k + 1) :
    variationSeminorm (((Psi₂ ∘ Psi₁)^[m]) u0) ≤
      variationSeminorm u0 + 4 * diam * (cost + Real.log n) := by
  have _hmk : m ≤ k + 1 := hmk
  exact
    graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_twoDiam_logDivGamma_of_isTopical
      Psi₁ Psi₂ closedFormUpdate hPsi₁ hPsi₂ hupdate hfix hgamma hbound m

/--
Index-bounded convenience form of the zero-seed explicit graph-`W₁` sweep bound.
-/
theorem graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_twoDiam_logDivGamma_from_zero_of_isTopical_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi₁ Psi₂ closedFormUpdate : (ι → ℝ) → (ι → ℝ))
    (hPsi₁ : IsTopical Psi₁)
    (hPsi₂ : IsTopical Psi₂)
    (hupdate : ∀ u : ι → ℝ, (Psi₂ ∘ Psi₁) u = closedFormUpdate u)
    {uStar u0 : ι → ℝ}
    (hfix : closedFormUpdate uStar = uStar)
    {diam cost gamma : ℝ} {n : ℕ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm uStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (hzero : variationSeminorm u0 = 0)
    (k m : ℕ)
    (hmk : m ≤ k + 1) :
    variationSeminorm (((Psi₂ ∘ Psi₁)^[m]) u0) ≤ 4 * diam * (cost + Real.log n) := by
  have _hmk : m ≤ k + 1 := hmk
  exact
    graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_twoDiam_logDivGamma_from_zero_of_isTopical
      Psi₁ Psi₂ closedFormUpdate hPsi₁ hPsi₂ hupdate hfix hgamma hbound hzero m

/--
Zero-function specialization of the explicit graph-`W₁` sweep bound.
-/
theorem graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_twoDiam_logDivGamma_zeroFn_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi₁ Psi₂ closedFormUpdate : (ι → ℝ) → (ι → ℝ))
    (hPsi₁ : IsTopical Psi₁)
    (hPsi₂ : IsTopical Psi₂)
    (hupdate : ∀ u : ι → ℝ, (Psi₂ ∘ Psi₁) u = closedFormUpdate u)
    {uStar : ι → ℝ}
    (hfix : closedFormUpdate uStar = uStar)
    {diam cost gamma : ℝ} {n : ℕ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm uStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (k : ℕ) :
    variationSeminorm (((Psi₂ ∘ Psi₁)^[k]) (0 : ι → ℝ)) ≤ 4 * diam * (cost + Real.log n) := by
  exact
    graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_twoDiam_logDivGamma_from_zero_of_isTopical
      Psi₁ Psi₂ closedFormUpdate hPsi₁ hPsi₂ hupdate hfix hgamma hbound
      (by simpa using (variationSeminorm_zero (ι := ι))) k

/--
Successor-index zero-function specialization of the explicit graph-`W₁` sweep bound.
-/
theorem graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_twoDiam_logDivGamma_zeroFn_succ_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi₁ Psi₂ closedFormUpdate : (ι → ℝ) → (ι → ℝ))
    (hPsi₁ : IsTopical Psi₁)
    (hPsi₂ : IsTopical Psi₂)
    (hupdate : ∀ u : ι → ℝ, (Psi₂ ∘ Psi₁) u = closedFormUpdate u)
    {uStar : ι → ℝ}
    (hfix : closedFormUpdate uStar = uStar)
    {diam cost gamma : ℝ} {n : ℕ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm uStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (k : ℕ) :
    variationSeminorm (((Psi₂ ∘ Psi₁)^[k + 1]) (0 : ι → ℝ)) ≤
      4 * diam * (cost + Real.log n) := by
  simpa using
    graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_twoDiam_logDivGamma_zeroFn_of_isTopical
      Psi₁ Psi₂ closedFormUpdate hPsi₁ hPsi₂ hupdate hfix hgamma hbound (k + 1)

/--
Index-bounded successor-index zero-function specialization of the explicit graph-`W₁` sweep bound.
-/
theorem graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_twoDiam_logDivGamma_zeroFn_succ_of_isTopical_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi₁ Psi₂ closedFormUpdate : (ι → ℝ) → (ι → ℝ))
    (hPsi₁ : IsTopical Psi₁)
    (hPsi₂ : IsTopical Psi₂)
    (hupdate : ∀ u : ι → ℝ, (Psi₂ ∘ Psi₁) u = closedFormUpdate u)
    {uStar : ι → ℝ}
    (hfix : closedFormUpdate uStar = uStar)
    {diam cost gamma : ℝ} {n : ℕ}
    (hgamma : 0 < gamma)
    (hbound : variationSeminorm uStar ≤
      PrimalDualBounds.hGammaKappaBudget (2 * diam) cost gamma (Real.log n / gamma))
    (k m : ℕ)
    (hmk : m + 1 ≤ k + 1) :
    variationSeminorm (((Psi₂ ∘ Psi₁)^[m + 1]) (0 : ι → ℝ)) ≤
      4 * diam * (cost + Real.log n) := by
  have _hmk : m + 1 ≤ k + 1 := hmk
  exact
    graphW1_flowSinkhorn_sweep_variationSeminorm_iterate_bound_twoDiam_logDivGamma_zeroFn_succ_of_isTopical
      Psi₁ Psi₂ closedFormUpdate hPsi₁ hPsi₂ hupdate hfix hgamma hbound m

open FlowSinkhorn.KLProjection in
/--
Concrete non-expansiveness proof for the graph-W₁ Ψ₂ block update.

Proposition `prop:Psi2_closed_nonexp`: the closed-form update `Ψ₂(v)_{i,j} = (v_j - v_i)/2`
satisfies `variationSeminorm (Ψ₂ v) ≤ variationSeminorm v`.

Proof: for any edge `(i,j)`, `|(v_j - v_i)/2| ≤ (coordMax v - coordMin v)/2 = variationSeminorm v`,
so by `variationSeminorm_le_of_shifted_supnorm_bound` (with shift `c = 0`) the result follows.
-/
theorem graphW1_Psi2_variationSeminorm_nonexpansive
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (v : ι → ℝ) :
    variationSeminorm (fun p : ι × ι => (v p.2 - v p.1) / 2) ≤ variationSeminorm v := by
  apply variationSeminorm_le_of_shifted_supnorm_bound _ (c := 0)
  intro ⟨i, j⟩
  simp only [add_zero]
  have hj  : v j ≤ coordMax v  := le_coordMax v j
  have hi  : coordMin v ≤ v i  := coordMin_le v i
  have hj' : coordMin v ≤ v j  := coordMin_le v j
  have hi' : v i ≤ coordMax v  := le_coordMax v i
  have hkey : |v j - v i| ≤ oscillation v := by
    unfold oscillation
    rw [abs_le]
    constructor <;> linarith
  calc |(v j - v i) / 2|
      = |v j - v i| / 2 := by rw [abs_div, abs_of_pos (by norm_num : (0 : ℝ) < 2)]
    _ ≤ oscillation v / 2 := by nlinarith
    _ = variationSeminorm v := rfl

/--
Consequence of Psi2 non-expansiveness for graph-W₁: the variation seminorm of
Psi2(v) is bounded by the variation seminorm of v.

This is the key nonexpansiveness fact used in the application of the generic blueprint
for graph-W₁ convergence, from Proposition prop:Psi2_closed_nonexp.
-/
theorem graphW1_Psi2_blockQuotient_le
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (v : ι → ℝ) :
    variationSeminorm (fun p : ι × ι => (v p.2 - v p.1) / 2) ≤ variationSeminorm v :=
  graphW1_Psi2_variationSeminorm_nonexpansive v

/--
Concrete statement of Proposition `prop:V1V2_closed_form_flow` for the vertex-potential block.

The block-quotient seminorm for `v : ι → ℝ` (the V₁ component) equals the variation seminorm.
This follows immediately from `blockQuotientSeminorm_eq_variationSeminorm`: quotienting `‖·‖_∞`
on `ℝ^n` by constant shifts is exactly the variation seminorm.
-/
theorem graphW1_V1_eq_variationSeminorm
    {ι : Type*} [Fintype ι] [Nonempty ι] (v : ι → ℝ) :
    Setup.blockQuotientSeminorm v = variationSeminorm v :=
  Setup.blockQuotientSeminorm_eq_variationSeminorm v

/--
Concrete statement of Proposition `prop:V1V2_closed_form_flow` for the edge-flow block.

The block-quotient seminorm for `U : ι × ι → ℝ` (the V₂ component) equals the variation seminorm
of `U` viewed as a function on pairs.
-/
theorem graphW1_V2_eq_variationSeminorm
    {ι : Type*} [Fintype ι] [Nonempty ι] (U : ι × ι → ℝ) :
    Setup.blockQuotientSeminorm U = variationSeminorm U :=
  Setup.blockQuotientSeminorm_eq_variationSeminorm U

/--
Consequence alias: the block-quotient seminorm of a vertex potential is the variation seminorm.
This repackages `graphW1_V1_eq_variationSeminorm` as the explicit equality stated in
Proposition `prop:V1V2_closed_form_flow`.
-/
theorem graphW1_blockQuotient_eq_variationSeminorm
    {ι : Type*} [Fintype ι] [Nonempty ι] (v : ι → ℝ) :
    Setup.blockQuotientSeminorm v = variationSeminorm v :=
  graphW1_V1_eq_variationSeminorm v

open FlowSinkhorn.KLProjection in
open FlowSinkhorn.KLProjection.PrimalDualBounds in
/--
Concrete non-expansiveness for graph-W₁ Ψ₂ operator using the topical bridge.

Given that the Ψ₂ operator is monotone (from `blockUpdate_monotone`) and
translation-equivariant (from `sweep_translationEquivariant` / `TranslationEquivariant`),
the bridge theorem `topical_implies_variationSeminorm_nonexpansive` directly gives
`SeminormNonexpansive variationSeminormAsSeminorm Ψ₂`.
-/
theorem graphW1_Psi2_variationSeminorm_nonexpansive_of_topical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi2 : (ι → ℝ) → (ι → ℝ))
    (hmono : Monotone Psi2)
    (htrans : TranslationEquivariant Psi2) :
    SeminormNonexpansive variationSeminormAsSeminorm Psi2 :=
  topical_implies_variationSeminorm_nonexpansive Psi2 hmono htrans

open FlowSinkhorn.KLProjection in
/--
Nonexpansiveness of a graph-W₁ block map from a bundled `IsTopical` certificate.

This is the `IsTopical`-facing variant of `graphW1_Psi2_variationSeminorm_nonexpansive_of_topical`.
-/
theorem graphW1_Psi2_variationSeminorm_nonexpansive_of_isTopical
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi2 : (ι → ℝ) → (ι → ℝ))
    (hT : IsTopical Psi2) :
    SeminormNonexpansive variationSeminormAsSeminorm Psi2 :=
  graphW1_Psi2_variationSeminorm_nonexpansive_of_topical Psi2 hT.mono hT.trans

end GraphW1
end Applications
end KLProjection
end FlowSinkhorn
