import FlowSinkhorn.KLProjection.Applications.GraphW1.ClosedFormsVocabulary
import FlowSinkhorn.KLProjection.PrimalDualBounds.Blueprint
import FlowSinkhorn.KLProjection.Variation
import Mathlib.Analysis.SpecialFunctions.Arsinh

/-!
# Closed-form graph-`W₁` block updates

This module is reserved for the closed-form block analysis from
the graph-W1 material in `neurips/paper.tex`.

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
The scalar `phi(t,u)` is positive when the mass ratio `u` is positive.
-/
theorem graphW1_phi_pos
    (t u : ℝ)
    (hu : 0 < u) :
    0 < graphW1_phi t u := by
  have hsqrt_gt : t < Real.sqrt (t ^ 2 + 4 * u) := by
    by_cases ht : 0 ≤ t
    · exact (Real.lt_sqrt ht).2 (by nlinarith [hu])
    · exact lt_of_lt_of_le (lt_of_not_ge ht) (Real.sqrt_nonneg _)
  unfold graphW1_phi
  exact div_pos (sub_pos.mpr hsqrt_gt) (by norm_num)

/--
The coordinate scaling used by `C1` is strictly positive under positive row and column masses.
-/
theorem graphW1_C1Scaling_pos
    {ι : Type*}
    (bDiff hRow hCol : ι → ℝ)
    (hrow_pos : ∀ i : ι, 0 < hRow i)
    (hcol_pos : ∀ i : ι, 0 < hCol i) :
    ∀ i : ι, 0 < graphW1_C1Scaling bDiff hRow hCol i := by
  intro i
  exact graphW1_phi_pos _ _ (div_pos (hcol_pos i) (hrow_pos i))

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

/--
Paper-facing formula package for Proposition `prop:graphw1-projection-closed-form`.

This statement introduces named Lean maps for the displayed candidates
`Proj_C1(h,h) = (diag(s)h, h diag(s)^{-1})` and
`Proj_C2(f,g) = (sqrt(f⊙g), sqrt(f⊙g))`.  It proves the explicit component
formulas, strict positivity of the scaling `s`, the coordinate `C1` constraint
equation, and the coordinate `C2` product equation.

The theorem certifies the closed-form algebra of the paper statement.  The
variational fact that these candidates are the unique KL minimizers remains
outside this endpoint.
-/
theorem graphW1_projection_closedForm_maps
    {ι : Type*}
    (bDiff hRow hCol : ι → ℝ)
    (h f g : ι → ι → ℝ)
    (hrow_pos : ∀ i : ι, 0 < hRow i)
    (hcol_pos : ∀ i : ι, 0 < hCol i)
    (hfg : ∀ i j : ι, 0 ≤ f i j * g i j) :
    (∀ i : ι, 0 < graphW1_C1Scaling bDiff hRow hCol i) ∧
    (∀ i j : ι,
      graphW1_projC1Left bDiff hRow hCol h i j =
          graphW1_C1Scaling bDiff hRow hCol i * h i j ∧
        graphW1_projC1Right bDiff hRow hCol h i j =
          h i j / graphW1_C1Scaling bDiff hRow hCol j) ∧
    (∀ i : ι,
      let s := graphW1_C1Scaling bDiff hRow hCol i
      (s ^ 2) * hRow i + s * bDiff i - hCol i = 0) ∧
    (∀ i j : ι,
      graphW1_projC2Common f g i j = Real.sqrt (f i j * g i j) ∧
        (graphW1_projC2Common f g i j) ^ 2 = f i j * g i j ∧
          0 ≤ graphW1_projC2Common f g i j) := by
  have hscaling_pos :
      ∀ i : ι, 0 < graphW1_C1Scaling bDiff hRow hCol i :=
    graphW1_C1Scaling_pos bDiff hRow hCol hrow_pos hcol_pos
  have hdisc :
      ∀ i : ι, 0 ≤ (bDiff i / hRow i) ^ 2 + 4 * (hCol i / hRow i) := by
    intro i
    exact add_nonneg (sq_nonneg _) <|
      mul_nonneg (by norm_num) (le_of_lt (div_pos (hcol_pos i) (hrow_pos i)))
  constructor
  · exact hscaling_pos
  constructor
  · intro i j
    constructor <;> rfl
  constructor
  · intro i
    exact graphW1_projection_C1_coordinate
      (bDiff i) (hRow i) (hCol i) (ne_of_gt (hrow_pos i)) (hdisc i)
  · intro i j
    constructor
    · rfl
    · exact graphW1_projection_C2_coordinate (f i j) (g i j) (hfg i j)

/--
Paper-facing variational package for Proposition `prop:graphw1-projection-closed-form`.

Compared with `graphW1_projection_closedForm_maps`, this endpoint also exposes the projection
meaning of the two displayed candidates.  Lean derives the `C1` constraint from the row/column
sum identities and the quadratic root, records the `C2` diagonal constraint definitionally, and
returns the named finite-KL optimality predicates supplied by
`GraphW1ProjectionVariationalCertificate`.  The statement also records nonnegativity of the
displayed `C1` projection candidate, matching the paper's flow cone.
-/
theorem graphW1_projection_closedForm_maps_with_variationalCertificate
    {ι : Type*} [Fintype ι]
    (bDiff hRow hCol : ι → ℝ)
    (h f g : ι → ι → ℝ)
    (hrow_pos : ∀ i : ι, 0 < hRow i)
    (hcol_pos : ∀ i : ι, 0 < hCol i)
    (hcert : GraphW1ProjectionVariationalCertificate bDiff hRow hCol h f g) :
    (∀ i : ι, 0 < graphW1_C1Scaling bDiff hRow hCol i) ∧
    (∀ i j : ι,
      graphW1_projC1Left bDiff hRow hCol h i j =
          graphW1_C1Scaling bDiff hRow hCol i * h i j ∧
        graphW1_projC1Right bDiff hRow hCol h i j =
          h i j / graphW1_C1Scaling bDiff hRow hCol j) ∧
    (∀ i : ι,
      let s := graphW1_C1Scaling bDiff hRow hCol i
      (s ^ 2) * hRow i + s * bDiff i - hCol i = 0) ∧
    GraphW1C1Constraint bDiff
      (graphW1_projC1Left bDiff hRow hCol h)
      (graphW1_projC1Right bDiff hRow hCol h) ∧
    GraphW1PairNonnegative
      (graphW1_projC1Left bDiff hRow hCol h)
      (graphW1_projC1Right bDiff hRow hCol h) ∧
    GraphW1C1ProjectionOptimality bDiff h
      (graphW1_projC1Left bDiff hRow hCol h)
      (graphW1_projC1Right bDiff hRow hCol h) ∧
    (∀ i j : ι,
      graphW1_projC2Common f g i j = Real.sqrt (f i j * g i j) ∧
        (graphW1_projC2Common f g i j) ^ 2 = f i j * g i j ∧
          0 ≤ graphW1_projC2Common f g i j) ∧
    GraphW1C2ProjectionOptimality f g (graphW1_projC2Common f g) := by
  classical
  let s : ι → ℝ := graphW1_C1Scaling bDiff hRow hCol
  have hfg : ∀ i j : ι, 0 ≤ f i j * g i j := by
    intro i j
    exact mul_nonneg (hcert.fg_nonneg.1 i j) (hcert.fg_nonneg.2 i j)
  have hmaps :=
    graphW1_projection_closedForm_maps bDiff hRow hCol h f g hrow_pos hcol_pos hfg
  have hconstraint :
      GraphW1C1Constraint bDiff
        (graphW1_projC1Left bDiff hRow hCol h)
        (graphW1_projC1Right bDiff hRow hCol h) := by
    intro i
    have hs_pos : 0 < s i := by
      simpa [s] using hmaps.1 i
    have hs_ne : s i ≠ 0 := ne_of_gt hs_pos
    have hleft_sum :
        (∑ j : ι, graphW1_projC1Left bDiff hRow hCol h i j) = s i * hRow i := by
      calc
        (∑ j : ι, graphW1_projC1Left bDiff hRow hCol h i j)
            = ∑ j : ι, s i * h i j := by
                simp [s, graphW1_projC1Left]
        _ = s i * ∑ j : ι, h i j := by
                rw [Finset.mul_sum]
        _ = s i * hRow i := by
                rw [← hcert.row_sum i]
    have hright_sum :
        (∑ j : ι, graphW1_projC1Right bDiff hRow hCol h j i) = hCol i / s i := by
      calc
        (∑ j : ι, graphW1_projC1Right bDiff hRow hCol h j i)
            = ∑ j : ι, h j i / s i := by
                simp [s, graphW1_projC1Right]
        _ = (∑ j : ι, h j i) / s i := by
                rw [Finset.sum_div]
        _ = hCol i / s i := by
                rw [← hcert.col_sum i]
    have hquad :
        (s i ^ 2) * hRow i + s i * bDiff i - hCol i = 0 := by
      simpa [s] using hmaps.2.2.1 i
    have hscalar : -(s i * hRow i) + hCol i / s i = bDiff i := by
      have hcol_eq : hCol i = (s i ^ 2) * hRow i + s i * bDiff i := by
        linarith
      rw [hcol_eq]
      field_simp [hs_ne]
      ring
    calc
      -(∑ j : ι, graphW1_projC1Left bDiff hRow hCol h i j) +
          (∑ j : ι, graphW1_projC1Right bDiff hRow hCol h j i)
          = -(s i * hRow i) + hCol i / s i := by
              rw [hleft_sum, hright_sum]
      _ = bDiff i := hscalar
  have hc1_nonneg :
      GraphW1PairNonnegative
        (graphW1_projC1Left bDiff hRow hCol h)
        (graphW1_projC1Right bDiff hRow hCol h) := by
    refine ⟨?_, ?_⟩
    · intro i j
      exact mul_nonneg (le_of_lt (hmaps.1 i)) (hcert.h_nonneg i j)
    · intro i j
      exact div_nonneg (hcert.h_nonneg i j) (le_of_lt (hmaps.1 j))
  exact
    ⟨hmaps.1, hmaps.2.1, hmaps.2.2.1, hconstraint,
      hc1_nonneg, hcert.c1_optimality, hmaps.2.2.2, hcert.c2_optimality⟩

/-!
## Projection-to-update algebra

The closed-form `C1` projection root feeds the Flow-Sinkhorn update with
`t = 2*r/b` and `u = a/b`.  The lemmas in this block package the positivity assumptions
needed to rewrite that root into the square-root update used by Proposition `prop:flow-algo`.
-/

/--
The `C1` closed-form root with parameters `t = 2*r/b`, `u = a/b` is exactly the
Flow-Sinkhorn numerator divided by `b`.
-/
theorem graphW1_phi_flowSinkhorn_ratio
    (r a b : ℝ)
    (hb : 0 < b)
    (hab : 0 ≤ a * b) :
    graphW1_phi (2 * r / b) (a / b) =
      (Real.sqrt (r ^ 2 + a * b) - r) / b := by
  have hb_ne : b ≠ 0 := ne_of_gt hb
  have hrad : 0 ≤ r ^ 2 + a * b := by nlinarith [sq_nonneg r, hab]
  have hsqrt :
      Real.sqrt ((2 * r / b) ^ 2 + 4 * (a / b)) =
        (2 / b) * Real.sqrt (r ^ 2 + a * b) := by
    have hscale_nonneg : 0 ≤ (2 / b) * Real.sqrt (r ^ 2 + a * b) := by
      exact mul_nonneg (le_of_lt (div_pos (by norm_num) hb)) (Real.sqrt_nonneg _)
    have harg :
        (2 * r / b) ^ 2 + 4 * (a / b) =
          ((2 / b) * Real.sqrt (r ^ 2 + a * b)) ^ 2 := by
      have hleft :
          (2 * r / b) ^ 2 + 4 * (a / b) = 4 * (r ^ 2 + a * b) / b ^ 2 := by
        field_simp [hb_ne]
        ring
      have hright :
          ((2 / b) * Real.sqrt (r ^ 2 + a * b)) ^ 2 =
            4 * (r ^ 2 + a * b) / b ^ 2 := by
        rw [mul_pow, div_pow, Real.sq_sqrt hrad]
        ring
      exact hleft.trans hright.symm
    rw [harg, Real.sqrt_sq_eq_abs, abs_of_nonneg hscale_nonneg]
  unfold graphW1_phi
  rw [hsqrt]
  field_simp [hb_ne]

/--
Closed-form projection/update bridge for one coordinate.

The same scalar obtained from the `C1` projection equation is the numerator used in the
stable Flow-Sinkhorn update after division by the current scale `s`.
-/
theorem graphW1_projection_C1_flowSinkhorn_bridge
    (r a b s : ℝ)
    (hb : 0 < b)
    (hs : 0 < s)
    (hab : 0 ≤ a * b) :
    let sigma := graphW1_phi (2 * r / b) (a / b)
    ((sigma ^ 2) * b + sigma * (2 * r) - a = 0) ∧
      sigma / s = (Real.sqrt (r ^ 2 + a * b) - r) / (s * b) := by
  intro sigma
  have hb_ne : b ≠ 0 := ne_of_gt hb
  have hs_ne : s ≠ 0 := ne_of_gt hs
  have hrad : 0 ≤ r ^ 2 + a * b := by nlinarith [sq_nonneg r, hab]
  have hdisc : 0 ≤ (2 * r / b) ^ 2 + 4 * (a / b) := by
    have harg : (2 * r / b) ^ 2 + 4 * (a / b) = 4 * (r ^ 2 + a * b) / b ^ 2 := by
      field_simp [hb_ne]
      ring
    rw [harg]
    exact div_nonneg (mul_nonneg (by norm_num) hrad) (sq_nonneg b)
  constructor
  · exact graphW1_projection_C1_coordinate (2 * r) b a hb_ne hdisc
  · have hphi :
        sigma = (Real.sqrt (r ^ 2 + a * b) - r) / b := by
      simpa [sigma] using graphW1_phi_flowSinkhorn_ratio r a b hb hab
    rw [hphi]
    field_simp [hb_ne, hs_ne]

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
The closed-form intermediate ratio `q` is nonnegative under the natural product
assumption `0 ≤ a*b` and positive denominator assumptions.
-/
theorem graphW1_flowSinkhorn_q_nonneg
    (r a b s q : ℝ)
    (hb : 0 < b)
    (hs : 0 < s)
    (hab : 0 ≤ a * b)
    (hq :
      q = (Real.sqrt (r ^ 2 + a * b) - r) / (s * b)) :
    0 ≤ q := by
  rw [hq]
  apply div_nonneg
  · have hrad : 0 ≤ r ^ 2 + a * b := by nlinarith [sq_nonneg r, hab]
    have hsqr : r ^ 2 ≤ r ^ 2 + a * b := by nlinarith [hab]
    have habs_le : |r| ≤ Real.sqrt (r ^ 2 + a * b) := by
      rw [← Real.sqrt_sq_eq_abs r]
      exact Real.sqrt_le_sqrt hsqr
    exact sub_nonneg.mpr (le_trans (le_abs_self r) habs_le)
  · exact mul_nonneg (le_of_lt hs) (le_of_lt hb)

/--
Packaged scalar Flow-Sinkhorn update: `q ≥ 0` is derived internally from `0 ≤ a*b`.
-/
theorem graphW1_flowSinkhorn_update_of_product_nonneg
    (r a b s q sNext : ℝ)
    (hb : 0 < b)
    (hs : 0 < s)
    (hab : 0 ≤ a * b)
    (hq :
      q = (Real.sqrt (r ^ 2 + a * b) - r) / (s * b))
    (hsNext : sNext = s * Real.sqrt q) :
    sNext = Real.sqrt ((s / b) * (Real.sqrt (r ^ 2 + a * b) - r)) := by
  exact graphW1_flowSinkhorn_update r a b s q sNext hb hs hq
    (graphW1_flowSinkhorn_q_nonneg r a b s q hb hs hab hq) hsNext

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
Packaged vector Flow-Sinkhorn update: coordinatewise nonnegativity of `q` is derived
from `0 ≤ a_i*b_i`.
-/
theorem graphW1_flowSinkhorn_update_vector_of_product_nonneg
    {ι : Type*}
    (r a b s q sNext : ι → ℝ)
    (hb : ∀ i : ι, 0 < b i)
    (hs : ∀ i : ι, 0 < s i)
    (hab : ∀ i : ι, 0 ≤ a i * b i)
    (hq :
      ∀ i : ι,
        q i = (Real.sqrt (r i ^ 2 + a i * b i) - r i) / (s i * b i))
    (hsNext : ∀ i : ι, sNext i = s i * Real.sqrt (q i)) :
    sNext =
      fun i =>
        Real.sqrt ((s i / b i) * (Real.sqrt (r i ^ 2 + a i * b i) - r i)) := by
  exact graphW1_flowSinkhorn_update_vector r a b s q sNext hb hs hq
    (fun i => graphW1_flowSinkhorn_q_nonneg
      (r i) (a i) (b i) (s i) (q i) (hb i) (hs i) (hab i) (hq i))
    hsNext

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
Paper-shaped Proposition `prop:flow-algo` with the nonnegativity side condition packaged as
nonnegativity of the forward row sum.  Since the backward row sum is already assumed positive,
the scalar `q ≥ 0` condition is discharged internally.
-/
theorem graphW1_flowSinkhorn_update_as_stated_of_forward_nonneg
    {ι : Type*} [Fintype ι]
    (zC : ι → ι → ℝ)
    (r s q sNext : ι → ℝ)
    (hb :
      ∀ i : ι,
        0 < Finset.univ.sum (fun j : ι => zC i j * (1 / s j)))
    (hs : ∀ i : ι, 0 < s i)
    (ha :
      ∀ i : ι,
        0 ≤ Finset.univ.sum (fun j : ι => zC i j * s j))
    (hq :
      ∀ i : ι,
        q i =
          (Real.sqrt
            (r i ^ 2 +
              (Finset.univ.sum (fun j : ι => zC i j * s j)) *
              (Finset.univ.sum (fun j : ι => zC i j * (1 / s j))) ) - r i) /
            (s i * (Finset.univ.sum (fun j : ι => zC i j * (1 / s j))))
      )
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
  exact graphW1_flowSinkhorn_update_as_stated zC r s q sNext hb hs hq
    (fun i => graphW1_flowSinkhorn_q_nonneg
      (r i)
      (Finset.univ.sum (fun j : ι => zC i j * s j))
      (Finset.univ.sum (fun j : ι => zC i j * (1 / s j)))
      (s i)
      (q i)
      (hb i)
      (hs i)
      (mul_nonneg (ha i) (le_of_lt (hb i)))
      (hq i))
    hsNext

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
Paper-facing stable dual-update package for Proposition
`prop:graphw1-flow-sinkhorn-update`.

The statement exposes the actual dual map `Ψ` and the stable `arsinh` formula from
Equation `eq:v-update-stable`.  The log-sum-exp quantities
`alphaPlus` and `alphaMinus`, and the scalar `beta`, are provided as already-computed
algorithmic fields; this theorem certifies the sign-sensitive algebra turning the
implementation update into the displayed dual formula.
-/
theorem graphW1_flowSinkhorn_stableDualUpdate_from_code
    {ι : Type*}
    (Psi : (ι → ℝ) → ι → ℝ)
    (v h hNext m alphaPlus alphaMinus beta : ι → ℝ)
    (gamma : ℝ)
    (hgamma : gamma ≠ 0)
    (hv : ∀ i : ι, v i = 2 * h i)
    (hPsi : Psi v = fun i : ι => 2 * hNext i)
    (hhNext : ∀ i : ι, hNext i = h i / 2 - gamma / 2 * m i)
    (hm :
      ∀ i : ι, m i = (alphaMinus i - alphaPlus i) / (2 * gamma) + Real.arsinh (beta i)) :
    Psi v =
      (fun i : ι =>
        (1 / 2) * v i + (1 / 2) * (alphaPlus i - alphaMinus i) -
          gamma * Real.arsinh (beta i)) := by
  rw [hPsi]
  exact graphW1_vUpdate_stable_correct_from_code_vector
    v h hNext m alphaPlus alphaMinus beta gamma hgamma hv hhNext hm

/--
Paper-facing stable dual-update package with the log-sum-exp fields internalized.

Compared with `graphW1_flowSinkhorn_stableDualUpdate_from_code`, this endpoint no longer treats
`α⁺`, `α⁻`, and `β` as opaque precomputed inputs: they are the finite log-sum-exp quantities
displayed in Proposition `prop:graphw1-flow-sinkhorn-update`.  The remaining bridge hypothesis is
the implementation identity for the intermediate variable `m`, which is the next projection-level
fact to internalize.
-/
theorem graphW1_flowSinkhorn_stableDualUpdate_logsumexp
    {ι : Type*} [Fintype ι]
    (Psi : (ι → ℝ) → ι → ℝ)
    (v h hNext m bDiff : ι → ℝ)
    (w : ι → ι → ℝ)
    (gamma : ℝ)
    (hgamma : gamma ≠ 0)
    (hv : ∀ i : ι, v i = 2 * h i)
    (hPsi : Psi v = fun i : ι => 2 * hNext i)
    (hhNext : ∀ i : ι, hNext i = h i / 2 - gamma / 2 * m i)
    (hm :
      ∀ i : ι,
        m i =
          (graphW1_alphaMinus w gamma v i - graphW1_alphaPlus w gamma v i) / (2 * gamma) +
            Real.arsinh (graphW1_beta bDiff w gamma v i)) :
    Psi v =
      (fun i : ι =>
        (1 / 2) * v i +
          (1 / 2) * (graphW1_alphaPlus w gamma v i - graphW1_alphaMinus w gamma v i) -
            gamma * Real.arsinh (graphW1_beta bDiff w gamma v i)) := by
  exact graphW1_flowSinkhorn_stableDualUpdate_from_code
    (Psi := Psi) (v := v) (h := h) (hNext := hNext) (m := m)
    (alphaPlus := graphW1_alphaPlus w gamma v)
    (alphaMinus := graphW1_alphaMinus w gamma v)
    (beta := graphW1_beta bDiff w gamma v)
    (gamma := gamma) hgamma hv hPsi hhNext hm

/--
Paper-facing stable dual-update package from the Lean projection-map update.

This removes the explicit intermediate `m` variable from the paper-facing statement.  The
remaining implementation bridge is the single statement that the dual sweep `Psi` is represented
by the Lean update `2 * graphW1_hNextFromDual`; the log-sum-exp fields and `beta` are all
defined internally.
-/
theorem graphW1_flowSinkhorn_stableDualUpdate_from_projectionMap
    {ι : Type*} [Fintype ι]
    (Psi : (ι → ℝ) → ι → ℝ)
    (v bDiff : ι → ℝ)
    (w : ι → ι → ℝ)
    (gamma : ℝ)
    (hgamma : gamma ≠ 0)
    (hPsi :
      Psi v =
        fun i : ι => 2 * graphW1_hNextFromDual bDiff w gamma v i) :
    Psi v =
      (fun i : ι =>
        (1 / 2) * v i +
          (1 / 2) * (graphW1_alphaPlus w gamma v i - graphW1_alphaMinus w gamma v i) -
            gamma * Real.arsinh (graphW1_beta bDiff w gamma v i)) := by
  rw [hPsi]
  funext i
  unfold graphW1_hNextFromDual graphW1_mUpdate
  field_simp [hgamma]
  ring

/--
Paper-facing stable dual-update theorem for the concrete Lean sweep map.

This is the cleanest current Proposition `prop:graphw1-flow-sinkhorn-update` endpoint: the sweep
map, the log-sum-exp fields, the `beta` field, and the intermediate half-dual update are all Lean
definitions.  No arbitrary `Psi`, `m`, `h`, or `hNext` variables remain in the statement.
-/
theorem graphW1_flowSinkhorn_stableDualUpdate_concreteMap
    {ι : Type*} [Fintype ι]
    (v bDiff : ι → ℝ)
    (w : ι → ι → ℝ)
    (gamma : ℝ)
    (hgamma : gamma ≠ 0) :
    graphW1_stableDualSweep bDiff w gamma v =
      (fun i : ι =>
        (1 / 2) * v i +
          (1 / 2) * (graphW1_alphaPlus w gamma v i - graphW1_alphaMinus w gamma v i) -
            gamma * Real.arsinh (graphW1_beta bDiff w gamma v i)) := by
  exact graphW1_flowSinkhorn_stableDualUpdate_from_projectionMap
    (Psi := graphW1_stableDualSweep bDiff w gamma)
    (v := v)
    (bDiff := bDiff)
    (w := w)
    (gamma := gamma)
    hgamma
    rfl

/--
Pointwise-block form of Proposition `prop:graphw1-flow-sinkhorn-update`.

This variant removes the named sweep certificate from the paper-facing theorem statement.  The
remaining implementation bridge is exposed directly as the two pointwise block identities: the
second block computes the Lean-defined log-domain `m` update, and the first block maps that `m`
update to the displayed stable dual step.
-/
theorem graphW1_flowSinkhorn_stableDualUpdate_from_pointwiseBlockIdentities
    {ι : Type*} [Fintype ι]
    (Ψ₁ Ψ₂ : (ι → ℝ) → ι → ℝ)
    (v bDiff : ι → ℝ)
    (w : ι → ι → ℝ)
    (gamma : ℝ)
    (hgamma : gamma ≠ 0)
    (hsecond :
      ∀ v : ι → ℝ, Ψ₂ v = graphW1_mUpdate bDiff w gamma v)
    (hfirst :
      ∀ v : ι → ℝ,
        Ψ₁ (graphW1_mUpdate bDiff w gamma v) =
          fun i : ι => v i / 2 - gamma * graphW1_mUpdate bDiff w gamma v i) :
    (Ψ₁ ∘ Ψ₂) v =
      (fun i : ι =>
        (1 / 2) * v i +
          (1 / 2) * (graphW1_alphaPlus w gamma v i - graphW1_alphaMinus w gamma v i) -
            gamma * Real.arsinh (graphW1_beta bDiff w gamma v i)) := by
  have hcomp :
      (Ψ₁ ∘ Ψ₂) v =
        fun i : ι => v i / 2 - gamma * graphW1_mUpdate bDiff w gamma v i := by
    unfold Function.comp
    rw [hsecond v]
    exact hfirst v
  rw [hcomp]
  funext i
  unfold graphW1_mUpdate
  field_simp [hgamma]
  ring

/--
Paper-facing block-sweep form of Proposition `prop:graphw1-flow-sinkhorn-update`.

This endpoint explicitly mentions the two block projection maps `Ψ₁` and `Ψ₂` from the paper.
The proof-free certificate `GraphW1StableProjectionSweepCertificate` is the remaining statement
that these concrete projection blocks produce the Lean-defined log-domain intermediate `m` and
then reconstruct the stable dual step from it.  Lean then unfolds the intermediate update and
proves the displayed `arsinh` formula for the composed sweep.
-/
theorem graphW1_flowSinkhorn_stableDualUpdate_from_blockSweepCertificate
    {ι : Type*} [Fintype ι]
    (Ψ₁ Ψ₂ : (ι → ℝ) → ι → ℝ)
    (v bDiff : ι → ℝ)
    (w : ι → ι → ℝ)
    (gamma : ℝ)
    (hgamma : gamma ≠ 0)
    (hblock : GraphW1StableProjectionSweepCertificate Ψ₁ Ψ₂ bDiff w gamma) :
    (Ψ₁ ∘ Ψ₂) v =
      (fun i : ι =>
        (1 / 2) * v i +
          (1 / 2) * (graphW1_alphaPlus w gamma v i - graphW1_alphaMinus w gamma v i) -
            gamma * Real.arsinh (graphW1_beta bDiff w gamma v i)) := by
  have hcomp :
      (Ψ₁ ∘ Ψ₂) v =
        fun i : ι => v i / 2 - gamma * graphW1_mUpdate bDiff w gamma v i := by
    unfold Function.comp
    rw [hblock.second_block_eq_mUpdate v]
    exact hblock.first_block_eq_from_mUpdate v
  rw [hcomp]
  funext i
  unfold graphW1_mUpdate
  field_simp [hgamma]
  ring

/--
Closed-form expression for the graph-`W₁` block quotient constant.

Concrete paper-facing statement of Proposition `prop:V1V2_closed_form_flow`.

It certifies that the quotient seminorm on both blocks (`V1` for vertex potentials, `V2` for
edge-flow tensors) is exactly the variation seminorm.
-/
theorem graphW1_blockQuotient_closedForm
    {vertex edge : Type*}
    [Fintype vertex] [Nonempty vertex]
    [Fintype edge] [Nonempty edge]
    (v : vertex → ℝ)
    (U : edge → ℝ) :
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
Closed form and non-expansiveness of `Ψ₂` on an explicit finite edge set.

This is the paper-facing content of Proposition `prop:graphw1-psi2-closed-nonexp`: the
second block map has the displayed formula, and its variation seminorm is bounded by the
variation seminorm of the vertex potential.
-/
theorem graphW1_Psi2_closedForm_nonexpansive
    {ι ε : Type*} [Fintype ι] [Nonempty ι] [Fintype ε] [Nonempty ε]
    (src dst : ε → ι)
    (v : ι → ℝ) :
    (∀ e : ε, graphW1_Psi2 src dst v e = (v (dst e) - v (src e)) / 2) ∧
      variationSeminorm (graphW1_Psi2 src dst v) ≤ variationSeminorm v := by
  constructor
  · intro e
    rfl
  · apply variationSeminorm_le_of_shifted_supnorm_bound _ (c := 0)
    intro e
    simp only [graphW1_Psi2, add_zero]
    have hkey : |v (dst e) - v (src e)| ≤ oscillation v := by
      unfold oscillation
      rw [abs_le]
      constructor
      · linarith [coordMin_le v (dst e), le_coordMax v (src e)]
      · linarith [coordMin_le v (src e), le_coordMax v (dst e)]
    calc
      |(v (dst e) - v (src e)) / 2|
          = |v (dst e) - v (src e)| / 2 := by
            rw [abs_div, abs_of_pos (by norm_num : (0 : ℝ) < 2)]
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
Paper-facing non-expansiveness package for Proposition `prop:graphw1-signed-structure`.

This endpoint states the conclusion for the full paper-order sweep `Ψ₁ ∘ Ψ₂` directly, without
passing through an auxiliary closed-form wrapper.  The signed-structure work is represented by the
two `IsTopical` certificates, which are the abstract output of the monotone-block and translation
equivariance criteria.
-/
theorem graphW1_signedStructure_fullSweep_variationSeminorm_nonexpansive
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi₁ Psi₂ : (ι → ℝ) → (ι → ℝ))
    (hPsi₁ : IsTopical Psi₁)
    (hPsi₂ : IsTopical Psi₂) :
    ∀ x y : ι → ℝ,
      variationSeminorm ((Psi₁ ∘ Psi₂) x - (Psi₁ ∘ Psi₂) y) ≤
        variationSeminorm (x - y) := by
  intro x y
  have hne : SeminormNonexpansive variationSeminormAsSeminorm (Psi₁ ∘ Psi₂) :=
    SeminormNonexpansive_comp
      variationSeminormAsSeminorm Psi₂ Psi₁
      (SeminormNonexpansive_variationSeminormAsSeminorm_of_isTopical hPsi₂)
      (SeminormNonexpansive_variationSeminormAsSeminorm_of_isTopical hPsi₁)
  change
    variationSeminormAsSeminorm ((Psi₁ ∘ Psi₂) x - (Psi₁ ∘ Psi₂) y) ≤
      variationSeminormAsSeminorm (x - y)
  exact hne x y

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
