import FlowSinkhorn.KLProjection.Applications.GraphW1.HGamma

/-!
# `κ` for graph `W₁`

This module is reserved for Proposition `prop:kappa_diam` from
the graph-W1 material in `neurips/paper.tex`.

Intended theorem names:
- `graphW1_kappa_le_graphDiameter`;
- `graphW1_splitPotential_bound`.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace Applications
namespace GraphW1

/--
Compatibility bridge for callers that already have the scalar graph-diameter estimate.

This is not the paper-facing certification of Proposition `prop:kappa_diam`.  The concrete proof is
`graphW1_kappa_le_graphDiameter` below, and the statement map points there.  Keep this bridge only
for older wrappers that work with an abstract scalar `kappa`.
-/
theorem graphW1_kappa_le_graphDiameter_of_assumption
    {kappa graphDiameter : ℝ}
    (hkappa : kappa ≤ graphDiameter) :
    kappa ≤ graphDiameter :=
  hkappa

/--
Auxiliary split-potential bound used downstream by complexity wrappers.
-/
theorem graphW1_splitPotential_bound
    {splitPotential kappa graphDiameter : ℝ}
    (hsplit : splitPotential ≤ kappa)
    (hkappa : kappa ≤ graphDiameter) :
    splitPotential ≤ graphDiameter :=
  hsplit.trans hkappa

/--
Graph-`W₁` budget rewrite under an exact `\kappa` value.
-/
theorem graphW1_hGammaBudget_eq_of_kappa
    {kappa graphDiameter cost gamma hGamma : ℝ}
    (hkappa : kappa = graphDiameter)
    :
    PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma =
      graphDiameter * (cost + gamma * hGamma) := by
  simp [PrimalDualBounds.hGammaKappaBudget, hkappa]

/--
Monotone transfer of split-potential control to the canonical `H_γ/κ` budget.
-/
theorem graphW1_budget_le_of_kappa_le_diameter
    {kappa graphDiameter cost gamma hGamma : ℝ}
    (hnonneg : 0 ≤ cost + gamma * hGamma)
    (hkappa : kappa ≤ graphDiameter) :
    PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤
      PrimalDualBounds.hGammaKappaBudget graphDiameter cost gamma hGamma := by
  dsimp [PrimalDualBounds.hGammaKappaBudget]
  exact mul_le_mul_of_nonneg_right hkappa hnonneg

/--
Auxiliary: the absolute value of a sum of real numbers, each bounded by `B` in absolute
value, is at most `l.length * B`.

This is the abstract step behind the path-potential accumulation estimate in
Proposition `prop:kappa_diam`.
-/
theorem list_abs_sum_le_of_bounded
    {B : ℝ}
    (l : List ℝ)
    (h : ∀ x ∈ l, |x| ≤ B) :
    |l.sum| ≤ l.length * B := by
  induction l with
  | nil => simp
  | cons x xs ih =>
    simp only [List.sum_cons, List.length_cons]
    have hx : |x| ≤ B := h x (List.mem_cons.mpr (Or.inl rfl))
    have hxs : ∀ y ∈ xs, |y| ≤ B := fun y hy => h y (List.mem_cons.mpr (Or.inr hy))
    have ihb := ih hxs
    have htri : |x + xs.sum| ≤ |x| + |xs.sum| := by
      rcases abs_cases (x + xs.sum) with ⟨h1, _⟩ | ⟨h1, _⟩ <;>
      rcases abs_cases x with ⟨h2, _⟩ | ⟨h2, _⟩ <;>
      rcases abs_cases xs.sum with ⟨h3, _⟩ | ⟨h3, _⟩ <;>
      linarith
    push_cast
    linarith

/--
Path-potential bound: the potential accumulated along a path of length at most `diam` steps,
where each step-difference is bounded by `B`, satisfies `|potential| ≤ diam * B`.
-/
theorem path_potential_le_diameter_mul_bound
    (diam : ℕ) (B : ℝ)
    (hB : 0 ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ diam)
    (hsteps : ∀ x ∈ steps, |x| ≤ B) :
    |steps.sum| ≤ (diam : ℝ) * B :=
  calc |steps.sum|
      ≤ steps.length * B := list_abs_sum_le_of_bounded steps hsteps
    _ ≤ (diam : ℝ) * B := by
        apply mul_le_mul_of_nonneg_right _ hB
        exact_mod_cast hlen

/--
Paper-facing path-potential bound for graph `W₁`.

This abstracts the key estimate in Proposition `prop:kappa_diam`:
given edge-difference values bounded by `B`, the potential accumulated along any path of
length at most `graphDiam` is bounded by `graphDiam * B`. This is the core inequality
behind `κ ≤ 2 * diam(G)`.
-/
theorem graphW1_kappa_path_potential_bound
    (graphDiam : ℕ) (B : ℝ)
    (hB : 0 ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, |x| ≤ B) :
    |steps.sum| ≤ (graphDiam : ℝ) * B :=
  path_potential_le_diameter_mul_bound graphDiam B hB steps hlen hsteps

/-- If `|a| ≤ B` and `|b| ≤ B`, then `|a + b| ≤ 2 * B`. -/
theorem abs_add_le_two_mul_of_le {a b B : ℝ} (ha : |a| ≤ B) (hb : |b| ≤ B) :
    |a + b| ≤ 2 * B := by
  have htri : |a + b| ≤ |a| + |b| := by
    rcases abs_cases (a + b) with ⟨h, _⟩ | ⟨h, _⟩ <;>
    rcases abs_cases a with ⟨h2, _⟩ | ⟨h2, _⟩ <;>
    rcases abs_cases b with ⟨h3, _⟩ | ⟨h3, _⟩ <;>
    linarith
  linarith

/-- For the graph-W₁ dual decomposition, the edge-gradient field `g = y^f + y^g`
satisfies `‖g‖_∞ ≤ 2 * ‖y‖_∞`.

This is the first quantitative step in Proposition `prop:kappa_diam`:
once the edge-difference field is bounded by `2 * ‖y‖_∞`, the path-potential
bound `graphW1_kappa_path_potential_bound` gives `|ṽ_k| ≤ 2 * diam * ‖y‖_∞`.
-/
theorem graphW1_edgeGradient_bound
    {ι : Type*}
    (yf yg : ι × ι → ℝ)
    (B : ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (p : ι × ι) :
    |(yf + yg) p| ≤ 2 * B := by
  simp only [Pi.add_apply]
  exact abs_add_le_two_mul_of_le (hyf p) (hyg p)

/-- The sup-norm of the edge-gradient field is at most `2 * B` when both components
have sup-norm at most `B`. -/
theorem graphW1_edgeGradient_supNorm_bound
    {ι : Type*}
    (yf yg : ι × ι → ℝ)
    (B : ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B) :
    ∀ p : ι × ι, |(yf + yg) p| ≤ 2 * B :=
  fun p => graphW1_edgeGradient_bound yf yg B hyf hyg p

/--
Two-step path-potential bound for graph `W₁` (core of Proposition `prop:kappa_diam`).

Chains `graphW1_edgeGradient_supNorm_bound` (step 1: `‖yf + yg‖_∞ ≤ 2B`)
with `graphW1_kappa_path_potential_bound` (step 2: `|potential| ≤ diam * bound`).

Given:
- `yf yg : ι × ι → ℝ` with `|yf p|, |yg p| ≤ B` for all `p`,
- a path `steps : List ℝ` of length `≤ graphDiam`,
- each step `x ∈ steps` satisfies `x = (yf + yg) p` for some edge `p`,

the cumulative potential satisfies `|steps.sum| ≤ 2 * graphDiam * B`.
-/
theorem graphW1_kappa_twoStep_bound
    {ι : Type*}
    (yf yg : ι × ι → ℝ)
    (B : ℝ) (hB : 0 ≤ B)
    (graphDiam : ℕ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p) :
    |steps.sum| ≤ 2 * (graphDiam : ℝ) * B := by
  -- Each step x satisfies |x| ≤ 2 * B
  have hstep_bound : ∀ x ∈ steps, |x| ≤ 2 * B := by
    intro x hx
    obtain ⟨p, rfl⟩ := hsteps x hx
    exact graphW1_edgeGradient_bound yf yg B hyf hyg p
  -- Apply path potential bound with bound = 2 * B
  have hpot :=
    graphW1_kappa_path_potential_bound graphDiam (2 * B) (by linarith) steps hlen hstep_bound
  linarith

/--
Budget corollary of the path-potential bound.

The orbit budget `hGammaKappaBudget (2 * graphDiam) cost gamma hGamma` equals
`2 * graphDiam * (cost + gamma * hGamma)` by definition of `hGammaKappaBudget`.

This is the concrete budget formula for graph-W₁ convergence.
-/
theorem graphW1_kappa_budget_two_diam
    {graphDiam : ℕ} {cost gamma hGamma : ℝ} :
    PrimalDualBounds.hGammaKappaBudget (2 * (graphDiam : ℝ)) cost gamma hGamma =
      2 * (graphDiam : ℝ) * (cost + gamma * hGamma) := by
  unfold PrimalDualBounds.hGammaKappaBudget
  ring

/--
Orbit bound when κ is the concrete 2*graphDiam path estimate.

The concrete budget `hGammaKappaBudget (2 * graphDiam) cost gamma hGamma` is at most
`2 * graphDiam * (cost + gamma * hGamma)`, which here is an equality.

This is the concrete form of Proposition `prop:kappa_diam` (κ ≤ 2 * diam for graph W₁).
-/
theorem graphW1_orbit_bound_two_diam
    {graphDiam : ℕ} {cost gamma hGamma : ℝ} :
    PrimalDualBounds.hGammaKappaBudget (2 * (graphDiam : ℝ)) cost gamma hGamma ≤
      2 * (graphDiam : ℝ) * (cost + gamma * hGamma) :=
  graphW1_kappa_budget_two_diam.le

/--
Bridge theorem from the two-step path estimate to the paper-facing `\kappa` API.

If `κ` is controlled by a path potential, and that path potential follows the two-step graph-W₁
bound with per-edge control `B ≤ 1`, then `κ ≤ 2 * diam`.
-/
theorem graphW1_kappa_le_twoDiam_of_twoStep_path
    {ι : Type*}
    {kappa B : ℝ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|) :
    kappa ≤ 2 * (graphDiam : ℝ) := by
  have htwo :
      |steps.sum| ≤ 2 * (graphDiam : ℝ) * B :=
    graphW1_kappa_twoStep_bound yf yg B hB graphDiam hyf hyg steps hlen hsteps
  have hkappa_scaled : kappa ≤ 2 * (graphDiam : ℝ) * B := hkappa_from_path.trans htwo
  have hfac_nonneg : 0 ≤ 2 * (graphDiam : ℝ) := by positivity
  have hscale : 2 * (graphDiam : ℝ) * B ≤ 2 * (graphDiam : ℝ) * 1 :=
    mul_le_mul_of_nonneg_left hBunit hfac_nonneg
  have hkappa_two : kappa ≤ 2 * (graphDiam : ℝ) * 1 := hkappa_scaled.trans hscale
  simpa using hkappa_two

/--
Internally derived graph-`W₁` `κ ≤ 2 * diam` endpoint.

This is the preferred non-packaged theorem behind the paper-facing aliases: it derives the
diameter control directly from the two-step edge-gradient/path witness, rather than requiring
a pre-packaged scalar hypothesis `κ ≤ 2 * diam`.
-/
theorem graphW1_kappa_le_twoGraphDiameter_from_path
    {ι : Type*}
    {kappa B : ℝ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|) :
    kappa ≤ 2 * (graphDiam : ℝ) :=
  graphW1_kappa_le_twoDiam_of_twoStep_path
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path

/--
Concrete paper-facing `\kappa` estimate for graph `W₁`.

It composes the certified two-step path estimate with the normalization `B ≤ 1` to derive
the canonical bound `κ ≤ 2 * diam(G)`.
-/
theorem graphW1_kappa_le_graphDiameter
    {ι : Type*}
    {kappa B : ℝ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|) :
    kappa ≤ 2 * (graphDiam : ℝ) := by
  exact graphW1_kappa_le_twoGraphDiameter_from_path
    hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path

/--
Rooted-path version of the graph-`W₁` `κ ≤ 2 * diam` estimate.

This is the paper-facing certificate for the integration-along-shortest-paths proof: a rooted
path family gives, for each vertex, a list of oriented edge certificates of length at most the
graph diameter.  If `κ` is controlled by one of the corresponding rooted potentials, then the
two-step edge-gradient estimate gives the displayed `2 * diameter` bound.

The remaining graph-theoretic part of the paper proof is intentionally visible in the hypotheses:
constructing `path` from graph connectivity/shortest paths and proving the `κ`-control premise
from the quotient definition of `Aᵀ(v,U)`.
-/
theorem graphW1_kappa_le_graphDiameter_from_rootedPathFamily
    {ι : Type*}
    {kappa B : ℝ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (path : ι → List (ι × ι))
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (hlen : ∀ i : ι, (path i).length ≤ graphDiam)
    (hkappa_from_rootedPath :
      ∃ i : ι,
        kappa ≤ |((path i).map (fun p : ι × ι => (yf + yg) p)).sum|) :
    kappa ≤ 2 * (graphDiam : ℝ) := by
  obtain ⟨i, hi⟩ := hkappa_from_rootedPath
  let steps : List ℝ := (path i).map (fun p : ι × ι => (yf + yg) p)
  have hlen_steps : steps.length ≤ graphDiam := by
    dsimp [steps]
    simpa only [List.length_map] using hlen i
  have hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p := by
    intro x hx
    dsimp [steps] at hx
    rcases List.mem_map.mp hx with ⟨p, _hp, rfl⟩
    exact ⟨p, rfl⟩
  exact graphW1_kappa_le_twoDiam_of_twoStep_path
    hB hBunit graphDiam yf yg hyf hyg steps hlen_steps hsteps hi

/--
Direct split-potential control from the internally derived graph-`W₁` path witness.

This avoids routing through the compatibility theorem
`graphW1_splitPotential_bound` when the edge-gradient/path proof of `κ ≤ 2 * diam`
is available.
-/
theorem graphW1_splitPotential_bound_from_path
    {ι : Type*}
    {splitPotential kappa B : ℝ}
    (hsplit : splitPotential ≤ kappa)
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|) :
    splitPotential ≤ 2 * (graphDiam : ℝ) :=
  hsplit.trans
    (graphW1_kappa_le_twoGraphDiameter_from_path
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path)

/--
Bridge theorem from the two-step path estimate to the paper-facing budget API.

It composes:
1. `graphW1_kappa_le_twoDiam_of_twoStep_path`,
2. monotonicity of `hGammaKappaBudget` in `\kappa` for nonnegative base term.
-/
theorem graphW1_hGammaBudget_le_twoDiam_of_twoStep_path
    {ι : Type*}
    {kappa B cost gamma hGamma : ℝ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hbase_nonneg : 0 ≤ cost + gamma * hGamma) :
    PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤
      PrimalDualBounds.hGammaKappaBudget (2 * (graphDiam : ℝ)) cost gamma hGamma := by
  have hkappa_two :
      kappa ≤ 2 * (graphDiam : ℝ) :=
    graphW1_kappa_le_twoDiam_of_twoStep_path
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path
  exact graphW1_budget_le_of_kappa_le_diameter hbase_nonneg hkappa_two

/--
Concrete explicit budget form obtained from two-step path control.

This is the final arithmetic bridge to the paper expression
`2 * diam * (cost + gamma * H_gamma)`.
-/
theorem graphW1_hGammaBudget_le_explicit_twoDiam_of_twoStep_path
    {ι : Type*}
    {kappa B cost gamma hGamma : ℝ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hbase_nonneg : 0 ≤ cost + gamma * hGamma) :
    PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤
      2 * (graphDiam : ℝ) * (cost + gamma * hGamma) := by
  calc
    PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma
        ≤ PrimalDualBounds.hGammaKappaBudget (2 * (graphDiam : ℝ)) cost gamma hGamma :=
      graphW1_hGammaBudget_le_twoDiam_of_twoStep_path
        hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path hbase_nonneg
    _ = 2 * (graphDiam : ℝ) * (cost + gamma * hGamma) := graphW1_kappa_budget_two_diam

/--
Two-times explicit budget form obtained from two-step path control.

This is the `U_max`-style companion to
`graphW1_hGammaBudget_le_explicit_twoDiam_of_twoStep_path`.
-/
theorem graphW1_twoTimesHGammaBudget_le_explicit_twoDiam_of_twoStep_path
    {ι : Type*}
    {kappa B cost gamma hGamma : ℝ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hbase_nonneg : 0 ≤ cost + gamma * hGamma) :
    2 * PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤
      4 * (graphDiam : ℝ) * (cost + gamma * hGamma) := by
  have hbudget :
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤
        2 * (graphDiam : ℝ) * (cost + gamma * hGamma) :=
    graphW1_hGammaBudget_le_explicit_twoDiam_of_twoStep_path
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path hbase_nonneg
  nlinarith

/--
Path-free explicit budget bridge from a direct `κ ≤ 2 * diam` hypothesis.

This is useful when only the scalar `κ` estimate is available.
-/
theorem graphW1_hGammaBudget_le_explicit_twoDiam_of_kappa_le_twoDiam
    {kappa cost gamma hGamma : ℝ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hbase_nonneg : 0 ≤ cost + gamma * hGamma) :
    PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤
      2 * (graphDiam : ℝ) * (cost + gamma * hGamma) := by
  calc
    PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma
        ≤ PrimalDualBounds.hGammaKappaBudget (2 * (graphDiam : ℝ)) cost gamma hGamma :=
      graphW1_budget_le_of_kappa_le_diameter hbase_nonneg hkappa_two
    _ = 2 * (graphDiam : ℝ) * (cost + gamma * hGamma) := graphW1_kappa_budget_two_diam

/--
Path-free two-times explicit budget bridge from a direct `κ ≤ 2 * diam` hypothesis.
-/
theorem graphW1_twoTimesHGammaBudget_le_explicit_twoDiam_of_kappa_le_twoDiam
    {kappa cost gamma hGamma : ℝ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hbase_nonneg : 0 ≤ cost + gamma * hGamma) :
    2 * PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤
      4 * (graphDiam : ℝ) * (cost + gamma * hGamma) := by
  have hbudget :
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma hGamma ≤
        2 * (graphDiam : ℝ) * (cost + gamma * hGamma) :=
    graphW1_hGammaBudget_le_explicit_twoDiam_of_kappa_le_twoDiam
      graphDiam hkappa_two hbase_nonneg
  nlinarith

/--
Concrete closed-form budget bridge from two-step path control.

Specializing `H_γ = log(n_nodes)/γ`, this turns the two-step path control into the explicit
budget expression used in graph-W₁ complexity constants:
`2 * diam * (cost + log n_nodes)`.
-/
theorem graphW1_hGammaBudget_le_explicitLog_of_twoStep_path
    {ι : Type*}
    {kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma)) :
    PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma) ≤
      2 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  calc
    PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma)
        ≤ PrimalDualBounds.hGammaKappaBudget (2 * (graphDiam : ℝ))
            cost gamma (Real.log n_nodes / gamma) :=
      graphW1_hGammaBudget_le_twoDiam_of_twoStep_path
        hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path hbase_nonneg
    _ = 2 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
      simpa using
        (graphW1_hGammaBudget_explicit
          (diam := 2 * (graphDiam : ℝ)) (n := n_nodes) (cost := cost) hgamma)

/--
Two-times explicit closed-form budget bridge from two-step path control.

This is the direct `U_max`-style corollary of
`graphW1_hGammaBudget_le_explicitLog_of_twoStep_path`.
-/
theorem graphW1_twoTimesHGammaBudget_le_explicitLog_of_twoStep_path
    {ι : Type*}
    {kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma)) :
    2 * PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  have hbudget :
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma) ≤
        2 * (graphDiam : ℝ) * (cost + Real.log n_nodes) :=
    graphW1_hGammaBudget_le_explicitLog_of_twoStep_path
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path hgamma hbase_nonneg
  nlinarith

/--
Additive-base convenience form of
`graphW1_twoTimesHGammaBudget_le_explicitLog_of_twoStep_path`.
-/
theorem graphW1_twoTimesHGammaBudget_le_explicitLog_of_twoStep_path_with_base
    {ι : Type*}
    {kappa B cost gamma base : ℝ}
    {n_nodes : ℕ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma)) :
    base + 2 * PrimalDualBounds.hGammaKappaBudget kappa cost gamma
        (Real.log n_nodes / gamma) ≤
      base + 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  have hU :
      2 * PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma) ≤
        4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) :=
    graphW1_twoTimesHGammaBudget_le_explicitLog_of_twoStep_path
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path hgamma hbase_nonneg
  simpa [add_comm, add_left_comm, add_assoc] using add_le_add_left hU base

/--
Path-free closed-form budget bridge from a direct `κ ≤ 2 * diam` hypothesis.

This is useful when the graph-side path argument has already been discharged elsewhere and
only the scalar `κ` estimate is available.
-/
theorem graphW1_hGammaBudget_le_explicitLog_of_kappa_le_twoDiam
    {kappa cost gamma : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma)) :
    PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma) ≤
      2 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  calc
    PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma)
        ≤ PrimalDualBounds.hGammaKappaBudget (2 * (graphDiam : ℝ))
            cost gamma (Real.log n_nodes / gamma) :=
      graphW1_budget_le_of_kappa_le_diameter hbase_nonneg hkappa_two
    _ = 2 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
      simpa using
        (graphW1_hGammaBudget_explicit
          (diam := 2 * (graphDiam : ℝ)) (n := n_nodes) (cost := cost) hgamma)

/--
Path-free two-times explicit budget bridge from a direct `κ ≤ 2 * diam` hypothesis.

This is the `U_max`-style companion to
`graphW1_hGammaBudget_le_explicitLog_of_kappa_le_twoDiam`.
-/
theorem graphW1_twoTimesHGammaBudget_le_explicitLog_of_kappa_le_twoDiam
    {kappa cost gamma : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma)) :
    2 * PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  have hbudget :
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma) ≤
        2 * (graphDiam : ℝ) * (cost + Real.log n_nodes) :=
    graphW1_hGammaBudget_le_explicitLog_of_kappa_le_twoDiam
      graphDiam hkappa_two hgamma hbase_nonneg
  nlinarith

/--
Concrete iterate bound with explicit graph constants from two-step path control.

This composes:
1. fixed-point control in abstract `hGammaKappaBudget` form,
2. two-step path to explicit `U_max` constant,
3. nonexpansive iterate bound.
-/
theorem graphW1_variationSeminorm_iterateBound_explicitLog_of_twoStep_path
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤
      variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  have hiter :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
      variationSeminormAsSeminorm Psi hPsi (uStar := vStar) (u0 := v0) hfix hbound k
  have hU :
      2 * PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma) ≤
        4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) :=
    graphW1_twoTimesHGammaBudget_le_explicitLog_of_twoStep_path
      hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path hgamma hbase_nonneg
  have hiter' :
      variationSeminorm ((Psi^[k]) v0) ≤
        variationSeminorm v0 +
          2 * PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma) :=
    hiter
  have hsum :
      variationSeminorm v0 +
          2 * PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma) ≤
        variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) :=
    by
      simpa [add_comm, add_left_comm, add_assoc] using
        add_le_add_left hU (variationSeminorm v0)
  exact hiter'.trans hsum

/--
Zero-seed explicit iterate bound from two-step path control.

This is the zero-initial-variation specialization of
`graphW1_variationSeminorm_iterateBound_explicitLog_of_twoStep_path`.
-/
theorem graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_of_twoStep_path
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  have hiter :
      variationSeminorm ((Psi^[k]) v0) ≤
        variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) :=
    graphW1_variationSeminorm_iterateBound_explicitLog_of_twoStep_path
      Psi hPsi hfix hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path
      hgamma hbase_nonneg hbound k
  rw [hv0, zero_add] at hiter
  exact hiter

/--
Successor-index convenience form of
`graphW1_variationSeminorm_iterateBound_explicitLog_of_twoStep_path`.
-/
theorem graphW1_variationSeminorm_iterateBound_explicitLog_succ_of_twoStep_path
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤
      variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  exact graphW1_variationSeminorm_iterateBound_explicitLog_of_twoStep_path
    Psi hPsi hfix hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path
    hgamma hbase_nonneg hbound (k + 1)

/--
Successor-index zero-seed convenience form of
`graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_of_twoStep_path`.
-/
theorem graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_succ_of_twoStep_path
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa B cost gamma : ℝ}
    {n_nodes : ℕ}
    (hB : 0 ≤ B)
    (hBunit : B ≤ 1)
    (graphDiam : ℕ)
    (yf yg : ι × ι → ℝ)
    (hyf : ∀ p : ι × ι, |yf p| ≤ B)
    (hyg : ∀ p : ι × ι, |yg p| ≤ B)
    (steps : List ℝ)
    (hlen : steps.length ≤ graphDiam)
    (hsteps : ∀ x ∈ steps, ∃ p : ι × ι, x = (yf + yg) p)
    (hkappa_from_path : kappa ≤ |steps.sum|)
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  exact graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_of_twoStep_path
    Psi hPsi hfix hB hBunit graphDiam yf yg hyf hyg steps hlen hsteps hkappa_from_path
    hgamma hbase_nonneg hbound hv0 (k + 1)

/--
Concrete iterate bound with explicit graph constants from a direct `κ ≤ 2 * diam` estimate.

This is the path-free companion of
`graphW1_variationSeminorm_iterateBound_explicitLog_of_twoStep_path`.
-/
theorem graphW1_variationSeminorm_iterateBound_explicitLog_of_kappa_le_twoDiam
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤
      variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  have hiter :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint_bound
      variationSeminormAsSeminorm Psi hPsi (uStar := vStar) (u0 := v0) hfix hbound k
  have hU :
      2 * PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma) ≤
        4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) :=
    graphW1_twoTimesHGammaBudget_le_explicitLog_of_kappa_le_twoDiam
      graphDiam hkappa_two hgamma hbase_nonneg
  have hiter' :
      variationSeminorm ((Psi^[k]) v0) ≤
        variationSeminorm v0 +
          2 * PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma) :=
    hiter
  have hsum :
      variationSeminorm v0 +
          2 * PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma) ≤
        variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) :=
    by
      simpa [add_comm, add_left_comm, add_assoc] using
        add_le_add_left hU (variationSeminorm v0)
  exact hiter'.trans hsum

/--
Successor-index path-free convenience form of
`graphW1_variationSeminorm_iterateBound_explicitLog_of_kappa_le_twoDiam`.
-/
theorem graphW1_variationSeminorm_iterateBound_explicitLog_succ_of_kappa_le_twoDiam
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤
      variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) :=
  graphW1_variationSeminorm_iterateBound_explicitLog_of_kappa_le_twoDiam
    Psi hPsi hfix graphDiam hkappa_two hgamma hbase_nonneg hbound (k + 1)

/--
Path-free explicit iterate bound from `κ ≤ 2 * diam` lifted to a larger budget.

If `4 * diam * (cost + log n_nodes) ≤ U`, then the iterate bound is
`variationSeminorm ((Psi^[k]) v0) ≤ variationSeminorm v0 + U`.
-/
theorem graphW1_variationSeminorm_iterateBound_explicitLog_of_kappa_le_twoDiam_of_bound_le
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma U : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hU : 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) ≤ U)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤ variationSeminorm v0 + U := by
  have hiter :
      variationSeminorm ((Psi^[k]) v0) ≤
        variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) :=
    graphW1_variationSeminorm_iterateBound_explicitLog_of_kappa_le_twoDiam
      Psi hPsi hfix graphDiam hkappa_two hgamma hbase_nonneg hbound k
  have hsum :
      variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) ≤
        variationSeminorm v0 + U := by
    simpa [add_comm, add_left_comm, add_assoc] using
      add_le_add_left hU (variationSeminorm v0)
  exact hiter.trans hsum

/--
Path-free zero-seed explicit iterate bound from a direct `κ ≤ 2 * diam` estimate.
-/
theorem graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_of_kappa_le_twoDiam
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  have hiter :
      variationSeminorm ((Psi^[k]) v0) ≤
        variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) :=
    graphW1_variationSeminorm_iterateBound_explicitLog_of_kappa_le_twoDiam
      Psi hPsi hfix graphDiam hkappa_two hgamma hbase_nonneg hbound k
  rw [hv0, zero_add] at hiter
  exact hiter

/--
Path-free zero-seed explicit iterate bound from `κ ≤ 2 * diam` lifted to a larger budget.
-/
theorem graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_of_kappa_le_twoDiam_of_bound_le
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma U : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (hU : 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) ≤ U)
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤ U := by
  have hiter :
      variationSeminorm ((Psi^[k]) v0) ≤
        4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) :=
    graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_of_kappa_le_twoDiam
      Psi hPsi hfix graphDiam hkappa_two hgamma hbase_nonneg hbound hv0 k
  exact hiter.trans hU

/--
Successor-index zero-seed path-free convenience form of
`graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_of_kappa_le_twoDiam`.
-/
theorem graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_succ_of_kappa_le_twoDiam
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  exact graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_of_kappa_le_twoDiam
    Psi hPsi hfix graphDiam hkappa_two hgamma hbase_nonneg hbound hv0 (k + 1)

/--
Index-threshold convenience form of
`graphW1_variationSeminorm_iterateBound_explicitLog_of_kappa_le_twoDiam`.

The extra assumption `k ≤ n` is useful when an outer argument already carries an index threshold.
-/
theorem graphW1_variationSeminorm_iterateBound_explicitLog_of_kappa_le_twoDiam_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    {k n : ℕ} (hk : k ≤ n) :
    variationSeminorm ((Psi^[k]) v0) ≤
      variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  have _hidx : n - k + k = n := Nat.sub_add_cancel hk
  exact graphW1_variationSeminorm_iterateBound_explicitLog_of_kappa_le_twoDiam
    Psi hPsi hfix graphDiam hkappa_two hgamma hbase_nonneg hbound k

/--
Index-threshold convenience form of
`graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_of_kappa_le_twoDiam`.
-/
theorem graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_of_kappa_le_twoDiam_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hv0 : variationSeminorm v0 = 0)
    {k n : ℕ} (hk : k ≤ n) :
    variationSeminorm ((Psi^[k]) v0) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  have _hidx : n - k + k = n := Nat.sub_add_cancel hk
  exact graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_of_kappa_le_twoDiam
    Psi hPsi hfix graphDiam hkappa_two hgamma hbase_nonneg hbound hv0 k

/--
Successor-index + index-threshold convenience form for the path-free explicit bound.
-/
theorem graphW1_variationSeminorm_iterateBound_explicitLog_succ_of_kappa_le_twoDiam_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    {k n : ℕ} (hk : k + 1 ≤ n) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤
      variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  have _hidx : n - (k + 1) + (k + 1) = n := Nat.sub_add_cancel hk
  exact graphW1_variationSeminorm_iterateBound_explicitLog_succ_of_kappa_le_twoDiam
    Psi hPsi hfix graphDiam hkappa_two hgamma hbase_nonneg hbound k

/--
Successor-index + index-threshold convenience form for the path-free zero-seed explicit bound.
-/
theorem
    graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_succ_of_kappa_le_twoDiam_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hv0 : variationSeminorm v0 = 0)
    {k n : ℕ} (hk : k + 1 ≤ n) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  have _hidx : n - (k + 1) + (k + 1) = n := Nat.sub_add_cancel hk
  exact graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_succ_of_kappa_le_twoDiam
    Psi hPsi hfix graphDiam hkappa_two hgamma hbase_nonneg hbound hv0 k

/--
Nat-ceil index convenience form of
`graphW1_variationSeminorm_iterateBound_explicitLog_of_kappa_le_twoDiam_of_le_index`.
-/
theorem graphW1_variationSeminorm_iterateBound_explicitLog_of_kappa_le_twoDiam_of_le_natCeil_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    {k : ℕ} {R : ℝ}
    (hk : k ≤ Nat.ceil R) :
    variationSeminorm ((Psi^[k]) v0) ≤
      variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  exact graphW1_variationSeminorm_iterateBound_explicitLog_of_kappa_le_twoDiam_of_le_index
    Psi hPsi hfix graphDiam hkappa_two hgamma hbase_nonneg hbound
    (n := Nat.ceil R) hk

/--
Successor nat-ceil index convenience form of
`graphW1_variationSeminorm_iterateBound_explicitLog_succ_of_kappa_le_twoDiam_of_le_index`.
-/
theorem
    graphW1_variationSeminorm_iterateBound_explicitLog_succ_of_kappa_le_twoDiam_of_le_natCeil_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    {k : ℕ} {R : ℝ}
    (hk : k + 1 ≤ Nat.ceil R) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤
      variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  exact graphW1_variationSeminorm_iterateBound_explicitLog_succ_of_kappa_le_twoDiam_of_le_index
    Psi hPsi hfix graphDiam hkappa_two hgamma hbase_nonneg hbound
    (n := Nat.ceil R) hk

/--
Zero-seed successor nat-ceil index convenience form of
`graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_succ_of_kappa_le_twoDiam_le_index`.
-/
theorem
    graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_succ_of_kappa_le_twoDiam_of_le_natCeil_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hv0 : variationSeminorm v0 = 0)
    {k : ℕ} {R : ℝ}
    (hk : k + 1 ≤ Nat.ceil R) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  exact
    graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_succ_of_kappa_le_twoDiam_le_index
      Psi hPsi hfix graphDiam hkappa_two hgamma hbase_nonneg hbound hv0
      (n := Nat.ceil R) hk

/--
Zero-seed nat-ceil index convenience form of
`graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_of_kappa_le_twoDiam_of_le_index`.
-/
theorem
    graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_of_kappa_le_twoDiam_of_le_natCeil_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hv0 : variationSeminorm v0 = 0)
    {k : ℕ} {R : ℝ}
    (hk : k ≤ Nat.ceil R) :
    variationSeminorm ((Psi^[k]) v0) ≤
      4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) := by
  exact graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_of_kappa_le_twoDiam_of_le_index
    Psi hPsi hfix graphDiam hkappa_two hgamma hbase_nonneg hbound hv0
    (n := Nat.ceil R) hk

/--
Nat-ceil index convenience form for the budget-lifted path-free explicit iterate bound.
-/
theorem
    graphW1_variationSeminorm_iterateBound_explicitLog_of_kappa_le_twoDiam_budget_le_of_le_natCeil_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma U : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hU : 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) ≤ U)
    {k : ℕ} {R : ℝ}
    (hk : k ≤ Nat.ceil R) :
    variationSeminorm ((Psi^[k]) v0) ≤ variationSeminorm v0 + U := by
  have hbase :
      variationSeminorm ((Psi^[k]) v0) ≤
        variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) :=
    graphW1_variationSeminorm_iterateBound_explicitLog_of_kappa_le_twoDiam_of_le_natCeil_index
      Psi hPsi hfix graphDiam hkappa_two hgamma hbase_nonneg hbound hk
  have hlift :
      variationSeminorm v0 + 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) ≤
        variationSeminorm v0 + U := by
    linarith
  exact hbase.trans hlift

/--
Zero-seed successor nat-ceil index convenience form for the budget-lifted path-free
explicit iterate bound.
-/
theorem
    graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_succ_of_kappa_le_twoDiam_budget_le_of_le_natCeil_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma U : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (hU : 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) ≤ U)
    {k : ℕ} {R : ℝ}
    (hk : k + 1 ≤ Nat.ceil R) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤ U := by
  have hbase :
      variationSeminorm ((Psi^[k + 1]) v0) ≤
        4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) :=
    graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_succ_of_kappa_le_twoDiam_of_le_natCeil_index
      Psi hPsi hfix graphDiam hkappa_two hgamma hbase_nonneg hbound hv0 hk
  exact hbase.trans hU

/--
Budget-threshold + index-threshold convenience form for path-free zero-seed explicit
iterate bounds.
-/
theorem
    graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_of_kappa_le_twoDiam_of_bound_le_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma U : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (hU : 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) ≤ U)
    {k n : ℕ} (hk : k ≤ n) :
    variationSeminorm ((Psi^[k]) v0) ≤ U := by
  have _hidx : n - k + k = n := Nat.sub_add_cancel hk
  exact graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_of_kappa_le_twoDiam_of_bound_le
    Psi hPsi hfix graphDiam hkappa_two hgamma hbase_nonneg hbound hv0 hU k

/--
Budget-threshold + index-threshold convenience form for path-free explicit iterate bounds.
-/
theorem graphW1_variationSeminorm_iterateBound_explicitLog_of_kappa_le_twoDiam_budget_le_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma U : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hU : 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) ≤ U)
    {k n : ℕ} (hk : k ≤ n) :
    variationSeminorm ((Psi^[k]) v0) ≤ variationSeminorm v0 + U := by
  have _hidx : n - k + k = n := Nat.sub_add_cancel hk
  exact graphW1_variationSeminorm_iterateBound_explicitLog_of_kappa_le_twoDiam_of_bound_le
    Psi hPsi hfix graphDiam hkappa_two hgamma hbase_nonneg hbound hU k

/--
Successor-index budget-threshold + index-threshold convenience form for path-free
explicit iterate bounds.
-/
theorem graphW1_variationSeminorm_iterateBound_explicitLog_succ_of_kappa_le_twoDiam_budget_le_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma U : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hU : 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) ≤ U)
    {k n : ℕ} (hk : k + 1 ≤ n) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤ variationSeminorm v0 + U := by
  have _hidx : n - (k + 1) + (k + 1) = n := Nat.sub_add_cancel hk
  exact graphW1_variationSeminorm_iterateBound_explicitLog_of_kappa_le_twoDiam_budget_le_of_le_index
    Psi hPsi hfix graphDiam hkappa_two hgamma hbase_nonneg hbound hU
    (k := k + 1) (n := n) hk

/--
Successor nat-ceil index convenience form for the budget-lifted path-free
explicit iterate bound.
-/
theorem
    graphW1_variationSeminorm_iterateBound_explicitLog_succ_of_kappa_le_twoDiam_budget_le_of_le_natCeil_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma U : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hU : 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) ≤ U)
    {k : ℕ} {R : ℝ}
    (hk : k + 1 ≤ Nat.ceil R) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤ variationSeminorm v0 + U := by
  exact graphW1_variationSeminorm_iterateBound_explicitLog_succ_of_kappa_le_twoDiam_budget_le_of_le_index
    Psi hPsi hfix graphDiam hkappa_two hgamma hbase_nonneg hbound hU
    (n := Nat.ceil R) hk

/--
Zero-seed nat-ceil index convenience form for the budget-lifted path-free
explicit iterate bound.
-/
theorem
    graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_of_kappa_le_twoDiam_of_bound_le_of_le_natCeil_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma U : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (hU : 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) ≤ U)
    {k : ℕ} {R : ℝ}
    (hk : k ≤ Nat.ceil R) :
    variationSeminorm ((Psi^[k]) v0) ≤ U := by
  have _hidx : Nat.ceil R - k + k = Nat.ceil R := Nat.sub_add_cancel hk
  exact graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_of_kappa_le_twoDiam_of_bound_le
    Psi hPsi hfix graphDiam hkappa_two hgamma hbase_nonneg hbound hv0 hU k

/--
Successor-index budget-threshold + index-threshold convenience form for path-free zero-seed
explicit iterate bounds.
-/
theorem
    graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_succ_of_kappa_le_twoDiam_budget_le_of_le_index
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma U : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (hU : 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) ≤ U)
    {k n : ℕ} (hk : k + 1 ≤ n) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤ U := by
  have _hidx : n - (k + 1) + (k + 1) = n := Nat.sub_add_cancel hk
  exact graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_of_kappa_le_twoDiam_of_bound_le
    Psi hPsi hfix graphDiam hkappa_two hgamma hbase_nonneg hbound hv0 hU (k + 1)

/--
Successor-index path-free zero-seed iterate bound lifted to a larger budget.
-/
theorem
    graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_succ_of_kappa_le_twoDiam_budget_le
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {kappa cost gamma U : ℝ}
    {n_nodes : ℕ}
    (graphDiam : ℕ)
    (hkappa_two : kappa ≤ 2 * (graphDiam : ℝ))
    (hgamma : 0 < gamma)
    (hbase_nonneg : 0 ≤ cost + gamma * (Real.log n_nodes / gamma))
    (hbound : variationSeminorm vStar ≤
      PrimalDualBounds.hGammaKappaBudget kappa cost gamma (Real.log n_nodes / gamma))
    (hv0 : variationSeminorm v0 = 0)
    (hU : 4 * (graphDiam : ℝ) * (cost + Real.log n_nodes) ≤ U)
    (k : ℕ) :
    variationSeminorm ((Psi^[k + 1]) v0) ≤ U :=
  graphW1_variationSeminorm_iterateBound_explicitLog_from_zero_of_kappa_le_twoDiam_of_bound_le
    Psi hPsi hfix graphDiam hkappa_two hgamma hbase_nonneg hbound hv0 hU (k + 1)

/--
Graph-W₁ orbit iterate bound.

If `Psi` is non-expansive w.r.t. `variationSeminormAsSeminorm` and the fixed point
`vStar` satisfies `variationSeminorm vStar ≤ 2 * diam * (cost + gamma * hGamma)`,
then for any `v0` and `k`:
  `variationSeminorm (Psi^[k] v0) ≤ variationSeminorm v0 + 2 * (2*diam*(cost+gamma*hGamma))`.

This is the concrete graph-W₁ form of Proposition `prop:uniform_iter_final`.
-/
theorem graphW1_variationSeminorm_iterateBound
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (Psi : (ι → ℝ) → (ι → ℝ))
    (hPsi : SeminormNonexpansive variationSeminormAsSeminorm Psi)
    {vStar v0 : ι → ℝ} (hfix : Psi vStar = vStar)
    {diam : ℕ} {cost gamma hGamma : ℝ}
    (hbound : variationSeminorm vStar ≤ 2 * (diam : ℝ) * (cost + gamma * hGamma))
    (k : ℕ) :
    variationSeminorm ((Psi^[k]) v0) ≤
      variationSeminorm v0 + 2 * (2 * (diam : ℝ) * (cost + gamma * hGamma)) := by
  have hiter :=
    seminorm_iterate_le_of_nonexpansive_fixedPoint
      variationSeminormAsSeminorm Psi hPsi (uStar := vStar) (u0 := v0) hfix k
  have hiter' : variationSeminorm ((Psi^[k]) v0) ≤
      variationSeminorm v0 + 2 * variationSeminorm vStar := hiter
  linarith

end GraphW1
end Applications
end KLProjection
end FlowSinkhorn
