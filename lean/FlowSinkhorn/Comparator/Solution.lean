import FlowSinkhorn.KLProjection.StatementMap

set_option linter.style.longLine false
set_option linter.unusedVariables false

/-!
# Comparator solution module

This file exposes the paper-facing theorem names required by
`leanprover/comparator` as actual theorem constants.

Important: this is only the untrusted solution side. The trusted challenge side
must be authored independently from the paper/blueprint statements; generating
it from this file would make the Comparator comparison circular.
-/

theorem prop_dual_gamma_correct : ∀ {m : Type u_1} {d : Type u_2} [inst : Fintype d] (A : (d → ℝ) →ₗ[ℝ] m → ℝ) (z C : d → ℝ) (gamma Z : ℝ) (b : m → ℝ) (pairing : (m → ℝ) → (m → ℝ) → ℝ) (scoreMap : (m → ℝ) → d → ℝ) (gradF : (m → ℝ) → m → ℝ) (feasible : (d → ℝ) → Prop) (primalObjective : (d → ℝ) → ℝ) (uStar : m → ℝ),
  FlowSinkhorn.KLProjection.Section2Duality.DualGammaPrimalDualCertificate A z C gamma Z b pairing scoreMap gradF feasible primalObjective uStar →
    have xStar := FlowSinkhorn.KLProjection.Section2Duality.primalFromDualScore z C gamma (scoreMap uStar);
    Z = ∑ i : d, z i ∧ (∀ (i : d), 0 < z i) ∧ 0 < gamma ∧ primalObjective xStar = FlowSinkhorn.KLProjection.Section2Duality.dualObjective_from_zC z C gamma Z b pairing scoreMap uStar ∧ FlowSinkhorn.KLProjection.Section2Duality.IsUniquePrimalMinimizer feasible primalObjective xStar ∧ FlowSinkhorn.KLProjection.Section2Duality.IsDualMaximizer (FlowSinkhorn.KLProjection.Section2Duality.dualObjective_from_zC z C gamma Z b pairing scoreMap) uStar ∧ (xStar = fun (i : d) => z i * Real.exp ((scoreMap uStar i - C i) / gamma)) ∧ (gradF uStar = 0 ↔ A xStar = b) ∧ FlowSinkhorn.KLProjection.Section2Duality.dualObjective_from_zC z C gamma Z b pairing scoreMap uStar = FlowSinkhorn.KLProjection.Section2Duality.dualObjective_from_kernel (FlowSinkhorn.KLProjection.Section2Duality.tiltedKernel z C gamma) gamma Z b pairing scoreMap uStar := by
  exact FlowSinkhorn.KLProjection.StatementMap.prop_dual_gamma_correct

theorem thm_kl_dual_rate : ∀ {gap residual : ℕ → ℝ} {gamma Xmax Umax Anorm : ℝ}, 0 < gamma → 0 < Xmax → 0 < Umax → 0 < Anorm → (∀ (k : ℕ), 0 ≤ gap k) → (∀ (k : ℕ), gap k ≤ 2 * Umax * residual k) → (∀ (k : ℕ), gamma / (2 * Xmax * Anorm ^ 2) * residual k ^ 2 ≤ gap k - gap (k + 1)) → ∀ (n : ℕ), 0 ≤ gap (n + 1) ∧ gap (n + 1) ≤ 8 * Xmax * Umax ^ 2 * Anorm ^ 2 / gamma / (↑n + 1) := by
  exact FlowSinkhorn.KLProjection.StatementMap.thm_kl_dual_rate

theorem thm_approx_linprog : ∀ {coord : Type u_1} [inst : Fintype coord] [Nonempty coord] (C x0 xgamma : coord → ℝ) (Feasible : (coord → ℝ) → Prop) (klTerm : (coord → ℝ) → coord → ℝ) {gamma Xmax Umax Anorm XmaxZero maxMass eps : ℝ} {Fgamma gap residual : ℕ → ℝ}, FlowSinkhorn.KLProjection.DualConvergence.ApproxLinprogCertificate C x0 xgamma Feasible klTerm gamma Xmax Umax Anorm XmaxZero maxMass eps Fgamma gap residual → ∀ (n : ℕ), ⌈64 * Xmax * Umax ^ 2 * Anorm ^ 2 * maxMass * Real.log ↑(Fintype.card coord) / eps ^ 2⌉₊ ≤ n + 1 → |FlowSinkhorn.KLProjection.DualConvergence.linearObjective C x0 - Fgamma n| ≤ eps := by
  exact FlowSinkhorn.KLProjection.StatementMap.thm_approx_linprog

theorem prop_uniform_iter_final : ∀ {𝕜 : Type u_1} {E : Type u_2} [inst : NormedField 𝕜] [inst_1 : AddCommGroup E] [inst_2 : _root_.Module 𝕜 E] (p : Seminorm 𝕜 E) (Psi : E → E), FlowSinkhorn.KLProjection.SeminormNonexpansive p Psi → ∀ {uStar u0 : E}, Psi uStar = uStar → ∀ {B : ℝ}, p uStar ≤ B → ∀ (k : ℕ), p (Psi^[k] u0) ≤ p u0 + 2 * B := by
  exact FlowSinkhorn.KLProjection.StatementMap.prop_uniform_iter_final

theorem prop_mass_bound_block : ∀ {coord : Type u_1} [inst : Fintype coord] {pot : Type u_2} [inst_1 : Fintype pot] [Nonempty pot] {xMass : ℕ → ℝ} {C : coord → ℝ} {b : pot → ℝ} {u : ℕ → pot → ℝ} {Umax gamma Cmin : ℝ}, 0 < gamma → FlowSinkhorn.KLProjection.PrimalDualBounds.CostLowerBound C Cmin → FlowSinkhorn.KLProjection.PrimalDualBounds.FiniteQuotientRadiusBound b u Umax → FlowSinkhorn.KLProjection.PrimalDualBounds.DisplayedFinitePairingAscent b u xMass gamma → FlowSinkhorn.KLProjection.PrimalDualBounds.ZeroStartPrimalMass C u xMass gamma → ∀ (k : ℕ), xMass k ≤ (∑ j : pot, |b j|) * Umax / gamma + ↑(Fintype.card coord) * Real.exp (-Cmin / gamma) := by
  exact FlowSinkhorn.KLProjection.StatementMap.prop_mass_bound_block

theorem prop_graphw1_projection_closed_form : ∀ {ι : Type u_1} [inst : Fintype ι] (bDiff hRow hCol : ι → ℝ) (h f g : ι → ι → ℝ),
  (∀ (i : ι), 0 < hRow i) →
    (∀ (i : ι), 0 < hCol i) →
      FlowSinkhorn.KLProjection.Applications.GraphW1.GraphW1ProjectionVariationalCertificate bDiff hRow hCol h f g →
        (∀ (i : ι), 0 < FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_C1Scaling bDiff hRow hCol i) ∧
          (∀ (i j : ι), FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_projC1Left bDiff hRow hCol h i j = FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_C1Scaling bDiff hRow hCol i * h i j ∧ FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_projC1Right bDiff hRow hCol h i j = h i j / FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_C1Scaling bDiff hRow hCol j) ∧
            (∀ (i : ι),
                have s := FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_C1Scaling bDiff hRow hCol i;
                s ^ 2 * hRow i + s * bDiff i - hCol i = 0) ∧
              FlowSinkhorn.KLProjection.Applications.GraphW1.GraphW1C1Constraint bDiff (FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_projC1Left bDiff hRow hCol h) (FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_projC1Right bDiff hRow hCol h) ∧ FlowSinkhorn.KLProjection.Applications.GraphW1.GraphW1PairNonnegative (FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_projC1Left bDiff hRow hCol h) (FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_projC1Right bDiff hRow hCol h) ∧ FlowSinkhorn.KLProjection.Applications.GraphW1.GraphW1C1ProjectionOptimality bDiff h (FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_projC1Left bDiff hRow hCol h) (FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_projC1Right bDiff hRow hCol h) ∧ (∀ (i j : ι), FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_projC2Common f g i j = √(f i j * g i j) ∧ FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_projC2Common f g i j ^ 2 = f i j * g i j ∧ 0 ≤ FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_projC2Common f g i j) ∧ FlowSinkhorn.KLProjection.Applications.GraphW1.GraphW1C2ProjectionOptimality f g (FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_projC2Common f g) := by
  exact FlowSinkhorn.KLProjection.StatementMap.prop_graphw1_projection_closed_form

theorem prop_graphw1_flow_sinkhorn_update : ∀ {ι : Type u_1} [inst : Fintype ι] (Ψ₁ Ψ₂ : (ι → ℝ) → ι → ℝ) (v bDiff : ι → ℝ) (w : ι → ι → ℝ) (gamma : ℝ), gamma ≠ 0 → (∀ (v : ι → ℝ), Ψ₂ v = FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_mUpdate bDiff w gamma v) → (∀ (v : ι → ℝ), Ψ₁ (FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_mUpdate bDiff w gamma v) = fun (i : ι) => v i / 2 - gamma * FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_mUpdate bDiff w gamma v i) → (Ψ₁ ∘ Ψ₂) v = fun (i : ι) => 1 / 2 * v i + 1 / 2 * (FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_alphaPlus w gamma v i - FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_alphaMinus w gamma v i) - gamma * Real.arsinh (FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_beta bDiff w gamma v i) := by
  exact FlowSinkhorn.KLProjection.StatementMap.prop_graphw1_flow_sinkhorn_update

theorem thm_graphw1_complexity : ∀ {w1Error : ℕ → ℝ} {eps p graphDiam logFactor iterationBudget perSweepOps operationCount : ℝ} {pOfEps : ℝ → ℝ} (k : ℕ), 0 < eps → 0 ≤ p → w1Error k ≤ eps → ↑k ≤ iterationBudget → iterationBudget ≤ logFactor * graphDiam ^ 3 / eps ^ 4 → perSweepOps ≤ p → operationCount ≤ ↑k * perSweepOps → p = pOfEps eps → FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1LittleOEdgeRegime pOfEps → 0 < eps ∧ w1Error k ≤ eps ∧ operationCount ≤ logFactor * p * graphDiam ^ 3 / eps ^ 4 ∧ p = pOfEps eps ∧ FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1LittleOEdgeRegime pOfEps := by
  exact FlowSinkhorn.KLProjection.StatementMap.thm_graphw1_complexity

theorem lem_per_step_ascent : ∀ {n₁ n₂ : ℕ} {F Fhalf : ℕ → ℝ} {p1 q1 : ℕ → Fin n₁ → ℝ} {p2 q2 : ℕ → Fin n₂ → ℝ} {r1 r2 : ℕ → ℝ} {gamma Xmax Anorm M : ℝ}, 0 ≤ gamma → 0 < Xmax → 0 < Anorm → 0 < M → M ≤ Xmax → (∀ (k : ℕ), FlowSinkhorn.KLProjection.DualConvergence.FiniteMassShellGammaExactSupportBlockUpdateCertificate (p1 k) (q1 k) M gamma (F k) (Fhalf k)) → (∀ (k : ℕ), FlowSinkhorn.KLProjection.DualConvergence.FiniteMassShellGammaExactSupportBlockUpdateCertificate (p2 k) (q2 k) M gamma (Fhalf k) (F (k + 1))) → (∀ (k : ℕ), 0 ≤ r1 k) → (∀ (k : ℕ), r1 k ≤ Anorm * FlowSinkhorn.KLProjection.DualConvergence.l1Norm fun (i : Fin n₁) => p1 k i - q1 k i) → (∀ (k : ℕ), 0 ≤ r2 k) → (∀ (k : ℕ), r2 k ≤ Anorm * FlowSinkhorn.KLProjection.DualConvergence.l1Norm fun (i : Fin n₂) => p2 k i - q2 k i) → ∀ (k : ℕ), gamma / (2 * Xmax) * (r1 k ^ 2 / Anorm ^ 2) ≤ Fhalf k - F k ∧ gamma / (2 * Xmax) * (r2 k ^ 2 / Anorm ^ 2) ≤ F (k + 1) - Fhalf k := by
  exact FlowSinkhorn.KLProjection.StatementMap.lem_per_step_ascent

theorem lem_gap_vs_res_quotient : ∀ {ι₁ : Type u_1} {ι₂ : Type u_2} [inst : Fintype ι₁] [inst_1 : Nonempty ι₁] [inst_2 : Fintype ι₂] [inst_3 : Nonempty ι₂] {gapNow Umax : ℝ} {r₁ uStar₁ uNow₁ : ι₁ → ℝ} {r₂ uStar₂ uNow₂ : ι₂ → ℝ} (τ : FlowSinkhorn.KLProjection.PairedSign), ∑ i : ι₁, r₁ i + τ.toReal * ∑ j : ι₂, r₂ j = 0 → gapNow ≤ ∑ i : ι₁, r₁ i * (uStar₁ i - uNow₁ i) + ∑ j : ι₂, r₂ j * (uStar₂ j - uNow₂ j) → FlowSinkhorn.KLProjection.signedPairedQuotientSupSeminorm τ (uStar₁, uStar₂) < Umax → FlowSinkhorn.KLProjection.signedPairedQuotientSupSeminorm τ (uNow₁, uNow₂) < Umax → gapNow ≤ 2 * Umax * FlowSinkhorn.KLProjection.DualConvergence.finiteL1Pair r₁ r₂ := by
  exact FlowSinkhorn.KLProjection.StatementMap.lem_gap_vs_res_quotient

theorem prop_pinsker_normalized : ∀ {n : ℕ} {p q : Fin n → ℝ}, (∀ (i : Fin n), 0 ≤ p i) → ∑ i : Fin n, p i = 1 → (∀ (i : Fin n), 0 < q i) → ∑ i : Fin n, q i = 1 → FlowSinkhorn.KLProjection.DualConvergence.finiteKL p q ≥ (FlowSinkhorn.KLProjection.DualConvergence.l1Norm fun (i : Fin n) => p i - q i) ^ 2 / 2 := by
  exact FlowSinkhorn.KLProjection.StatementMap.prop_pinsker_normalized

theorem lem_pinsker_nonnormalized : ∀ {n : ℕ} {p q : Fin n → ℝ} {M : ℝ}, 0 < M → (∀ (i : Fin n), 0 ≤ p i) → ∑ i : Fin n, p i = M → (∀ (i : Fin n), 0 < q i) → ∑ i : Fin n, q i = M → FlowSinkhorn.KLProjection.DualConvergence.finiteKL p q ≥ (FlowSinkhorn.KLProjection.DualConvergence.l1Norm fun (i : Fin n) => p i - q i) ^ 2 / (2 * M) := by
  exact FlowSinkhorn.KLProjection.StatementMap.lem_pinsker_nonnormalized

theorem lem_kl_bias : ∀ {coord : Type u_1} [inst : Fintype coord] [Nonempty coord] (C x0 xgamma : coord → ℝ) (Feasible : (coord → ℝ) → Prop) (klTerm : (coord → ℝ) → coord → ℝ) {gamma XmaxZero : ℝ}, 0 ≤ gamma → FlowSinkhorn.KLProjection.DualConvergence.IsLinearMinimizer Feasible C x0 → FlowSinkhorn.KLProjection.DualConvergence.IsRegularizedMinimizer Feasible C gamma (FlowSinkhorn.KLProjection.DualConvergence.coordinateSumKL klTerm) xgamma → FlowSinkhorn.KLProjection.DualConvergence.NonnegativeOn Feasible (FlowSinkhorn.KLProjection.DualConvergence.coordinateSumKL klTerm) → (∀ (i : coord), klTerm x0 i ≤ x0 i * Real.log ↑(Fintype.card coord)) → ∑ i : coord, x0 i ≤ XmaxZero → 0 ≤ FlowSinkhorn.KLProjection.DualConvergence.regularizedObjective C gamma (FlowSinkhorn.KLProjection.DualConvergence.coordinateSumKL klTerm) xgamma - FlowSinkhorn.KLProjection.DualConvergence.linearObjective C x0 ∧ FlowSinkhorn.KLProjection.DualConvergence.regularizedObjective C gamma (FlowSinkhorn.KLProjection.DualConvergence.coordinateSumKL klTerm) xgamma - FlowSinkhorn.KLProjection.DualConvergence.linearObjective C x0 ≤ gamma * FlowSinkhorn.KLProjection.DualConvergence.coordinateSumKL klTerm x0 ∧ FlowSinkhorn.KLProjection.DualConvergence.regularizedObjective C gamma (FlowSinkhorn.KLProjection.DualConvergence.coordinateSumKL klTerm) xgamma - FlowSinkhorn.KLProjection.DualConvergence.linearObjective C x0 ≤ gamma * ∑ i : coord, klTerm x0 i ∧ FlowSinkhorn.KLProjection.DualConvergence.regularizedObjective C gamma (FlowSinkhorn.KLProjection.DualConvergence.coordinateSumKL klTerm) xgamma - FlowSinkhorn.KLProjection.DualConvergence.linearObjective C x0 ≤ gamma * XmaxZero * Real.log ↑(Fintype.card coord) := by
  exact FlowSinkhorn.KLProjection.StatementMap.lem_kl_bias

theorem prop_hgamma_ot : ∀ {ι₁ : Type u_1} {ι₂ : Type u_2} [inst : Fintype ι₁] [inst_1 : Fintype ι₂] [Nonempty ι₁] [Nonempty ι₂] (b : FlowSinkhorn.KLProjection.Applications.OT.ProbabilityVector ι₂) (u : FlowSinkhorn.KLProjection.Applications.OT.NonnegativeField ι₁) (v : FlowSinkhorn.KLProjection.Applications.OT.NonnegativeField ι₂) {C_max : ℝ} (C : FlowSinkhorn.KLProjection.Applications.OT.BoundedCostField ι₁ ι₂ C_max) (logRatio : ι₁ → ι₂ → ℝ) {min_b gamma : ℝ}, 0 < gamma → 0 < min_b → (∀ (j : ι₂), min_b ≤ b.val j) → (∀ (i : ι₁), ∑ j : ι₂, b.val j * Real.exp (logRatio i j) = 1) → (∀ (j : ι₂), v.val j * ∑ i : ι₁, Real.exp (-(C.val i j / gamma)) * u.val i = b.val j) → (∀ (i : ι₁) (j : ι₂), Real.exp (logRatio i j) = Real.exp (-(C.val i j / gamma)) / ((∑ j' : ι₂, Real.exp (-(C.val i j' / gamma)) * v.val j') * ∑ i' : ι₁, Real.exp (-(C.val i' j / gamma)) * u.val i')) → (∀ (i : ι₁) (j : ι₂), |logRatio i j| ≤ |Real.log min_b| + 2 * C_max / gamma) ∧ 0 ≤ |Real.log min_b| + 2 * C_max / gamma := by
  exact FlowSinkhorn.KLProjection.StatementMap.prop_hgamma_ot

theorem prop_kappa_ot : ∀ {ι₁ : Type u_1} {ι₂ : Type u_2} [inst : Fintype ι₁] [inst_1 : Nonempty ι₁] [inst_2 : Fintype ι₂] [inst_3 : Nonempty ι₂] (alpha : ι₁ → ℝ) (beta : ι₂ → ℝ) (j₀ : ι₂) (Y : ι₁ × ι₂ → ℝ), (∀ (i : ι₁) (j : ι₂), alpha i + beta j = Y (i, j)) → ∃ (w₁ : ι₁ → ℝ) (w₂ : ι₂ → ℝ), (∀ (i : ι₁) (j : ι₂), w₁ i + w₂ j = Y (i, j)) ∧ FlowSinkhorn.KLProjection.coordSupNorm w₁ ≤ FlowSinkhorn.KLProjection.coordSupNorm Y := by
  exact FlowSinkhorn.KLProjection.StatementMap.prop_kappa_ot

theorem cor_ot_xgamma_ugamma : ∀ {ι₁ : Type u_1} {ι₂ : Type u_2} [inst : Fintype ι₁] [inst_1 : Nonempty ι₁] [inst_2 : Fintype ι₂] [inst_3 : Nonempty ι₂] (block : FlowSinkhorn.KLProjection.Applications.OT.SignedBlockSweepData ι₁ ι₂) {gamma min_b C_max : ℝ}, FlowSinkhorn.KLProjection.Applications.OT.ComplexityScalars gamma min_b C_max → ∀ (cert : FlowSinkhorn.KLProjection.Applications.OT.SeparableFixedPointCertificate ι₁ ι₂ block gamma min_b C_max), ∃ (X_gamma : ℝ), X_gamma = 1 ∧ ∀ (k : ℕ), FlowSinkhorn.KLProjection.variationSeminorm ((FlowSinkhorn.KLProjection.sweep block.Ψ₁ block.Ψ₂)^[k] fun (x : ι₁) => 0) ≤ 6 * C_max + 2 * gamma * |Real.log min_b| := by
  exact FlowSinkhorn.KLProjection.StatementMap.cor_ot_xgamma_ugamma

theorem prop_graphw1_v1v2_closed_form : ∀ {vertex : Type u_1} {edge : Type u_2} [inst : Fintype vertex] [inst_1 : Nonempty vertex] [inst_2 : Fintype edge] [inst_3 : Nonempty edge] (v : vertex → ℝ) (U : edge → ℝ), FlowSinkhorn.KLProjection.Setup.blockQuotientSeminorm v = FlowSinkhorn.KLProjection.variationSeminorm v ∧ FlowSinkhorn.KLProjection.Setup.blockQuotientSeminorm U = FlowSinkhorn.KLProjection.variationSeminorm U := by
  exact FlowSinkhorn.KLProjection.StatementMap.prop_graphw1_v1v2_closed_form

theorem prop_graphw1_signed_structure : ∀ {ι : Type u_1} [inst : Fintype ι] [inst_1 : Nonempty ι] (Psi₁ Psi₂ : (ι → ℝ) → ι → ℝ), FlowSinkhorn.KLProjection.IsTopical Psi₁ → FlowSinkhorn.KLProjection.IsTopical Psi₂ → ∀ (x y : ι → ℝ), FlowSinkhorn.KLProjection.variationSeminorm ((Psi₁ ∘ Psi₂) x - (Psi₁ ∘ Psi₂) y) ≤ FlowSinkhorn.KLProjection.variationSeminorm (x - y) := by
  exact FlowSinkhorn.KLProjection.StatementMap.prop_graphw1_signed_structure

theorem prop_graphw1_psi2_closed_nonexp : ∀ {ι : Type u_1} {ε : Type u_2} [inst : Fintype ι] [inst_1 : Nonempty ι] [inst_2 : Fintype ε] [inst_3 : Nonempty ε] (src dst : ε → ι) (v : ι → ℝ), (∀ (e : ε), FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_Psi2 src dst v e = (v (dst e) - v (src e)) / 2) ∧ FlowSinkhorn.KLProjection.variationSeminorm (FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1_Psi2 src dst v) ≤ FlowSinkhorn.KLProjection.variationSeminorm v := by
  exact FlowSinkhorn.KLProjection.StatementMap.prop_graphw1_psi2_closed_nonexp

theorem prop_hgamma_graphw1 : ∀ {edge : Type u_1} [Nonempty edge] (opp : edge → edge) (f z : FlowSinkhorn.KLProjection.Applications.GraphW1.PositiveField edge) (length : edge → ℝ) {XStar lengthMax gamma logZSup : ℝ}, 0 < gamma → 0 ≤ lengthMax → (∀ (e : edge), f.val e ≤ XStar) → (∀ (e : edge), |Real.log (z.val e)| ≤ logZSup) → (∀ (e : edge), length e + length (opp e) ≤ 2 * lengthMax) → (∀ (e : edge), Real.log (f.val e) + Real.log (f.val (opp e)) = Real.log (z.val e) + Real.log (z.val (opp e)) - (length e + length (opp e)) / gamma) → (∀ (e : edge), |Real.log (f.val e) - Real.log (z.val e)| ≤ Real.log XStar + 2 * lengthMax / gamma + 3 * logZSup) ∧ 0 ≤ Real.log XStar + 2 * lengthMax / gamma + 3 * logZSup := by
  exact FlowSinkhorn.KLProjection.StatementMap.prop_hgamma_graphw1

theorem lem_l1_bound_from_feasible : ∀ {ι : Type u_1} [inst : Fintype ι] [inst_1 : Nonempty ι] (C xStar xbar : ι → ℝ) (Constraint : (ι → ℝ) → Prop) (klTerm : (ι → ℝ) → ι → ℝ) {gamma : ℝ}, 0 < gamma → (∀ (i : ι), 0 < C i) → (∀ (x : ι → ℝ), (∀ (i : ι), 0 ≤ x i) ∧ Constraint x → ∀ (i : ι), 0 ≤ klTerm x i) → FlowSinkhorn.KLProjection.Applications.GraphW1.IsFeasibleEntropicMinimizer (fun (x : ι → ℝ) => (∀ (i : ι), 0 ≤ x i) ∧ Constraint x) C gamma (FlowSinkhorn.KLProjection.DualConvergence.coordinateSumKL klTerm) xStar → (∀ (i : ι), 0 ≤ xbar i) ∧ Constraint xbar → ∑ i : ι, xStar i ≤ (∑ i : ι, C i * xbar i + gamma * FlowSinkhorn.KLProjection.DualConvergence.coordinateSumKL klTerm xbar) / FlowSinkhorn.KLProjection.Applications.GraphW1.graphW1CostMin C := by
  exact FlowSinkhorn.KLProjection.StatementMap.lem_l1_bound_from_feasible

theorem prop_kappa_graph_diameter : ∀ {ι : Type u_1} {kappa B : ℝ}, 0 ≤ B → B ≤ 1 → ∀ (graphDiam : ℕ) (yf yg : ι × ι → ℝ) (path : ι → List (ι × ι)), (∀ (p : ι × ι), |yf p| ≤ B) → (∀ (p : ι × ι), |yg p| ≤ B) → (∀ (i : ι), (path i).length ≤ graphDiam) → (∃ (i : ι), kappa ≤ |(List.map (fun (p : ι × ι) => (yf + yg) p) (path i)).sum|) → kappa ≤ 2 * ↑graphDiam := by
  exact FlowSinkhorn.KLProjection.StatementMap.prop_kappa_graph_diameter

theorem cor_graphw1_xgamma_ugamma : ∀ {ι₁ : Type u_1} {ι₂ : Type u_2} [inst : Fintype ι₁] [inst_1 : Nonempty ι₁] (block : FlowSinkhorn.KLProjection.Applications.GraphW1.SignedBlockSweepData ι₁ ι₂) {kappa lengthMax gamma hGamma bMass p lengthMin : ℝ} (fixed : FlowSinkhorn.KLProjection.Applications.GraphW1.SweepFixedPointBudget ι₁ (FlowSinkhorn.KLProjection.sweep block.Ψ₁ block.Ψ₂) kappa lengthMax gamma hGamma) (edge : FlowSinkhorn.KLProjection.Applications.GraphW1.UnitBoundedTwoStepFields ι₁) (graphDiam : ℕ) (path : FlowSinkhorn.KLProjection.Applications.GraphW1.TwoStepPathCertificate edge graphDiam kappa), 0 < gamma → 0 ≤ lengthMax + gamma * hGamma → ∀ (mass : FlowSinkhorn.KLProjection.Applications.GraphW1.GraphW1MassProxy ι₁ (FlowSinkhorn.KLProjection.sweep block.Ψ₁ block.Ψ₂) gamma bMass p lengthMin), ∃ (U_gamma : ℝ) (X_gamma : ℝ), U_gamma = 4 * ↑graphDiam * (lengthMax + gamma * hGamma) ∧ X_gamma = bMass * U_gamma / gamma + p * Real.exp (-lengthMin / gamma) ∧ ∀ (k : ℕ), FlowSinkhorn.KLProjection.variationSeminorm ((FlowSinkhorn.KLProjection.sweep block.Ψ₁ block.Ψ₂)^[k] 0) ≤ U_gamma ∧ mass.xMass k ≤ X_gamma := by
  exact FlowSinkhorn.KLProjection.StatementMap.cor_graphw1_xgamma_ugamma

theorem prop_topical_nonexpansive : ∀ {ι : Type u_1} [inst : Fintype ι] [inst_1 : Nonempty ι] (T : (ι → ℝ) → ι → ℝ), Monotone T → FlowSinkhorn.KLProjection.TranslationEquivariant T → ∀ (x y : ι → ℝ), FlowSinkhorn.KLProjection.variationSeminorm (T x - T y) ≤ FlowSinkhorn.KLProjection.variationSeminorm (x - y) := by
  exact FlowSinkhorn.KLProjection.StatementMap.prop_topical_nonexpansive

theorem prop_block_monotone : ∀ {ι₁ : Type u_1} {ι₂ : Type u_2} (R₂ : (ι₂ → ℝ) → (ι₂ → ℝ) → Prop) (Ψ₁ : (ι₂ → ℝ) → ι₁ → ℝ) (Ψ₂ : (ι₁ → ℝ) → ι₂ → ℝ), (∀ {u v : ι₂ → ℝ}, R₂ u v → Ψ₁ v ≤ Ψ₁ u) → (∀ {u v : ι₁ → ℝ}, u ≤ v → R₂ (Ψ₂ v) (Ψ₂ u)) → (∀ {u v : ι₂ → ℝ}, R₂ u v → Ψ₁ v ≤ Ψ₁ u) ∧ (∀ {u v : ι₁ → ℝ}, u ≤ v → R₂ (Ψ₂ v) (Ψ₂ u)) ∧ Monotone (FlowSinkhorn.KLProjection.sweep Ψ₁ Ψ₂) := by
  exact FlowSinkhorn.KLProjection.StatementMap.prop_block_monotone

theorem lem_moment_monotone : ∀ {source : Type u_1} {atom : Type u_2} {moment : Type u_3} [inst : Fintype source] [inst_1 : Fintype atom] (A : source → atom → ℝ) (B : atom → moment → ℝ) (x y : source → ℝ), (∀ (r : source) (i : atom), 0 ≤ A r i) → (∀ (i : atom) (j : moment), 0 ≤ B i j) → (∀ (r : source), x r ≤ y r) → (fun (j : moment) => ∑ i : atom, B i j * ∑ r : source, A r i * x r) ≤ fun (j : moment) => ∑ i : atom, B i j * ∑ r : source, A r i * y r := by
  exact FlowSinkhorn.KLProjection.StatementMap.lem_moment_monotone

theorem prop_translation_equivariance : ∀ {ι₁ : Type u_1} {ι₂ : Type u_2} (τ : FlowSinkhorn.KLProjection.PairedSign) (pairedBalance : Prop) (Ψ₁ : (ι₂ → ℝ) → ι₁ → ℝ) (Ψ₂ : (ι₁ → ℝ) → ι₂ → ℝ), pairedBalance → (pairedBalance → FlowSinkhorn.KLProjection.SignedBlockTranslationEquivariant1 τ Ψ₁) → (pairedBalance → FlowSinkhorn.KLProjection.SignedBlockTranslationEquivariant2 τ Ψ₂) → FlowSinkhorn.KLProjection.SignedBlockTranslationEquivariant2 τ Ψ₂ ∧ FlowSinkhorn.KLProjection.SignedBlockTranslationEquivariant1 τ Ψ₁ ∧ FlowSinkhorn.KLProjection.TranslationEquivariant (FlowSinkhorn.KLProjection.sweep Ψ₁ Ψ₂) := by
  exact FlowSinkhorn.KLProjection.StatementMap.prop_translation_equivariance

