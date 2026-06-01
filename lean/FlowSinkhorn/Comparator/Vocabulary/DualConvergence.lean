import Mathlib

/-!
Canonical proof-free Comparator vocabulary.

This module is part of the trusted statement language imported by
`FlowSinkhorn.Comparator.Challenge`.  It may define structures, predicates,
and auxiliary notation used to state paper theorems, but it must not contain
paper-facing proofs, theorem declarations, axioms, or opaque constants.
The implementation imports this same vocabulary through compatibility shims,
so Challenge and Solution share one statement language without duplicating
definitions.
-/

/-!
# Dual-convergence statement vocabulary

This proof-free module contains the definitions needed to state the dual-rate,
Pinsker, per-step ascent, gap-vs-residual, and KL-bias paper results.  The
proofs remain in the specialized modules `Pinsker.lean`, `PerStepAscent.lean`,
`GapResidual.lean`, and `Rate.lean`.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace DualConvergence

open scoped BigOperators

variable {n : ℕ}

/-- Finite-dimensional KL expression used in the appendix argument. -/
noncomputable def finiteKL (p q : Fin n → ℝ) : ℝ :=
  ∑ i, p i * Real.log (p i / q i)

/-- Finite-dimensional `ℓ¹` norm. -/
noncomputable def l1Norm (v : Fin n → ℝ) : ℝ :=
  ∑ i, |v i|

/-- Finite `ℓ¹` norm used in the quotient-gap Hölder bridge. -/
noncomputable def finiteL1 {n : ℕ} (r : Fin n → ℝ) : ℝ :=
  ∑ i, |r i|

/-- Two-block finite `ℓ¹` norm used by the block-quotient A.2 bridge. -/
noncomputable def finiteL1Pair {ι₁ ι₂ : Type*} [Fintype ι₁] [Fintype ι₂]
    (r₁ : ι₁ → ℝ) (r₂ : ι₂ → ℝ) : ℝ :=
  (∑ i, |r₁ i|) + ∑ j, |r₂ j|

/--
Exact additive objective identity for a finite block update.
-/
def FiniteKLExactAddGainFromBlockUpdate
    {n : ℕ} (p q : Fin n → ℝ) (Fbefore Fafter : ℝ) : Prop :=
  Fafter = Fbefore + finiteKL p q

/--
Gamma-scaled exact additive objective identity for a finite block update.

This is the literal form used by the paper's dual objective increment
`F_γ(after)-F_γ(before)=γ KL(after‖before)`.
-/
def FiniteKLGammaExactAddGainFromBlockUpdate
    {n : ℕ} (p q : Fin n → ℝ) (gamma Fbefore Fafter : ℝ) : Prop :=
  Fafter = Fbefore + gamma * finiteKL p q

/--
Exact-gain variant of the support-aware mass-shell block-update certificate.
-/
structure FiniteMassShellExactSupportBlockUpdateCertificate
    {n : ℕ} (p q : Fin n → ℝ) (M Fbefore Fafter : ℝ) : Prop where
  source_nonneg : ∀ i, 0 ≤ p i
  source_mass : ∑ i, p i = M
  update_nonneg : ∀ i, 0 ≤ q i
  update_mass : ∑ i, q i = M
  support : ∀ i, q i = 0 → p i = 0
  exact_gain : FiniteKLExactAddGainFromBlockUpdate p q Fbefore Fafter

/--
Gamma-scaled exact-gain support-aware mass-shell block-update certificate.
-/
structure FiniteMassShellGammaExactSupportBlockUpdateCertificate
    {n : ℕ} (p q : Fin n → ℝ) (M gamma Fbefore Fafter : ℝ) : Prop where
  source_nonneg : ∀ i, 0 ≤ p i
  source_mass : ∑ i, p i = M
  update_nonneg : ∀ i, 0 ≤ q i
  update_mass : ∑ i, q i = M
  support : ∀ i, q i = 0 → p i = 0
  exact_gain : FiniteKLGammaExactAddGainFromBlockUpdate p q gamma Fbefore Fafter

/--
Linear objective used by the KL-bias lemma.
-/
def linearObjective
    {coord : Type*} [Fintype coord]
    (C x : coord → ℝ) : ℝ :=
  ∑ i, C i * x i

/--
Finite coordinate-sum KL functional used by the paper-facing KL-bias lemma.
-/
def coordinateSumKL
    {coord : Type*} [Fintype coord]
    (klTerm : (coord → ℝ) → coord → ℝ) (x : coord → ℝ) : ℝ :=
  ∑ i, klTerm x i

/--
Entropically regularized objective used by Appendix Lemma `app-lem:kl-bias`.
-/
def regularizedObjective
    {coord : Type*} [Fintype coord]
    (C : coord → ℝ) (gamma : ℝ) (KL : (coord → ℝ) → ℝ) (x : coord → ℝ) : ℝ :=
  linearObjective C x + gamma * KL x

/--
Positive scalar constants in the paper's Section-3 KL dual-rate theorem.

This is proof-free statement vocabulary for the assumptions
`gamma > 0`, `Xmax > 0`, `Umax > 0`, and `||A|| > 0`.
-/
structure PositiveKLRateConstants
    (gamma Xmax Umax Anorm : ℝ) : Prop where
  gamma_pos : 0 < gamma
  xmax_pos : 0 < Xmax
  umax_pos : 0 < Umax
  anorm_pos : 0 < Anorm

/--
Scalar Appendix-A ingredients used to prove the paper's Section-3 KL dual rate.

The paper proof combines nonnegative dual gaps, the quotient gap-vs-residual estimate
`Delta_k <= 2*Umax*residual_k`, and the Pinsker/per-step-ascent estimate
`gamma/(2*Xmax*||A||^2)*residual_k^2 <= Delta_k-Delta_{k+1}`.  This record names exactly those
three scalar facts without importing any implementation theorem.
-/
structure KLRateScalarIngredients
    (gap residual : ℕ → ℝ) (gamma Xmax Umax Anorm : ℝ) : Prop where
  gap_nonneg : ∀ k : ℕ, 0 ≤ gap k
  gap_residual : ∀ k : ℕ, gap k ≤ (2 * Umax) * residual k
  per_step_ascent : ∀ k : ℕ,
    (gamma / (2 * Xmax * Anorm ^ 2)) * residual k ^ 2 ≤ gap k - gap (k + 1)

/--
Named certificate for Theorem `thm:kl-dual-rate`.

This bundles the complete scalar interface of the paper proof: positivity of
the constants appearing in the displayed rate and the Appendix-A scalar
ingredients obtained from per-step ascent plus gap-vs-residual control.  The
proof-bearing rate module unfolds this certificate and derives the reciprocal
`O(1/k)` bound internally.
-/
structure KLDualRateCertificate
    (gap residual : ℕ → ℝ) (gamma Xmax Umax Anorm : ℝ) : Prop where
  constants : PositiveKLRateConstants gamma Xmax Umax Anorm
  scalar_ingredients : KLRateScalarIngredients gap residual gamma Xmax Umax Anorm

/--
The unregularized optimizer predicate for the KL-bias comparison.
-/
def IsLinearMinimizer
    {coord : Type*} [Fintype coord]
    (Feasible : (coord → ℝ) → Prop)
    (C : coord → ℝ) (xStar : coord → ℝ) : Prop :=
  Feasible xStar ∧ ∀ y, Feasible y → linearObjective C xStar ≤ linearObjective C y

/--
The regularized optimizer predicate for the KL-bias comparison.
-/
def IsRegularizedMinimizer
    {coord : Type*} [Fintype coord]
    (Feasible : (coord → ℝ) → Prop)
    (C : coord → ℝ) (gamma : ℝ) (KL : (coord → ℝ) → ℝ) (xStar : coord → ℝ) : Prop :=
  Feasible xStar ∧
    ∀ y, Feasible y → regularizedObjective C gamma KL xStar ≤
      regularizedObjective C gamma KL y

/--
Nonnegativity of a functional on the feasible set.
-/
def NonnegativeOn
    {coord : Type*}
    (Feasible : (coord → ℝ) → Prop) (KL : (coord → ℝ) → ℝ) : Prop :=
  ∀ x, Feasible x → 0 ≤ KL x

/--
Finite LP/KL-bias certificate used by the paper-facing approximation theorem.

This proof-free record packages exactly the finite ingredients discharged by Appendix Lemma B.1:
`x0` is a linear minimizer, `xgamma` is a regularized minimizer, the finite KL functional is
nonnegative on the feasible set, the reference-entropy terms are coordinatewise bounded by
`x0_i * log(card coord)`, and `x0` has mass at most `XmaxZero`.
-/
structure FiniteKLBiasApproximationCertificate
    {coord : Type*} [Fintype coord]
    (C x0 xgamma : coord → ℝ)
    (Feasible : (coord → ℝ) → Prop)
    (klTerm : (coord → ℝ) → coord → ℝ)
    (gamma XmaxZero : ℝ) : Prop where
  linear_min : IsLinearMinimizer Feasible C x0
  regularized_min :
    IsRegularizedMinimizer Feasible C gamma (coordinateSumKL klTerm) xgamma
  kl_nonneg : NonnegativeOn Feasible (coordinateSumKL klTerm)
  coordinate_envelope :
    ∀ i : coord, klTerm x0 i ≤ x0 i * Real.log (Fintype.card coord : ℝ)
  mass_bound : (∑ i : coord, x0 i) ≤ XmaxZero

/--
Proof-free certificate for the paper-facing approximation theorem.

This names the primitive ingredients used by Theorem `thm:approx-linprog`: the
accuracy side conditions and temperature choice, the finite LP/KL-bias
certificate, the link between displayed dual values and the gap sequence, and
the Section-3 dual-rate certificate.
-/
structure ApproxLinprogCertificate
    {coord : Type*} [Fintype coord]
    (C x0 xgamma : coord → ℝ)
    (Feasible : (coord → ℝ) → Prop)
    (klTerm : (coord → ℝ) → coord → ℝ)
    (gamma Xmax Umax Anorm XmaxZero maxMass eps : ℝ)
    (Fgamma gap residual : ℕ → ℝ) : Prop where
  eps_pos : 0 < eps
  rate_certificate : KLDualRateCertificate gap residual gamma Xmax Umax Anorm
  xmaxZero_logcard_pos : 0 < XmaxZero * Real.log (Fintype.card coord : ℝ)
  xmaxZero_le_maxMass : XmaxZero ≤ maxMass
  gamma_choice : gamma = eps / (2 * XmaxZero * Real.log (Fintype.card coord : ℝ))
  bias :
    FiniteKLBiasApproximationCertificate C x0 xgamma Feasible klTerm gamma XmaxZero
  gap_eval :
    ∀ n : ℕ,
      gap (n + 1) = regularizedObjective C gamma (coordinateSumKL klTerm) xgamma - Fgamma n

end DualConvergence
end KLProjection
end FlowSinkhorn
