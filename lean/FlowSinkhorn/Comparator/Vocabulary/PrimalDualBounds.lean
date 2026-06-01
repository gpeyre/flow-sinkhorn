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
# Primal-dual bound statement vocabulary

This proof-free module contains the common budget expression used to state the
fixed-point, OT, and graph-W1 primal-dual bounds.  Keeping it separate from the
proof file `FixedPointControl.lean` lets Comparator challenges mention the
budget without importing the uniform-iterate proof endpoint.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace PrimalDualBounds

open scoped BigOperators

/--
Paper-facing fixed-point budget from `H_γ` and `κ`.

This packages the optimizer-specific control expected in Proposition 4.2:
once one proves `p u_γ ≤ κ (cost + γ H_γ)` for the fixed point, it can be
re-used as an explicit constant in every iterate bound.
-/
def hGammaKappaBudget (kappa cost gamma hGamma : ℝ) : ℝ :=
  kappa * (cost + gamma * hGamma)

/--
Finite cost lower-bound certificate.

This is the statement-vocabulary form of the paper assumption `C_i >= C_min`.
-/
def CostLowerBound {coord : Type*} (C : coord → ℝ) (Cmin : ℝ) : Prop :=
  ∀ i : coord, Cmin ≤ C i

/--
Uniform quotient-radius certificate for a finite dual-potential sequence.

The paper writes a block-quotient radius bound `||u^(k)||_V <= Umax`.  In finite coordinates this
means that each iterate admits a gauge/quotient representative, encoded here by a shift orthogonal
to `b`, whose coordinates are bounded by `Umax`.
-/
def FiniteQuotientRadiusBound {pot : Type*} [Fintype pot]
    (b : pot → ℝ) (u : ℕ → pot → ℝ) (Umax : ℝ) : Prop :=
  ∀ k : ℕ, ∃ shift : pot → ℝ,
    (∑ j : pot, b j * shift j) = 0 ∧
      ∀ j : pot, |u k j + shift j| ≤ Umax

/--
Displayed finite-pairing ascent certificate.

This is the exact scalar monotonicity statement for the finite dual objective used in
Proposition 4.2, namely `sum_j b_j u_j - gamma * mass`.
-/
def DisplayedFinitePairingAscent {pot : Type*} [Fintype pot]
    (b : pot → ℝ) (u : ℕ → pot → ℝ) (xMass : ℕ → ℝ) (gamma : ℝ) : Prop :=
  ∀ k : ℕ,
    (∑ j : pot, b j * u k j) - gamma * xMass k ≤
      (∑ j : pot, b j * u (k + 1) j) - gamma * xMass (k + 1)

/--
Zero-start primal-mass certificate.

This records the paper's zero dual initialization together with the corresponding finite
exponential primal mass at time zero.
-/
def ZeroStartPrimalMass {coord pot : Type*} [Fintype coord]
    (C : coord → ℝ) (u : ℕ → pot → ℝ) (xMass : ℕ → ℝ) (gamma : ℝ) : Prop :=
  (∀ j : pot, u 0 j = 0) ∧
    xMass 0 = ∑ i : coord, Real.exp (-C i / gamma)

/--
Named finite certificate for Proposition `prop:mass-bound-block`.

The paper proof combines exactly these ingredients: positive temperature, a
finite cost floor, a quotient/gauge radius bound for every dual iterate, ascent
of the displayed finite dual objective, and the zero-start primal-mass identity.
Packaging them as a single statement-vocabulary structure makes the Comparator
boundary say precisely which dynamical facts must be supplied by the concrete
KL block-update construction.
-/
structure PrimalMassBoundBlockCertificate {coord : Type*} [Fintype coord]
    {pot : Type*} [Fintype pot]
    (xMass : ℕ → ℝ)
    (C : coord → ℝ)
    (b : pot → ℝ)
    (u : ℕ → pot → ℝ)
    (Umax gamma Cmin : ℝ) : Prop where
  gamma_positive : 0 < gamma
  cost_lower_bound : CostLowerBound C Cmin
  quotient_radius_bound : FiniteQuotientRadiusBound b u Umax
  displayed_pairing_ascent : DisplayedFinitePairingAscent b u xMass gamma
  zero_start_primal_mass : ZeroStartPrimalMass C u xMass gamma

end PrimalDualBounds
end KLProjection
end FlowSinkhorn
