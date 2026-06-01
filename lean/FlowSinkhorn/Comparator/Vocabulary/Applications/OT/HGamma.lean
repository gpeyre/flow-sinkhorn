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
# OT H_gamma statement vocabulary

This proof-free module contains finite data records used by paper-facing OT
`H_gamma` statements.  The proofs live in `HGamma.lean`.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace Applications
namespace OT

open scoped BigOperators

/--
Finite probability vector.

This records the paper phrase "probability vector" as data: nonnegative
coordinates with total mass one.
-/
structure ProbabilityVector (ι : Type*) [Fintype ι] where
  val : ι → ℝ
  nonneg : ∀ i, 0 ≤ val i
  mass : (∑ i, val i) = 1

instance {ι : Type*} [Fintype ι] : CoeFun (ProbabilityVector ι) (fun _ => ι → ℝ) where
  coe b := b.val

/--
Nonnegative real-valued field.

This is used for finite Sinkhorn scaling vectors `u` and `v` in the
paper-facing OT certificate.
-/
structure NonnegativeField (ι : Type*) where
  val : ι → ℝ
  nonneg : ∀ i, 0 ≤ val i

instance {ι : Type*} : CoeFun (NonnegativeField ι) (fun _ => ι → ℝ) where
  coe u := u.val

/--
Cost field certified to lie in `[0,Cmax]`.

This packages the finite cost-envelope assumptions used in the OT `H_gamma`
bound, while keeping the scalar `Cmax` explicit in the theorem statement.
-/
structure BoundedCostField (ι₁ ι₂ : Type*) (Cmax : ℝ) where
  val : ι₁ → ι₂ → ℝ
  nonneg : ∀ i j, 0 ≤ val i j
  le_bound : ∀ i j, val i j ≤ Cmax

instance {ι₁ ι₂ : Type*} {Cmax : ℝ} :
    CoeFun (BoundedCostField ι₁ ι₂ Cmax) (fun _ => ι₁ → ι₂ → ℝ) where
  coe C := C.val

end OT
end Applications
end KLProjection
end FlowSinkhorn
