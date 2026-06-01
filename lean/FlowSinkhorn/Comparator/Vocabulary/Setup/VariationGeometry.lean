import FlowSinkhorn.Comparator.Vocabulary.BlockQuotient

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
# Variation-geometry statement vocabulary

This statement-vocabulary module contains definitions from the variation-geometry
setup that are needed to state paper-facing results.  The corresponding proofs
remain in `FlowSinkhorn.KLProjection.Setup.VariationGeometry`.

Keeping this file free of paper-facing proof endpoints lets the Comparator
challenge import the statement vocabulary without also importing the canonical
implementation theorems it is supposed to check independently.
-/

namespace FlowSinkhorn
namespace KLProjection
namespace Setup

/--
Block-quotient seminorm for a single finite-dimensional real block.

Paper role: Definition `def:block-seminorm` (first block, constant-shift quotient direction).
This is the infimum of `‖x + c · 1‖_∞` over all scalar shifts `c`, and the proof module
shows that it equals the variation seminorm.
-/
noncomputable def blockQuotientSeminorm {ι : Type*} [Fintype ι] [Nonempty ι] (x : ι → ℝ) : ℝ :=
  sInf (Set.range fun c : ℝ => coordSupNorm (fun i => x i + c))

end Setup
end KLProjection
end FlowSinkhorn
