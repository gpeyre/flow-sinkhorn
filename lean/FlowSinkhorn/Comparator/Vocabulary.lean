import FlowSinkhorn.Comparator.Vocabulary.Legacy.Section2Duality
import FlowSinkhorn.Comparator.Vocabulary.UniformBound
import FlowSinkhorn.Comparator.Vocabulary.Topical
import FlowSinkhorn.Comparator.Vocabulary.BlockQuotient
import FlowSinkhorn.Comparator.Vocabulary.Setup.VariationGeometry
import FlowSinkhorn.Comparator.Vocabulary.Sweep
import FlowSinkhorn.Comparator.Vocabulary.PrimalDualBounds
import FlowSinkhorn.Comparator.Vocabulary.DualConvergence
import FlowSinkhorn.Comparator.Vocabulary.Applications.OT.HGamma
import FlowSinkhorn.Comparator.Vocabulary.Applications.OT.Complexity
import FlowSinkhorn.Comparator.Vocabulary.Applications.GraphW1.ClosedForms
import FlowSinkhorn.Comparator.Vocabulary.Applications.GraphW1.HGamma
import FlowSinkhorn.Comparator.Vocabulary.Applications.GraphW1.Complexity

/-!
# Comparator statement vocabulary

This umbrella module is for navigation and local experimentation with the
trusted statement language used by `FlowSinkhorn.Comparator.Challenge`.

The frozen challenge intentionally imports the vocabulary modules above
explicitly rather than importing this umbrella.  Keeping the challenge imports
flat makes the trusted import surface easy to audit and lets
`scripts/check_comparator_scaffold.py` compare it against an exact allowlist.

Rules for this subtree:

* It may contain definitions, structures, predicates, notation, and lightweight
  instances needed to state paper theorems.
* It must not contain paper-facing proofs, theorem declarations, axioms, opaque
  constants, unsafe declarations, or proof holes.
* It must not import implementation-side `FlowSinkhorn.KLProjection.*` modules.

Existing `FlowSinkhorn.KLProjection.*Vocabulary` files are compatibility
imports into these canonical modules.  They should remain import-only shims so
that Challenge and Solution share one vocabulary without duplicating
definitions.
-/
