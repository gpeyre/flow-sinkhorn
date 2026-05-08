import FlowSinkhorn.KLProjection.Legacy.Section2Duality
import FlowSinkhorn.KLProjection.DualConvergence
import FlowSinkhorn.KLProjection.PrimalDualBounds
import FlowSinkhorn.KLProjection.Setup
import FlowSinkhorn.KLProjection.Applications.OT
import FlowSinkhorn.KLProjection.Applications.GraphW1

/-!
# Blueprint-aligned entrypoint

This module provides a paper-order import surface for the KL-projection blueprint:
1. dual setup;
2. dual rate;
3. primal/dual bound transfer;
4. non-expansiveness setup;
5. OT and graph-W1 instantiations.
-/
