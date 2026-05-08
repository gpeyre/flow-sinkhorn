import FlowSinkhorn.KLProjection.StatementMap

/-!
# Paper Statement Map Facade

Re-export of the canonical synchronization layer.

Use this import when navigating by paper labels first:
- `FlowSinkhorn.Paper.StatementMap`

Canonical source remains:
- `FlowSinkhorn.KLProjection.StatementMap`

Implementation-file hints:
- each alias in `FlowSinkhorn.KLProjection.StatementMap` carries an `-- impl: ...` comment
  indicating where the underlying theorem constant is defined.
-/

namespace FlowSinkhorn
namespace Paper
namespace StatementMap

open KLProjection.StatementMap

end StatementMap
end Paper
end FlowSinkhorn
