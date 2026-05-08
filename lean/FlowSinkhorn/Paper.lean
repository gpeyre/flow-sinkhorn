import FlowSinkhorn.Paper.MainBody
import FlowSinkhorn.Paper.Appendix
import FlowSinkhorn.Paper.StatementMap

/-!
# Paper-Structured Facade

Primary paper-facing navigation layer for the Lean formalization.

This facade is non-breaking:
- existing canonical modules stay under `FlowSinkhorn.KLProjection.*`;
- this namespace adds section/appendix entrypoints aligned with the manuscript.

Recommended usage for readers/auditors:
- import `FlowSinkhorn.Paper`
- navigate by `SectionN` / `AppendixX` modules.
-/
