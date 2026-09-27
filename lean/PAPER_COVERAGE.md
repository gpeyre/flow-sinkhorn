# Revised article: formalization coverage

Status: **partial; not a full certification of `paper/paper.tex`** (September 2026).

`lake build` checks Lean declarations as stated. It does not prove that their
hypotheses match the article. A theorem that assumes a projection certificate,
an ascent identity, a bias estimate, or a final error bound does not derive that
fact from the original variational problem. Historical alias/shape counts and
generated Comparator statements are insufficient evidence of such a derivation.

## Revised statements requiring synchronization

| Area | Revised mathematical requirement | Formalization work still required |
| --- | --- | --- |
| Setting and duality | Closed nonnegative feasible set; strictly positive feasible point; finite attained LP optimum; existence of exact block solves | Recheck existence/positivity and dual attainment against the certificate interfaces |
| Pinsker | Possibly unequal masses bounded by a common upper bound; coefficient `1/(2X)` | Existing equal-mass endpoints do not cover this generality |
| Main dual rate | Preliminary second-block solve; all full/half-step masses bounded; first-block quotient radius; constant 8 | Reconnect the actual variational ascent and residual lemmas, rather than supplying them as certificates |
| Bias/accuracy | General positive reference and explicit `B0`; scalar optimal-value estimator | Prove the bias envelope and distinguish it from feasible-primal or potential accuracy |
| Primal helper | Pairing budget, or a *simultaneous* quotient radius; independent block infima do not suffice | Remove any implicit common-gauge inference |
| Graph lift | Half cost and half regularization; outgoing-minus-incoming divergence; averaged edge multiplier | Synchronize operator definitions and all physical/lifted factors |
| Graph order | Separable scalar root is increasing in edge multipliers | Prove this for the actual map; a nonnegative vector Jacobian or signature alone is insufficient |
| Graph constants | `a=2`, `kappa1 <= D_hop`, reference-dependent `H`, full/half lifted mass | Supply the variational and geometric bridges, not bounds as assumptions |
| Graph complexity | Explicit bias and iteration constants; fixed-data exponent 3; optimal value only | Derive final accuracy from these ingredients, not from a hypothesis already bounding the error |
| General Bregman/quantum | Stated domain hypotheses, factor 4, correct dual constant, trace/operator quotient geometry | No claim of complete formalization or a proved quantum complexity theorem |

## How to read the existing project

- `FlowSinkhorn/KLProjection.lean` imports the reusable mathematical development.
- `DualConvergence/`, `PrimalDualBounds/`, `Setup/`, and `Applications/` contain its
  principal thematic components.
- `KLProjection/StatementMap.lean` and the `Paper/` facade retain **historical** aliases.
  Their theorem numbers are not updated labels for the revised PDF.
- `Comparator/Challenge.lean` is statement-only and intentionally contains `sorry`;
  it is not a proof-producing certificate. Automatically comparing it against the
  implementation that supplied those statements cannot establish fidelity to the article.
- Earlier files in `audit/` are historical records, not current completion certificates.

The revision does not silently rename old aliases to stronger statements. Missing
proofs must be supplied before updating the certification claim. Optional local
label audits use `paper/paper.aux` and can fail until that work is complete; CI
reports this scope and builds the Lean library without asserting manuscript coverage.
