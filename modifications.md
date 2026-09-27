# Final article revision

Date: 27 September 2026.

## Scope and source material

The canonical article is now `paper/paper.tex`, with a compiled `paper/paper.pdf`.
The revision uses the submitted manuscript (`reviews/original.pdf`), all three
reviews (`reviews/reviews.md`), the rebuttal (`reviews/answers.md`), the detailed
mathematical/Lean audit (`reviews/mathematical-error.md`), and the more recent
manuscript previously in `neurips/`. The submitted numbering is treated as
historical; current references are generated from labels, not copied numbers.

The reviews and existing rebuttal are not rewritten or silently reclassified as
proved results. In particular, incorrect or overly broad rebuttal claims are not
repeated merely to agree with the response. The untracked review documents and
the pre-existing `.DS_Store` modification are left out of this publication commit.

## Publication format and organization

- Renamed `neurips/` to `paper/` and removed the unused conference style/templates
  and NeurIPS checklist. The article uses standard `article`, geometry, Latin
  Modern, numeric bracketed citations, and no anonymity or conference footer.
- Restored Gabriel Peyre's author/affiliation block and the requested ERC WOLF /
  ANR-23-IACL-0008 (PRAIRIE-PSAI) acknowledgement.
- Split the monolithic source into five main sections and thematic proof
  appendices. Preserved balanced OT, Bregman and quantum discussions, graph
  proofs, non-expansiveness, the Lean discussion, notation, and all nine used
  CPU/GPU figure panels. Removed obsolete certificate-shaped statements rather
  than presenting an assumed conclusion as a mathematical result.
- Rebuilt the TikZ roadmap from current hypotheses, constants, and live labels.
  It no longer asserts the invalid general signed-matrix inverse argument, the
  wrong stacked-operator norm, or an incorrectly oriented Pinsker inequality.
- Updated repository documentation, benchmark output paths, and optional Lean
  audit paths. The audit readers now expand the multi-file LaTeX sources and
  report actual included-file locations rather than assuming one giant file.
- Moved old numerical report tables from the article to
  `benchmarks/results/tables/`; none is required to compile the article.
- Added `paper/README.md`, a clean build target, and `make -C paper arxiv` to
  generate a source archive containing exactly the article's dependencies.
  This prepares an uploadable archive but does not submit anything to arXiv.

## Reviewer 1: mathematical corrections

### 1. Duplicated graph cost

The lifted objective is now
$\frac12(\langle W,f\rangle+\langle W,g\rangle)
+\frac\gamma2(\mathrm{KL}(f|z)+\mathrm{KL}(g|z))$.
On `f=g` this is precisely the intended physical flow objective. The generic
cost is `(W/2,W/2)`, reference `(z,z)`, and regularization `theta=gamma/2`.
The Gibbs kernel and projection sequence are unchanged, while values and gaps
have the correct normalization. The final complexity proof substitutes these
lifted parameters explicitly.

Locations: `paper/sections/graph_transport.tex`, `paper/appendices/graph_proofs.tex`.

### 2. Divergence and multiplier signs

The convention is uniformly `f_ij: j -> i`, with outgoing-minus-incoming
`Bf=f^T 1-f1`, `A1(f,g)=-f1+g^T1`, and `A2(f,g)=f-g`.
The adjoint is `(-v_i+U_ij, v_j-U_ij)`. Therefore the second-block maximizer is
`U_ij=(v_i+v_j)/2`, not a difference of potentials. The projection root, stable
potential update, stationarity, and residual routing all use the same convention.

### 3. General reference versus unit reference

The positive reference is retained throughout. The dual includes `gamma sum(z)`;
its zero-start mass is `sum_i z_i exp(-C_i/gamma)`, bounded by
`sum(z) exp(-C_min/gamma)`. Graph log-ratio and bias envelopes explicitly include
`||log z||_infinity`, `min(z)`, and `sum(z)`. No unit-reference formula is silently
applied to an arbitrary reference.

### 4. Hidden regularization dependence

The abstract bound is stated as
$\Delta_k\leq8X_\gamma U_\gamma^2\lVert A\rVert_{1\to1}^2/(\gamma k)$.
The abstract, introduction, theorem discussion, and graph proof distinguish a
parameterized `1/(gamma k)` rate from the substituted graph bound, which is
`O(1/(gamma^2 k))` for fixed graph data because `X_gamma=O(1/gamma)`.
Balanced OT has unit mass and really does retain the former dependence.

### 5. Pinsker without mass preservation

Lemma A.3 now proves
$\mathrm{KL}(p|q)\geq\lVert p-q\rVert_1^2/(2X)$ whenever both masses are at most
`X`, without requiring equal masses. The proof uses the entropy Hessian integral
and weighted Cauchy--Schwarz. Boundary cases use an explicit positive
approximation and scalar convergence, not an invalid appeal to lower
semicontinuity in the wrong direction. The main theorem bounds both full and
intermediate half-step masses.

### 6. Accuracy exponent

The generic bias is measured by an explicit envelope
`B0 >= KL(x0_star|z)`. Choosing `gamma=epsilon/(2 B0)` gives the conservative
generic threshold `32 X_gamma U_gamma^2 a^2 B0 / epsilon^2`.
For the normalized graph lift the physical threshold is
`256 B0 X_gamma U_gamma^2 / epsilon^2`, where `X_gamma` is lifted mass.
The resulting fixed-data arithmetic count is `O(p epsilon^-3)`, not the earlier
`epsilon^-4` claim. Full graph/reference dependence remains visible in the
nonasymptotic formulas. No asymptotic graph-family dependence is silently hidden.

The theorem estimates the **unregularized optimal value**. It does not assert
that an intermediate flow is feasible or has the same accuracy in L2 distance.
A separate residual-routing construction gives a posteriori feasible-primal
corrections and is not conflated with this iteration guarantee.

### 7. Impossible edge-count condition

Removed `p=o(1/log(1/epsilon))`. The exact exponential term in the primal bound
is retained, so no coupling condition between `p` and `epsilon` is required.

## Reviewer 2: exposition, comparison, split, and stability

- Simplified the main narrative to the two convergence theorems, two bound
  helpers, sparse graph updates, and their explicit complexity theorem. Full
  auxiliary arguments are in referenced appendices; 23 theorem-like statements
  now have labels across the article, compared with 27 historical aliases.
- Explained that no network-simplex/Orlin runtime comparison has been performed.
  The experiments compare sparse and dense entropy-based methods only. Removed
  universal low-accuracy speed claims and corrected the conflation of Orlin's
  strongly polynomial scaling method with network simplex.
- Explained that choosing the constraint split changes the block-solve cost,
  quotient geometry, non-expansiveness question, and convergence constants.
  An efficient, stable, analyzable split is a problem-specific construction,
  not a general consequence of entropic LP regularization.
- Distinguished entropy's finite boundary value from a log barrier. Added the
  limits of log-sum-exp stabilization for general LPs and the extra analysis
  needed for inexact block solves or finite-precision runtime claims.
- For graph flow, derived the scalar inverse and translation laws directly.
  The stable implementation description now also avoids overflow inside the
  inverse-hyperbolic-sine term, rather than only stabilizing exponential sums.

## Reviewer 3: other applications and the sparse/dense distinction

- Gave a complete balanced-OT specialization with unit primal mass, direct
  soft-transform variation bounds, an optimum log-ratio bound, and a mutual
  information bias envelope.
- Expanded quantum OT in the main introduction and Appendix C.1: exact
  partial-trace variational problem, Umegaki KL, the dual constant, variational
  block equations, nuclear/operator geometry, quantum Pinsker, spectral
  quotient, and explicit partial-trace/quotient operator bounds.
- Corrected the rebuttal's overly broad quantum open-problem wording. Existing
  noncommutative Sinkhorn convergence and a priori dual estimates are proved by
  Feliciangeli--Gerolin--Portinale, and convex-regularized extensions exist.
  What is not established here is the particular quantitative spectral-quotient
  orbit bound/non-expansiveness route needed for this article's robust complexity
  instantiation. Failure of operator monotonicity of the exponential alone is
  not presented as a proof that the full quantum sweep is expansive.
- Clarified that putting `+infinity` on non-edges in a coupling matrix is not graph
  W1: it forbids those couplings rather than routing them. Vanilla Sinkhorn for
  graph W1 generally uses the dense geodesic matrix. Classical dense rounding
  results remain applicable but do not eliminate that bottleneck.

## Additional mathematical repairs found during revision

1. **Primal domain and existence.** The unregularized feasible set is closed and
   nonnegative, not restricted to strictly positive vectors. Strict feasibility
   and a finite attained LP optimum are explicit. Entropic and block positivity,
   coercivity, and multiplier existence are proved rather than assumed through
   opaque certificates.
2. **Initialization.** A preliminary second-block solve ensures the second
   residual is zero even at iteration zero. The graph Gibbs pair already has
   this property. The rate is for `k>=1`; zero-gap/zero-operator cases are addressed.
3. **Quotient subtlety.** Independent block quotient infima do not produce a
   common small gauge. The primal helper now uses a pairing budget (or a genuine
   simultaneous quotient radius). A one-dimensional counterexample exposes the
   invalid inference; graph flow is safe because its second right-hand side is zero.
4. **Geometry constants.** The stacked graph operator norm is 2, not the
   first-block norm 1. A path proof gives `kappa1 <= D_hop`. Weighted diameter,
   minimum-hop diameter, and spanning-tree path length are distinguished.
5. **Comparison flow.** Explained the spanning-tree construction of a feasible
   comparison flow and separately the existence of a mass-bounded optimal flow
   via shortest-path routing. The proof no longer invokes an undefined `T`.
6. **Bias.** A general positive reference requires a mass/reference-dependent KL
   envelope, not an unrestricted `mass log(d)` shortcut.
7. **General Bregman rate.** Restored the conjugate constant in the dual, added
   domain/attainment hypotheses, and derived the factor
   `4 U_gamma^2 a^2 / (gamma eta_gamma k)`, consistent with the KL factor eight.
8. **No vector inverse monotonicity shortcut.** The non-expansiveness appendix
   uses an explicitly separable scalar-root criterion and paired-kernel
   translation law, not the false implication from a signed/nonnegative matrix
   pattern to a globally order-preserving inverse.

## Numerical reporting

The existing figure data were **not rerun or relabeled as a new experiment**.
Inspection of `benchmarks/run_benchmark.py` and `benchmarks/wot_benchmark.py`
shows that the convergence plots use `dual_l2_rel_vs_lp`: best-so-far relative
L2 error of centered dual potentials, with a minimum over sign conventions.
The previous manuscript incorrectly called this an L2 error on flows.

The main text and both captions now describe the actual statistic, including
its dependence on the chosen LP reference when the optimum is nonunique. It is
not the theorem's objective-gap guarantee. Historical caption gamma ranges are
retained rather than replaced by newer, different driver defaults. CPU/GPU
figures, devices, graph sizes, source-support sizes, single-cell construction,
flow-rendering convention, and the non-vanishing-regularization caution are
explained. No new downstream generalization or lineage-reconstruction claim is made.

## Bibliography

See `paper/audit/bibliography.md` for the full ledger and primary links, and
`paper/audit/bibliography_inventory.json` for all original records. All 91 old
records were inventoried and screened; the revised live bibliography has 33
cited works, each manually checked against a primary source. Unused/unresolved
records are not asserted to be fictitious or silently marked verified.

Material corrections include the unsubstantiated Censor--Rezac record, unreliable
Friedrichs entry, Crandall's name/title, the Schiebinger author list and original
article DOI, Chizat's published title/pagination, duplicate identities, and the
missing directly relevant quantum Sinkhorn literature. Crossref rate-limit errors
are recorded as failed lookups, not treated as evidence of hallucination.

## Lean: an important limitation, not a cosmetic change

The existing library builds, but the earlier claim that all paper results were
fully formalized is not justified. Some mapped endpoints assume the estimates
or certificates whose derivation is the substance of the corresponding proof.
A generated challenge/solution match does not independently establish fidelity
to the mathematical paper.

The article, root README, Lean README, and CI now state this scope accurately.
`lean/PAPER_COVERAGE.md` lists the revised proof obligations requiring work.
Historical aliases and audit records are preserved, not silently renamed to
stronger theorems. The optional current-numbering check intentionally reports
three alias mismatches after the graph statements were consolidated; even fixing
those names would not close the semantic proof gaps. The revised article is
**not** described as completely Lean-certified. No attempt to finish the entire
formalization is hidden inside this publication-preparation task.

## Validation

- Full multi-file pdfLaTeX/BibTeX build: **20 pages**, no undefined references or
  citations, no overfull boxes, and no bibliography warnings.
- All 33 live bibliography entries are cited; duplicate citation keys and missing
  local figure/input dependencies are checked.
- Source-only arXiv archive extracted into a fresh temporary directory and rebuilt
  successfully without the repository, datasets, Lean, or previous LaTeX auxiliaries.
- Full Lean library build: **passed (8,112 jobs)**. This is a build result, not a
  proof of manuscript coverage.
- Seven deterministic NumPy regression tests passed: unequal-mass Pinsker; bias
  envelope; normalized lift and exact ascent; primal/log-potential equivalence;
  graph order, translations, and variation non-expansiveness; reverse-edge/log
  bounds at a numerical optimum; and the independent-gauge counterexample.
- The structural source reader agrees with the compiled list of **23** labeled
  theorem-like statements. The legacy Lean numbering audit remains explicitly partial.
- PDF pages were rendered and inspected, including the roadmap, main constants,
  CPU/GPU panels, bibliography, appendix proofs, and multipage notation table.
