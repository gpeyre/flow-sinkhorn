Current theory-agent note for `/Users/gpeyre/Dropbox/github/flow-sinkhorn/draft/kl-projection-quantum/`.

Files currently in scope:
- `/Users/gpeyre/Dropbox/github/flow-sinkhorn/draft/kl-projection-quantum/kl-projection-quantum.tex`
- `/Users/gpeyre/Dropbox/github/flow-sinkhorn/draft/kl-projection-quantum/NOTES.md`

Current mathematical stance:
- The draft now treats the **exact variational entropic Q-OT problem** as the main object.
- The abstraction layer is the **general-Bregman extension** from `/Users/gpeyre/Dropbox/github/flow-sinkhorn/papers/kl-projections/sections/sec-dual-convergence.tex`, with
  - primal variable `T` on the positive definite cone,
  - Bregman generator `phi(T) = tr(T log T - T)`,
  - Bregman divergence `D_phi(T|S) = tr(T(log T-log S)-T+S)`.
- The exact block maps are the implicit dual maximizers
  - `Phi_1(G) = argmax_F Ftilde_gamma(F,G)`
  - `Phi_2(F) = argmax_G Ftilde_gamma(F,G)`
  and they are characterized by the exact marginal equations through partial traces of the matrix exponential.

Current conclusion on closed forms:
- In the genuinely noncommutative case, I still do **not** see a classical Sinkhorn-style closed form for the exact block maps.
- This is now stated as a secondary issue, not the main obstruction:
  the general-Bregman convergence route only needs exact block solves plus the right coercivity / boundedness / non-expansiveness controls.

Current candidate definitions:
- Primal confinement set:
  - `X_gamma(X) = { T >> 0 : tr(T) = X }`
  - in the normalized Q-OT setting, `X = 1`
- Primal norm:
  - Schatten-1 / nuclear norm `||T||_1`
- Generalized Pinsker constant:
  - on `tr(T)=tr(S)=X`, candidate
    - `D_phi(T|S) >= (1/(2X)) ||T-S||_1^2`
  - dimension-free, obtained by rescaling to density matrices and applying quantum Pinsker
- Quotient dual norm:
  - pair norm
    - `||(F,G)||_{V,op} = inf_lambda max( ||F+lambda I_n||_op, ||G-lambda I_m||_op )`
  - single-block prototype
    - `||F||_{var,op} = inf_lambda ||F+lambda I||_op = (lambda_max(F)-lambda_min(F))/2`
  - this is the operator analogue of the classical variation seminorm; the closest existing notion seems to be spectral spread / half spectral diameter
- Residuals:
  - `r_1(T) = Tr_2(T)-A`
  - `r_2(T) = Tr_1(T)-B`
  measured in nuclear norm
- Quantum `H_gamma` candidate:
  - `H_gamma^q = ||log T_gamma - log Z||_op`
  from the stationarity identity
  - `A^*(F_gamma,G_gamma) = C + gamma (log T_gamma - log Z)`
- Quantum `kappa` candidate:
  - decomposition constant for the map
    - `(F,G) -> F⊗I + I⊗G`
  with range norm `||.||_op` and block quotient operator norm
  - now proved, for the one-block blueprint constant:
    - `kappa_q <= 1`
    - by choosing the representative `F = (1/m) Tr_2(Y)` up to gauge

Main obstacles still open:
- Need a **spectral** bound on `H_gamma^q`; pointwise/entrywise lower bounds from the commutative paper are not the right geometry here.
- Need a precise bound for the fully coupled two-block quotient constant if we want a completely symmetric formulation; the one-block blueprint constant is now under control with `kappa_q <= 1`.
- The biggest missing piece is still the **non-expansiveness / dual boundedness** step:
  - Loewner monotonicity alone probably does not imply non-expansiveness for the quotient operator norm,
  - because Hermitian order is not a lattice and the scalar topical-map proof does not transfer.
- The new contractance pass sharpens this:
  - the raw exponential coupling map is now shown explicitly to fail Loewner monotonicity in a `2 x 2 x 2` example,
  - so any exact proof of sweep non-expansiveness must use something subtler than order preservation of `T(F,G)`.
- Likely replacements to investigate:
  - Thompson/Hilbert projective metrics on the positive definite cone,
  - order-preserving properties for exponentiated variables,
  - direct operator-norm cocoercivity / strong-concavity arguments for the implicit block resolvents.

New iteration on the contractance question:
- The draft now contains three concrete structural results:
  1. `Phi_1(G + lambda I_m) = Phi_1(G) - lambda I_n` and similarly for `Phi_2`, so the block maps and the sweeps descend to the quotient by scalar identities.
  2. A local Jacobian formula on a gauge slice:
     - `D Phi_1(G) = - M_G^{-1} N_G`
     where `M_G, N_G` are obtained by differentiating the partial-trace equation through the Fréchet derivative of the matrix exponential.
  3. A commuting-case positive result:
     - on a common diagonal subalgebra, the exact quantum block maps reduce exactly to classical Sinkhorn, hence the sweep is non-expansive in spectral variation.
- There is now also a fourth, genuinely useful local theorem:
  4. at a fixed point, the **full sweep** is locally non-expansive in the Hessian metric of the exact dual objective:
     - `D Psi_F = M_*^{-1} N_* P_*^{-1} N_*^*`
     - `D Psi_G = P_*^{-1} N_*^* M_*^{-1} N_*`
     - and Schur complement of the positive block Hessian yields contraction in the `M_*` / `P_*` norms
- This makes the logical situation much clearer:
  - in the commuting case, the classical blueprint survives;
  - in the genuinely noncommutative case, the first naive Loewner-monotonicity step already fails for `T(F,G)`;
  - the implicit correction through the two exact block solves does restore a local contraction structure for the **sweep**, but in a Hessian metric rather than yet in spectral variation.
- Numerical update on the spectral-variation question for the full sweep:
  - I ran a finite-difference search on random noncommutative `2 x 2` instances for the exact sweep;
  - across `382` successful tests with `gamma in {0.2, 0.4, 0.8, 1.5}`, the largest observed local spectral-variation Lipschitz constant of the sweep was about `0.866112 < 1`;
  - by contrast, the single block map `Phi_1` still shows local amplification around `3.371588 > 1` in another explicit instance.
- Blueprint-side progress:
  - the decomposition constant no longer looks like a blocker:
    - `kappa_q <= 1` is now in the draft;
  - the draft now contains a cleaner checklist theorem:
    - the transferred ingredients are
      - fixed-trace primal confinement,
      - generalized Pinsker with `eta_gamma = 1/(2 X_gamma)`,
      - quotient operator geometry,
      - partial-trace nuclear-norm contraction,
      - and `kappa_q <= 1`;
    - the only genuinely missing inputs are now isolated as
      - a quotient-norm sweep bound / non-expansiveness statement,
      - and a fixed-point bound (obtained either from `H_gamma^q` or by direct dual confinement)
  - there is also now a clean conditional rate theorem:
    - if the exact sweep is non-expansive in `||.||_{var,op}` and one has `||F_gamma||_{var,op} <= U_*`,
      then
      - `sup_k ||F^(k)||_{var,op} <= ||F^(0)||_{var,op} + 2 U_*`
      - `Delta_k <= (8 X_gamma U_max^2 / gamma) * 1/k`
    - with `U_* = ||C||_op + gamma H_gamma^q` as one possible way to close the bound
- Contractance-side progress:
  - under spectral confinement `a I <= T_* <= b I`, the draft now compares the local Hessian norms to `||.||_{var,op}`;
  - this gives an explicit but still crude local bound for the sweep in spectral variation,
    namely
    - `||D Psi_F||_{var,op -> var,op} <= sqrt(b n /(2 a))`
    - `||D Psi_G||_{var,op -> var,op} <= sqrt(b m /(2 a))`;
  - so we now know exactly where the loss is: the missing step is to sharpen that norm comparison enough to push the constant below `1`.
- New diagnosis about the diagonal case versus the noncommutative case:
  - in the diagonal/commuting setting, the right explanation is not the crude Hessian-vs-Hilbert-Schmidt route;
  - it is simply: Loewner monotonicity + translation equivariance => spectral-variation non-expansiveness.
- Better route now identified:
  - the draft now emphasizes that `H_gamma^q` is only one route to dual confinement:
    - any direct fixed-point or iterate confinement theorem in quotient operator norm would replace it without changing the rate proof
  - for unital linear maps, `||.||_{var,op}`-contraction is exactly a trace-distance contraction statement for the preadjoint;
  - so the local sweep question can be reformulated as a **quantum Dobrushin coefficient** problem for `D Psi_F` / `D Psi_G`.
- New concrete step in that direction:
  - the draft now defines the **local quantum Dobrushin coefficient**
    - `delta_q(F) = ||D Psi_F(F)||_{var,op -> var,op}`
  - and proves a local confinement/rate proposition:
    - if `delta_q(F) <= theta < 1` on a `||.||_{var,op}`-ball around a fixed point `F_*`,
      then the exact sweep is locally linearly convergent in spectral variation,
      the orbit is automatically dual-confined there,
      and the same `O(1/k)` dual-gap estimate follows on that local orbit
      **without any use of** `H_gamma^q`;
  - so the next exact-theory target is now very explicit:
    - prove a local bound `delta_q(F) <= theta < 1`,
      or at least identify a structural condition weaker than positivity that implies it.
- New focused clarification on what this target really means:
  - for a unital Hermiticity-preserving linear map `L`,
    `||L||_{var,op -> var,op}` is exactly the maximal trace-distance amplification on
    **differences of states**;
  - equivalently, it is enough to test the preadjoint on **orthogonal state pairs**:
    - `||L||_{var,op -> var,op} = (1/2) sup_{rho sigma = 0} ||L^*(rho)-L^*(sigma)||_1`;
  - so for the exact local sweep derivative, the proof target is now sharper than “find some monotonicity”:
    it is to prove direct trace-distance contraction of `(D Psi_F(F_*))^*` (or `(D Psi_G(G_*))^*`) on state differences, without needing positivity of the map itself.
- One notch sharper now:
  - the orthogonal-state supremum can itself be reduced to **orthogonal pure states**;
  - this uses only spectral decomposition inside the two orthogonal support spaces and the triangle inequality for the trace norm;
  - so the exact local proof target is now:
    - control
      `||(D Psi_F(F_*))^*(uu^*) - (D Psi_F(F_*))^*(vv^*)||_1`
      for unit vectors `u ⟂ v`,
    - and similarly on the `G` side.
- New equivalent formulation on the sweep side:
  - for a unital Hermiticity-preserving map `L`,
    `||L||_{var,op -> var,op}` is also
    the supremum over effects `0 <= E <= I` of the spectral diameter of `L(E)`:
    - `||L||_{var,op -> var,op} = sup_{0<=E<=I} (lambda_max(L(E)) - lambda_min(L(E)))`;
  - hence local non-expansiveness of the exact sweep follows if one can prove, for every effect `E`,
    - `diam_sp(M_*^{-1} N_* P_*^{-1} N_*^*(E)) <= 1`;
  - this is a concrete operator estimate on the explicit local sweep derivative, and is equivalent to the pure-state/trace-distance formulation via Helstrom duality.
- One more notch sharper:
  - the effect supremum can be reduced rigorously to a supremum over **orthogonal projections**,
    because spectral diameter is convex in the effect and the extreme points of the effect set are exactly projections;
  - so the exact local target is now:
    - `diam_sp(M_*^{-1} N_* P_*^{-1} N_*^*(P)) <= 1` for every projection `P`;
  - a further reduction to rank-one projections does **not** follow from this argument:
    extreme effects are all projections, and spectral diameter is convex rather than affine, so decomposing a projection into rank-one projectors does not preserve the maximization problem.
- New concrete sufficient route from the projection test:
  - if for every projection `P`, the local sweep derivative sends `P` into a scalar strip of width `1`,
    i.e. there exists `a_P` such that
    - `a_P I <= M_*^{-1} N_* P_*^{-1} N_*^*(P) <= a_P I + I`,
    then the exact local sweep is non-expansive in `||.||_{var,op}`;
  - strongest easy-to-state special case:
    - if `M_*^{-1} N_* P_*^{-1} N_*^*(P) >= 0` for every projection `P`,
      then unitality gives `0 <= ... <= I` on projections, hence contraction;
  - this is weaker than full positivity on all PSD operators, and it pinpoints an order-theoretic property of the explicit local derivative that would be enough.
- New local blueprint-closure package:
  - the draft now proves that if this projection-strip estimate holds **uniformly on a neighborhood of a fixed point** with width `theta < 1`, then:
    - `||D Psi_F(F)||_{var,op -> var,op} <= theta` throughout that neighborhood,
    - the exact sweep is locally linearly contractive in `||.||_{var,op}`,
    - the orbit is locally dual-confined,
    - and the local `O(1/k)` gap bound follows with no use of `H_gamma^q`;
  - so the remaining exact local hypothesis is now fully explicit:
    verify a neighborhood-wise strip estimate for
    `M_F^{-1} N_F P_F^{-1} N_F^*(P)` on projections.
- New obstruction / correction:
  - projection-positivity is **not** a genuinely weaker route than full positivity;
  - for a linear Hermiticity-preserving map, positivity on all projections, positivity on rank-one projectors, and positivity on the whole PSD cone are equivalent by spectral decomposition;
  - moreover, there are unital Hermiticity-preserving maps with real spectrum in `[0,1]` and self-adjointness in some weighted inner product that are still not positive;
  - so the current Hessian-metric information on
    `L_F = M_F^{-1} N_F P_F^{-1} N_F^*`
    is not enough by itself to imply projection-positivity or projection-strip;
  - conclusion: the first genuinely weaker extra hypothesis remains the projection-strip estimate itself, not projection-positivity.
- New genuinely weaker sufficient condition:
  - if
    `L_F = K_F + eta_F(.) I`
    where `K_F` is positive and sub-unital in the sense `K_F(I) <= theta I`,
    then `L_F` automatically satisfies the projection-strip condition with width `theta`;
  - indeed, for every projection `P`,
    positivity gives `0 <= K_F(P) <= K_F(I) <= theta I`,
    hence
    `eta_F(P) I <= L_F(P) <= eta_F(P) I + theta I`;
  - this is genuinely weaker than positivity of `L_F` itself, because the scalar part is invisible to the quotient / spectral-variation seminorm;
  - so a promising exact local target is now:
    find a scalar-shifted positive decomposition of
    `M_F^{-1} N_F P_F^{-1} N_F^*`
    uniformly near the fixed point.
- New proof-status packaging:
  - the draft now contains a local theorem saying that, once
    1. local projection-strip control holds with width `theta < 1`, and
    2. a fixed-point bound `||F_*||_{var,op} <= U_*` is available,
    then all local blueprint hypotheses are checked with explicit constants:
    - `eta_gamma = 1/(2 X_gamma)`
    - `kappa_q <= 1`
    - `||Tr_1||_{1->1}, ||Tr_2||_{1->1} <= 1`
    - local sweep Lipschitz constant `<= theta`
    - local orbit bound `U_max = U_* + r`
    - rate constant `8 X_gamma U_max^2 / gamma`
  - so the current exact local frontier is now very clean:
    every constant is explicit, and the remaining nontrivial input is exactly
    projection-strip control plus a usable `U_*` bound.
- New fixed-point-side advance:
  - a spectral lower bound `alpha_gamma I <= T_gamma` now gives an explicit bound on
    `H_gamma^q`:
    - `H_gamma^q <= max(-log alpha_gamma, log X_gamma) + ||log Z||_op`;
  - via stationarity this yields an explicit fixed-point bound
    - `U_* <= ||C||_op + gamma ( max(-log alpha_gamma, log X_gamma) + ||log Z||_op )`;
  - so the fixed-point side of the blueprint is now reduced to one concrete spectral input:
    a lower bound on `lambda_min(T_gamma)`.
- Sharpened synthesis:
  - the draft now states explicitly that only two analytic inputs remain:
    1. local projection-strip / non-expansiveness control for the exact sweep derivative;
    2. either a direct bound on `||F_*||_{var,op}` or a spectral lower bound on `T_gamma`;
  - all other blueprint constants are already checked explicitly:
    - `eta_gamma = 1/(2 X_gamma)`
    - `kappa_q <= 1`
    - `||Tr_1||_{1->1}, ||Tr_2||_{1->1} <= 1`
    - local rate constant `8 X_gamma (U_* + r)^2 / gamma` once the two open inputs are supplied.
- New comparison with the classical Sinkhorn proof route:
  - the draft now records that classical Sinkhorn admits a fully direct differential proof of non-expansiveness, not only the topical/monotone-map proof from the main paper;
  - differentiating the soft-`c` transforms shows that the classical sweep derivative is a row-stochastic matrix, hence a Markov operator with Dobrushin coefficient at most `1`;
  - this is the precise local mechanism behind classical variation-seminorm contraction.
- Consequence for the quantum case:
  - the exact analogue would be that the local sweep derivative (or its preadjoint) should be a positive trace-preserving map on states;
  - but the existing noncommutative obstruction already shows a PSD direction `H >= 0` with `D Psi_F(F_*)[H] \not>= 0`;
  - so the exact quantum sweep derivative is generally **not positive**, and the classical Markov-kernel / stochastic-matrix proof does not transfer verbatim;
  - what remains as the right target is therefore a contraction estimate for the quantum Dobrushin coefficient of a generally nonpositive unital map.
- Serious new obstruction:
  - if the exact sweep were locally Loewner-monotone, its derivative would have to map PSD directions to PSD directions;
  - but a dedicated `2x2` fixed-point search found a PSD direction for which the finite-difference sweep derivative has a negative eigenvalue about `-1.57e-2`;
  - so plain local Loewner monotonicity of the exact sweep seems to fail in the genuinely noncommutative case.
- I am **not** claiming a full counterexample to exact-sweep non-expansiveness yet.
- New global proof-status packaging:
  - the draft now also contains a global theorem saying that, once
    1. the exact sweep is non-expansive in `||.||_{var,op}`, and
    2. `H_gamma^q < +infty`,
    then every constant needed by the abstract general-Bregman blueprint is explicit:
    - `eta_gamma = 1/(2 X_gamma)`
    - `kappa_q <= 1`
    - `||Tr_1||_{1->1}, ||Tr_2||_{1->1} <= 1`
    - `U_* = ||C||_op + gamma H_gamma^q`
    - `U_max = ||F^(0)||_{var,op} + 2 U_*`
    - rate constant `8 X_gamma U_max^2 / gamma`
  - so globally, just as locally, the exact blueprint is now reduced to a very short list of genuinely missing analytic inputs.
- New structured local closure route:
  - beyond the projection-strip criterion, the draft now has a corollary saying that local blueprint closure follows as soon as the derivative admits a decomposition
    `D Psi_F(F) = K_F + eta_F(.) I`
    with `K_F` positive on the PSD cone and `K_F(I) = theta_F I <= theta I`;
  - this is useful because it gives a more algebraic target than checking scalar-strip bounds projection by projection.
- Exposition reshaped:
  - the draft now reads in the order
    1. exact variational setting,
    2. exact dual / block maps / sweep formalism,
    3. blueprint checklist and explicit constants,
    4. conditional closure theorems,
    5. non-expansiveness questions and obstructions,
    6. final status / remaining open inputs;
  - this makes the blueprint perspective visible much earlier, before the long operator-theoretic discussion.
- New concrete global route:
  - the draft now contains a theorem saying that if
    1. the exact sweep derivative admits a global shifted-positive decomposition
       `D Psi_F(F) = K_F + eta_F(.) I`
       with `K_F` positive and `K_F(I) = theta_F I <= I`, and
    2. the optimizer satisfies a spectral lower bound `alpha_gamma I <= T_gamma`,
    then the exact sweep is globally non-expansive and the whole abstract blueprint closes with fully explicit constants;
  - if in addition `sup_F theta_F <= theta < 1`, the theorem also gives a global linear contraction estimate toward any fixed point.
- New spectral bridge from dual to primal:
  - the draft now contains a proposition showing that a quotient dual bound
    `||(F,G)||_{V,op} <= U`
    implies the explicit lower spectral bound
    `T(F,G) >= lambda_min(Z) exp(-( ||C||_op + 2U )/gamma) I`;
  - this gives the exact quantum analogue of “primal bound from a dual bound”;
  - combined with the existing `alpha_gamma -> H_gamma^q -> ||F_gamma||_{var,op}` route, the fixed-point side of the blueprint can now be attacked from either direction.
- Blueprint fit made more explicit:
  - the draft now contains a short “dictionary” proposition mapping the abstract general-Bregman objects directly to the Q-OT ones:
    primal cone, Legendre generator, affine blocks, shifted reference, exact block maps, and the primal/dual norms;
  - the checklist/status sections now explicitly say that the variational side of the blueprint is fully specialized, and that the remaining gap is only the orbit-control / non-expansiveness side.
- Stronger pair-side result:
  - the draft now proves a genuine two-block decomposition constant `kappa_q^pair <= 1`;
  - this yields an explicit pair quotient bound from stationarity:
    `||(F_gamma,G_gamma)||_{V,op} <= ||C||_op + gamma H_gamma^q`;
  - and, more importantly, a new global theorem shows that if a pair quotient bound on an exact maximizer is available, then the full blueprint closes with explicit constants, because the pair bound now implies the needed spectral lower bound on `T_gamma`.
- Stronger `H_gamma^q` consequences:
  - the draft now also derives an explicit lower spectral bound on the optimizer directly from `H_gamma^q`:
    `T_gamma >= lambda_min(Z) exp( -3 ||C||_op / gamma - 2 H_gamma^q ) I`;
  - so the global proof-status theorem now carries not only one-block constants, but also:
    - the pair quotient bound,
    - the pair decomposition constant,
    - and the induced explicit lower spectral bound on `T_gamma`.
- The current rigorous output is instead:
  - an exact quotient/sweep formulation,
  - an exact implicit Jacobian formula,
  - a local Hessian-metric contraction theorem for the sweep,
  - a spectral-confinement-to-quotient comparison estimate,
  - a precise explanation of why the diagonal case makes the crude norm-comparison route look suspicious,
  - a local obstruction to the monotone-map route in the noncommutative case,
  - a reformulation of the local contraction problem as a quantum Dobrushin / trace-distance contraction question,
  - a commuting-case theorem,
  - an explicit noncommutative obstacle statement,
  - the bound `kappa_q <= 1`,
  - a conditional blueprint corollary,
  - plus encouraging numerical evidence that the **full sweep** may still be non-expansive in spectral variation.

Short next actions:
1. Study the quantum Dobrushin coefficient of the local exact sweep derivative directly through projection-strip or shifted-positive estimates.
2. Prove a spectral lower bound on `lambda_min(T_gamma)` strong enough to make `H_gamma^q` explicit.
3. Globalize the local contraction/confinement theorem.
4. Look for a direct dual-confinement theorem that bypasses `H_gamma^q` entirely.
