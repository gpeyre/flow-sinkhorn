# Comparator Certification Plan and Status

Status date: 2026-06-01.

This document tracks the plan to use
`leanprover/comparator` for the paper-facing Lean statements.

## Goal

Use Comparator to independently check every theorem/proposition/lemma/corollary appearing in the
LaTeX paper:

- the solution theorem proves the same statement as a trusted challenge theorem;
- the solution theorem uses no axioms beyond the permitted list;
- the solution theorem is accepted by the Lean kernel, optionally also by Nanoda.

The current paper-facing statement list is generated from and cross-checked against:

- `neurips/paper.tex`
- `neurips/paper.aux`
- `lean/FlowSinkhorn/KLProjection/StatementMap.lean`

Generated files:

- `lean/audit/comparator-paper-manifest.json`
- `lean/audit/comparator-paper-config.template.json`
- `lean/audit/comparator-paper-review.json`
- `lean/audit/comparator-paper-review.md`
- `lean/audit/comparator-challenge-lock.json`
- `lean/FlowSinkhorn/Comparator/Challenge.lean`
- `lean/FlowSinkhorn/Comparator/Solution.lean`

Regenerate the manifest/config with:

```bash
cd lean
python3 scripts/generate_comparator_manifest.py
```

Regenerate the challenge candidate with:

```bash
cd lean
python3 scripts/generate_comparator_challenge.py
```

Freeze the reviewed challenge candidate with:

```bash
cd lean
python3 scripts/generate_comparator_challenge_lock.py
python3 scripts/check_comparator_challenge_lock.py
```

Regenerate the solution module with:

```bash
cd lean
python3 scripts/generate_comparator_solution.py
```

Regenerate the paper-to-challenge review dossier with:

```bash
cd lean
python3 scripts/generate_comparator_review.py
```

Check the local scaffold with:

```bash
cd lean
python3 scripts/check_comparator_scaffold.py
```

## Comparator Requirements

Comparator is not just another `lake build`.  Its intended hardened workflow requires:

- an existing Lean installation;
- `landrun` in `PATH` (newer Comparator commits also support `COMPARATOR_LANDRUN`);
- `lean4export` compatible with the target project's Lean version, here Lean `4.28.0`;
- optionally `nanoda_bin` and `enable_nanoda = true`;
- preferably a Linux/systemd environment for the fully hardened invocation.

The documented Comparator guarantee assumes a trusted `Challenge` module and an untrusted
`Solution` module.  Therefore, the challenge statements must be authored or generated from the
paper/blueprint, not copied from the solution proofs in a way that would make the comparison
circular.

Trust-boundary status as of 2026-06-01: the current `FlowSinkhorn.Comparator.Challenge` module is
locally hardened against direct leakage of the canonical implementation endpoints listed in the
Comparator manifest.  It imports only Mathlib and the canonical proof-free statement vocabulary
under `FlowSinkhorn.Comparator.Vocabulary.*`, and `lean/audit/comparator-challenge-lock.json` freezes the
audit status date, exact reviewed imports, theorem order, paper-facing comments, and statement
hashes.  Therefore a regenerated or edited challenge cannot silently replace the reviewed artifact:
`scripts/check_comparator_scaffold.py` fails unless the current challenge matches the lock.  This
is still not a substitute for a final Linux Comparator certificate with real `landrun` and optional
Nanoda.  The independent semantic audit now records all `27/27` paper-facing statements as
faithful without qualification; the remaining caveat is external Comparator execution in a hardened
Linux sandbox, not a paper-to-challenge statement mismatch.

The hardened split is now implemented as a trusted statement-language layer plus an untrusted proof
layer:

- `Challenge` may import Mathlib and `FlowSinkhorn.Comparator.Vocabulary.*` only.
- `lean/FlowSinkhorn/Comparator/Vocabulary.lean` is the human-facing umbrella for the trusted
  statement vocabulary.
- `lean/FlowSinkhorn/Comparator/Vocabulary/` contains the shared definitions, structures, and
  predicates needed to state the paper theorems, but no paper theorem proof endpoints.
- Existing `FlowSinkhorn.KLProjection.*Vocabulary` files are import-only compatibility shims into
  the canonical Comparator vocabulary, so the statement language is not duplicated.
- `Solution` imports the proof layer through `FlowSinkhorn.KLProjection.StatementMap` and proves
  the same theorem names against the trusted challenge statements.
- A final trust-boundary check must confirm that importing `Challenge` alone does not expose any
  canonical implementation theorem listed in `StatementMap.lean`.

The current endpoint-leak check is machine-checkable:

```bash
cd lean
python3 scripts/check_comparator_trust_boundary.py
```

Expected current result after the latest import-hygiene pass: `TRUST BOUNDARY CHECK PASSED`.
Importing `FlowSinkhorn.Comparator.Challenge` exposes `0/27` canonical implementation endpoints
from `lean/audit/comparator-paper-manifest.json`.

## Current Local Environment

Observed on this machine:

- OS: macOS/Darwin ARM64.
- Project Lean toolchain: `leanprover/lean4:v4.28.0`.
- `comparator` was not initially installed on `PATH`.
- `landrun` was not installed on `PATH`.
- `lean4export` was not initially installed on `PATH`.
- `nanoda_bin` was not installed on `PATH`.

Action completed:

- Cloned `https://github.com/leanprover/comparator` to `/private/tmp/lean-comparator`.
- Built Comparator there with:

```bash
cd /private/tmp/lean-comparator
lake build lean4export comparator
```

Build result:

- Comparator binary built successfully at `/private/tmp/lean-comparator/.lake/build/bin/comparator`.
- lean4export binary built successfully at `/private/tmp/lean-comparator/.lake/build/bin/lean4export`.

The first build was useful for understanding deployment, but not compatible with this project:

- The cloned Comparator checkout currently uses Lean `4.31.0-rc1`.
- This project uses Lean `4.28.0`.
- Comparator's `lean4export` must be compatible with the target project's Lean version, so the
  built `/private/tmp/lean-comparator/.lake/build/bin/lean4export` should not yet be treated as a
  valid checker for this project.

Lean-4.28-compatible build now available:

- isolated checkout: `/private/tmp/lean-comparator-428`
- Comparator commit: `11443b9` (`chore: bump toolchain to v4.28.0`)
- lean4export revision: `048394e1afeeb52b0fa27bcf3f1ade2ff0f0ab6d`
- build command:

```bash
cd /private/tmp/lean-comparator-428
lake build lean4export comparator
```

Build result:

```text
Build completed successfully (23 jobs).
```

Remaining environment caveat:

- The fully hardened Comparator command depends on real `landrun`/Linux sandboxing.  This macOS
  host does not currently have real `landrun` on `PATH`.  The repository script therefore now
  refuses to run by default unless real `landrun` is available through `PATH` or
  `COMPARATOR_LANDRUN`.
- The development fake-landrun shim remains available only through the explicit opt-in
  `COMPARATOR_ALLOW_FAKE_LANDRUN=1`.  This is useful for testing wiring, but it does not provide
  the final sandbox guarantee.

## Paper Statement Manifest

The generated manifest currently contains `27` paper-facing entries.

The manifest generator now performs a direct `neurips/paper.tex` coverage check before writing
anything:

- every theorem/proposition/lemma/corollary environment in `neurips/paper.tex` must carry at least
  one paper-facing label;
- the ordered paper-facing labels extracted directly from `neurips/paper.tex` must match the
  ordered labels extracted from `neurips/paper.aux`;
- the resulting labels must be resolved through `StatementMap.lean` to canonical Lean theorem
  targets.

This guards against a stale `paper.aux` silently dropping or reordering paper statements.  The
scaffold checker repeats the direct `paper.tex` coverage check, so a stale or incomplete manifest
fails before Comparator is run.  The scaffold checker also rejects unresolved manifest entries:
`target` and implementation `source` must both be real values, never missing placeholders.

## Paper-to-Challenge Review Dossier

The Comparator standard has two separate layers:

- kernel/comparison layer: Comparator checks that `Solution` proves exactly the same theorem
  statements as `Challenge`, with only the configured axioms;
- trust layer: the `Challenge` statements themselves must be checked against the LaTeX paper and
  blueprint, because Comparator cannot infer that a trusted challenge theorem is the intended
  paper theorem.

To make the trust layer auditable, the repository now generates:

- `lean/audit/comparator-paper-review.json`
- `lean/audit/comparator-paper-review.md`
- `lean/audit/comparator-challenge-lock.json`

These files list all `27` paper theorem/proposition/lemma/corollary labels, the LaTeX source
location in `neurips/paper.tex`, the Comparator challenge theorem, and the canonical Lean
implementation target.  They also record SHA-256 fingerprints of:

- the compacted LaTeX statement excerpt;
- the compacted Comparator challenge theorem statement;
- the compacted Comparator solution theorem statement.

The scaffold checker requires this dossier to exist and to be synchronized with both the paper
manifest and the Comparator theorem list.  It also checks that every challenge theorem statement
matches the corresponding solution theorem statement textually before Comparator performs the
kernel/export check, and that the review entries are not stale relative to the manifest target,
source, and numbering metadata.

The lock file is deliberately separate from the review dossier: it is the frozen artifact
Comparator should consume.  It records the audit status date, the full `Challenge.lean` SHA-256
hash, and per-entry comment and theorem-statement hashes.  Regenerate it only after a fresh
paper-to-challenge review; otherwise the scaffold checker will correctly report that the challenge
has drifted.

The lock generator is now fail-closed: it refuses to write `comparator-challenge-lock.json` unless
the manifest, review dossier, independent audit labels, all-faithful verdicts, review statuses,
implementation locations, numbering metadata, audited Challenge/Solution import boundaries, and
challenge-vs-solution statement matches are all synchronized.

The standalone lock checker also cross-checks the lock against the independent audit ledger: every
locked entry must have the same paper label as the audit entry, the audit verdict must be
`faithful`, the locked review status must be `independent audit: faithful`, and the locked
challenge-vs-solution statement-match flag must be `true`.

The scaffold checker now also treats the generated Comparator modules as allowlisted artifacts:

- `Challenge.lean` must import exactly Mathlib plus the audited
  `FlowSinkhorn.Comparator.Vocabulary.*` modules and must not import `StatementMap.lean`,
  `Comparator/Solution.lean`, or implementation-side `FlowSinkhorn.KLProjection.*` modules;
- `Solution.lean` must import exactly `FlowSinkhorn.KLProjection.StatementMap`;
- every challenge proof must be exactly a statement-only `by`/`sorry` hole in paper order;
- every solution proof must be exactly the corresponding one-line `StatementMap` alias proof;
- the Comparator config theorem list, modules, and permitted axioms must match the generated
  manifest;
- the lock metadata must agree with the independent audit ledger, including all-faithful verdicts
  and `27/27` challenge-vs-solution statement matches.

Current status of the dossier:

- coverage is complete: `27/27` paper-facing statements are represented;
- the direct challenge-vs-solution statement match check passes for `27/27` entries;
- the independent challenge-vs-LaTeX audit is recorded in
  `lean/audit/comparator-challenge-audit.md`;
- the audit ledger now records `27` statements faithful without qualification, `0` qualified
  entries, and `0` mismatches;
- every entry records generated source fingerprints and the independent challenge-vs-LaTeX audit
  verdict copied from `lean/audit/comparator-challenge-audit.md`.

The last remediation pass removed the seven remaining paper-to-challenge qualifications by making
the LaTeX statements expose the same finite certificate or primitive-hypothesis interfaces that the
trusted Comparator challenge checks.  This affects Proposition 2.1, Theorem 3.1, Theorem 3.2,
Proposition 4.2, Proposition 5.1, Proposition 5.2, and Theorem 5.1.  In each case the challenge is
now faithful to the written paper statement rather than relying on an implicit bridge in the prose.

Important interpretation: this does not claim that a final external Comparator run has already been
performed.  It says the local dossier has no known statement-shape gap: the trusted challenge is
locked, the solution proves the same statements, the proof layer maps to concrete Lean endpoints,
and the independent paper-to-challenge audit finds no remaining non-faithful entry.

## Comparator Solution Module

The untrusted solution side has been started and is build-checked:

- module: `FlowSinkhorn.Comparator.Solution`
- file: `lean/FlowSinkhorn/Comparator/Solution.lean`
- theorem constants: the same `27` names listed in `audit/comparator-paper-config.template.json`
- proof source: each theorem is proved from the corresponding canonical implementation endpoint
  exposed by `FlowSinkhorn.KLProjection.StatementMap`

The scaffold checker enforces the intended solution shape: each solution theorem must be proved by
the corresponding one-line proof
`exact FlowSinkhorn.KLProjection.StatementMap.<paper_statement_name>`, in the same order as the
Comparator theorem list.  This keeps the solution side as a transparent alias layer rather than a
place where hidden paper-facing proof logic can drift.

Validation command:

```bash
cd lean
lake build FlowSinkhorn.Comparator.Solution
```

Observed result on 2026-05-29:

```text
Build completed successfully (8059 jobs).
```

This is deliberately only the solution side.  It does not replace Comparator's trusted challenge
side: a challenge generated from `StatementMap.lean` or from `Comparator/Solution.lean` would only
test that the current Lean constants are self-consistent, not that they match the paper statements.
The challenge module must therefore be authored independently from the paper/blueprint statement
text.

## Frozen Challenge Module

The challenge side now exists as a frozen statement-only module:

- module: `FlowSinkhorn.Comparator.Challenge`
- file: `lean/FlowSinkhorn/Comparator/Challenge.lean`
- theorem constants: the same `27` names listed in `audit/comparator-paper-config.template.json`
- proofs: exactly one `sorry` per statement, as expected for a Comparator challenge
- imports: exactly the audited allowlist of Mathlib plus proof-free `Vocabulary` modules
- lock file: `lean/audit/comparator-challenge-lock.json`

Important discipline:

- `Challenge.lean` may be regenerated as a candidate from the paper-facing statement map, but the
  reviewed artifact is the locked file, not the generator output by itself.
- Any challenge edit or regeneration must be followed by a fresh paper-to-challenge review and then
  `python3 scripts/generate_comparator_challenge_lock.py`.
- `python3 scripts/check_comparator_challenge_lock.py` and
  `python3 scripts/check_comparator_scaffold.py` enforce that the challenge still matches the
  frozen audit date, imports, comments, theorem order, and statement hashes.
- Final Comparator compliance still requires running Comparator with real `landrun` in a clean
  Linux environment.

Validation command:

```bash
cd lean
lake build FlowSinkhorn.Comparator.Challenge
```

Observed result on 2026-05-31:

```text
Build completed successfully (8072 jobs).
```

The expected output includes 27 Lean warnings of the form `declaration uses sorry`; these warnings
are normal for the trusted statement-only challenge side.

## Local Comparator Wiring Run

The local Comparator wiring run passes only with the explicit development shim:

- Comparator: `/private/tmp/lean-comparator-428/.lake/build/bin/comparator`
- lean4export: `/private/tmp/lean-comparator-428/.lake/packages/lean4export/.lake/build/bin/lean4export`
- landrun shim: `/private/tmp/lean-comparator/scripts/fake-landrun.sh`
- Lean toolchain: `leanprover/lean4:v4.28.0`
- config: `lean/audit/comparator-paper-config.template.json`

Reproducible command:

```bash
cd lean
COMPARATOR_ALLOW_FAKE_LANDRUN=1 scripts/run_comparator_bootstrap.sh
```

Observed terminal conclusion on 2026-05-31:

```text
Running Lean default kernel on solution.
Lean default kernel accepts the solution
Your solution is okay!
```

The same script now fails intentionally without real `landrun` or the explicit fake-landrun opt-in:

```bash
cd lean
scripts/run_comparator_bootstrap.sh
```

Expected conclusion on this macOS host:

```text
ERROR: real landrun was not found.
```

What the opt-in wiring run checks:

- `FlowSinkhorn.Comparator.Solution` proves the same 27 theorem statements as
  `FlowSinkhorn.Comparator.Challenge`.
- The solution export uses no axioms beyond:
  `propext`, `Quot.sound`, `Classical.choice`.
- The exported solution is accepted by the Lean default kernel.

What the opt-in wiring run does not check:

- It does not provide sandbox security, because fake-landrun is intentionally unsandboxed.
- It does not run Nanoda; `enable_nanoda` remains `false`.

Comparator template:

```json
{
  "challenge_module": "FlowSinkhorn.Comparator.Challenge",
  "solution_module": "FlowSinkhorn.Comparator.Solution",
  "theorem_names": [
    "prop_dual_gamma_correct",
    "thm_kl_dual_rate",
    "thm_approx_linprog",
    "prop_uniform_iter_final",
    "prop_mass_bound_block",
    "prop_graphw1_projection_closed_form",
    "prop_graphw1_flow_sinkhorn_update",
    "thm_graphw1_complexity",
    "lem_per_step_ascent",
    "lem_gap_vs_res_quotient",
    "prop_pinsker_normalized",
    "lem_pinsker_nonnormalized",
    "lem_kl_bias",
    "prop_hgamma_ot",
    "prop_kappa_ot",
    "cor_ot_xgamma_ugamma",
    "prop_graphw1_v1v2_closed_form",
    "prop_graphw1_signed_structure",
    "prop_graphw1_psi2_closed_nonexp",
    "prop_hgamma_graphw1",
    "lem_l1_bound_from_feasible",
    "prop_kappa_graph_diameter",
    "cor_graphw1_xgamma_ugamma",
    "prop_topical_nonexpansive",
    "prop_block_monotone",
    "lem_moment_monotone",
    "prop_translation_equivariance"
  ],
  "permitted_axioms": ["propext", "Quot.sound", "Classical.choice"],
  "enable_nanoda": false
}
```

## Deployment Plan

1. Freeze the trusted challenge source.
   The `Challenge` module must contain the paper statements with `sorry` proofs.  It must import
   only Mathlib and canonical `FlowSinkhorn.Comparator.Vocabulary.*` definitions, and it must not
   import the solution theorem map in a way that makes the statement comparison circular.

2. Build the solution module.
   The `Solution` module already exposes the same `27` theorem names and proves each one from the
   current canonical Lean endpoint in `StatementMap.lean`.  Re-run
   `python3 scripts/generate_comparator_solution.py` if the paper-facing alias list changes.

3. Run Comparator in a clean Linux environment.
   Use a Lean `4.28.0` compatible Comparator/lean4export build and real `landrun`:

```bash
cd lean
PATH="/path/to/dir-containing-landrun-and-lean4export:$PATH" \
  /path/to/lean-4.28.0/bin/lake env \
  /path/to/comparator-lean-4.28.0 \
  audit/comparator-paper-config.template.json
```

The local Lean-4.28 Comparator commit used here predates the later environment-variable interface
for `COMPARATOR_LANDRUN` and `COMPARATOR_LEAN4EXPORT`, so the robust invocation is to put
`landrun` and `lean4export` on `PATH`.

4. Optional stronger check.
   Install/build Nanoda and set:

```json
"enable_nanoda": true
```

5. Archive results.
   Store the exact Comparator binary version, Lean toolchain, command line, config file, and full
   output in this document.

## Current Status

- Paper-label synchronization: passed via `python3 scripts/check_statementmap_sync.py`.
- Direct `neurips/paper.tex` statement-label coverage: passed inside
  `python3 scripts/generate_comparator_manifest.py` and
  `python3 scripts/check_comparator_scaffold.py`.
- Manifest target resolution: passed; all `27/27` entries resolve to concrete Lean declaration
  targets and source locations.
- Structural internalization audit: passed via `python3 scripts/audit_paper_certification.py`, with
  `27/27` Tier-3 candidates, `0` open structural review flags, and `0` mapped `_of_assumption`
  endpoints.
- Paper-to-challenge semantic audit: passed with `27` faithful entries, `0` qualified entries, and
  `0` mismatches.
- Frozen challenge lock: generated via `python3 scripts/generate_comparator_challenge_lock.py` and
  checked via `python3 scripts/check_comparator_challenge_lock.py`; the lock pins the audit status
  date, full challenge file hash, imports, theorem order, comments, and theorem-statement hashes.
- Comparator scaffold check: passed via `python3 scripts/check_comparator_scaffold.py`.
- Comparator trust-boundary check: passed via
  `python3 scripts/check_comparator_trust_boundary.py`; importing
  `FlowSinkhorn.Comparator.Challenge` exposes `0/27` canonical implementation endpoints from
  `lean/audit/comparator-paper-manifest.json`.
- Lean build check: `lake build FlowSinkhorn.Comparator.Challenge FlowSinkhorn.Comparator.Solution
  FlowSinkhorn.Paper` completed successfully.  The `Challenge` module intentionally contains `sorry`
  placeholders, as required by the Comparator workflow.
- Local Comparator wiring run with explicit `COMPARATOR_ALLOW_FAKE_LANDRUN=1`: passed for
  all `27` paper-facing theorem names.
- Full hardened Comparator run on Linux with real `landrun`: not yet complete.

Latest faithful-statement pass:

- `prop:dual-gamma-correct` now states the finite primal-dual certificate used by Lean and proves
  the optimizer, value, stationarity, reconstruction, and dual-formula conclusions from it.
- `thm:kl-dual-rate` now states the scalar gap/residual/ascent interface from which Lean proves the
  Section-3 reciprocal-rate bound.
- `thm:approx-linprog` now states the finite approximation certificate combining the rate input,
  KL-bias certificate, displayed-dual identity, and closed-form iteration threshold.
- `prop:mass-bound-block` now states the finite quotient-radius, displayed-pairing-ascent, and
  zero-start mass predicates from which Lean proves the mass estimate with exact finite constants.
- `prop:graphw1-projection-closed-form` now states the C1/C2 algebra together with the finite
  nonnegative-flow variational certificate checked by Lean.
- `prop:graphw1-flow-sinkhorn-update` now states the two pointwise block identities from which Lean
  derives the displayed stable dual update.
- `thm:graphw1-complexity` now states the explicit operation-budget hypotheses from which Lean
  proves the final operation bound.

The trusted statement vocabulary remains separated from the proof layer:

- `FlowSinkhorn.Comparator.Challenge` imports only Mathlib plus audited
  `FlowSinkhorn.Comparator.Vocabulary.*` modules.
- `lean/FlowSinkhorn/Comparator/Vocabulary.lean` is the human-facing umbrella for the trusted
  proof-free statement language.
- The old `FlowSinkhorn.KLProjection.*Vocabulary` files are import-only compatibility shims into the
  canonical Comparator vocabulary.
- `FlowSinkhorn.Comparator.Solution` imports `FlowSinkhorn.KLProjection.StatementMap` and proves the
  same theorem names against the trusted challenge statements.

## Next Concrete Tasks

1. Treat `lean/audit/comparator-challenge-audit.md` as the semantic ledger and
   `lean/audit/comparator-paper-review.md` as the statement-by-statement source surface.
2. If any paper statement or `StatementMap.lean` alias changes, regenerate the manifest, challenge,
   solution, review, and lock, then rerun the scaffold, trust-boundary, lock, sync, and Lean build
   checks.
3. Run the same Comparator config on Linux with real `landrun` instead of fake-landrun.
4. Optionally enable Nanoda and archive that stronger run as well.
