#!/usr/bin/env python3
"""Generate the repository-side Comparator challenge candidate module.

The generated challenge imports only the audited statement-vocabulary layer.
Its theorem statements are printed from the current `StatementMap` endpoints so
that their Lean shapes stay synchronized with the paper-facing aliases.  This is
only the bootstrap source for the trusted statement file: after independent
paper-to-challenge review, `comparator-challenge-lock.json` freezes the exact
comments and theorem statements that Comparator should consume.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEAN_DIR = REPO / "lean"
AUDIT_DIR = LEAN_DIR / "audit"
CONFIG = AUDIT_DIR / "comparator-paper-config.template.json"
MANIFEST = AUDIT_DIR / "comparator-paper-manifest.json"
AUDIT = AUDIT_DIR / "comparator-challenge-audit.json"
CHALLENGE = LEAN_DIR / "FlowSinkhorn" / "Comparator" / "Challenge.lean"

CHALLENGE_IMPORTS = [
    "Mathlib",
    "FlowSinkhorn.Comparator.Vocabulary.Legacy.Section2Duality",
    "FlowSinkhorn.Comparator.Vocabulary.UniformBound",
    "FlowSinkhorn.Comparator.Vocabulary.Topical",
    "FlowSinkhorn.Comparator.Vocabulary.BlockQuotient",
    "FlowSinkhorn.Comparator.Vocabulary.Setup.VariationGeometry",
    "FlowSinkhorn.Comparator.Vocabulary.Sweep",
    "FlowSinkhorn.Comparator.Vocabulary.PrimalDualBounds",
    "FlowSinkhorn.Comparator.Vocabulary.DualConvergence",
    "FlowSinkhorn.Comparator.Vocabulary.Applications.OT.HGamma",
    "FlowSinkhorn.Comparator.Vocabulary.Applications.OT.Complexity",
    "FlowSinkhorn.Comparator.Vocabulary.Applications.GraphW1.ClosedForms",
    "FlowSinkhorn.Comparator.Vocabulary.Applications.GraphW1.HGamma",
    "FlowSinkhorn.Comparator.Vocabulary.Applications.GraphW1.Complexity",
]


def compact(text: object, max_chars: int | None = None) -> str:
    out = " ".join(str(text or "").split())
    out = out.replace("-/", "- /")
    if max_chars is None or len(out) <= max_chars:
        return out
    return out[: max_chars - 3].rstrip() + "..."


def comment_lines(text: str, width: int = 94) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append(" *")
            continue
        wrapped = textwrap.wrap(
            paragraph,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.extend(f" * {line}" for line in wrapped)
    return lines


def load_challenge_comments(theorem_names: list[str]) -> dict[str, str]:
    if not MANIFEST.exists():
        return {
            name: (
                "Paper statement: metadata unavailable; run "
                "`python3 scripts/generate_comparator_manifest.py` first.\n"
                "Formalization intent: Comparator checks this bootstrap challenge statement "
                "against the solution theorem of the same name."
            )
            for name in theorem_names
        }

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_alias = {
        item["label_alias"]: item
        for item in manifest.get("paper_statements", [])
    }
    audit_by_label: dict[str, dict[str, object]] = {}
    if AUDIT.exists():
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        audit_by_label = {
            item["label"]: item
            for item in audit.get("entries", [])
            if "label" in item
        }

    comments: dict[str, str] = {}
    for name in theorem_names:
        item = by_alias.get(name)
        if item is None:
            comments[name] = (
                f"Paper statement: `{name}` is missing from the generated manifest.\n"
                "Formalization intent: Comparator checks this bootstrap challenge statement "
                "against the solution theorem of the same name."
            )
            continue

        audit_entry = audit_by_label.get(item["label"], {})
        verdict = audit_entry.get("verdict", "not audited")
        note = compact(audit_entry.get("note", ""))
        title = item.get("paper_title") or "(untitled)"
        comments[name] = "\n".join(
            [
                (
                    f"Paper statement: {item.get('paper_env_kind', 'statement')} "
                    f"{item.get('number')} `{item.get('label')}` ({title}), "
                    f"from {item.get('paper_source')}."
                ),
                (
                    f"Lean implementation: `{item.get('target')}` "
                    f"at `{item.get('source')}`."
                ),
                f"Audit verdict: `{verdict}`.",
                (
                    "Formalization intent: this statement-only challenge theorem is the "
                    "reviewed paper-facing statement checked against the untrusted solution "
                    "alias. Its current Lean type was mechanically bootstrapped from the "
                    "audited endpoint, then frozen by comparator-challenge-lock.json after "
                    "paper-to-challenge review. The cited implementation file contains the "
                    "actual proof, but Challenge imports only Mathlib and canonical "
                    "Comparator statement vocabulary; final certification requires "
                    "Comparator with a real landrun sandbox."
                ),
                f"Reviewer note: {note}" if note else "Reviewer note: no independent note recorded.",
            ]
        )
    return comments


def add_paper_comments(generated: str, theorem_names: list[str]) -> str:
    comments = load_challenge_comments(theorem_names)
    out: list[str] = []
    for line in generated.splitlines():
        inserted = False
        for name in theorem_names:
            if line.startswith(f"theorem {name} :"):
                out.append("/--")
                out.extend(comment_lines(comments[name]))
                out.append("-/")
                inserted = True
                break
        out.append(line)
    return "\n".join(out) + "\n"


def lean_generator_source(theorem_names: list[str]) -> str:
    lines = [
        "import FlowSinkhorn.KLProjection.StatementMap",
        "import Lean",
        "set_option pp.funBinderTypes true",
        "open Lean Meta",
        "def comparatorAliases : Array (String × Name) := #[",
    ]
    for name in theorem_names:
        lines.append(f'  ("{name}", `FlowSinkhorn.KLProjection.StatementMap.{name}),')
    lines += [
        "]",
        "",
        "#eval show MetaM Unit from do",
    ]
    for module in CHALLENGE_IMPORTS:
        lines.append(f'  IO.println "import {module}"')
    lines += [
        '  IO.println ""',
        '  IO.println "set_option linter.style.longLine false"',
        '  IO.println "set_option linter.style.docString false"',
        '  IO.println "set_option linter.unusedVariables false"',
        '  IO.println ""',
        '  IO.println "/-!"',
        '  IO.println "# Frozen Comparator challenge module"',
        '  IO.println ""',
        '  IO.println "This file contains statement-only theorem declarations for Comparator."',
        '  IO.println "The proofs intentionally use `sorry`, as expected for a Comparator challenge."',
        '  IO.println ""',
        '  IO.println "Important: this file was mechanically bootstrapped from the audited Lean"',
        '  IO.println "paper-facing statement map and is then frozen by"',
        '  IO.println "`lean/audit/comparator-challenge-lock.json` after independent paper-to-challenge review."',
        '  IO.println "Changing `StatementMap` or regenerating this file is not sufficient for trust;"',
        '  IO.println "the lock must be regenerated only after a fresh independent review."',
        '  IO.println ""',
        '  IO.println "Trust boundary: this challenge imports only Mathlib and the canonical"',
        '  IO.println "`FlowSinkhorn.Comparator.Vocabulary.*` statement-language layer, not"',
        '  IO.println "proof-bearing implementation modules. The paired solution imports the"',
        '  IO.println "paper-facing StatementMap aliases and supplies the actual proofs."',
        '  IO.println "-/"',
        '  IO.println ""',
        "  for pair in comparatorAliases do",
        "    let aliasName := pair.fst",
        "    let n := pair.snd",
        "    let ci ← getConstInfo n",
        "    let fmt ← PrettyPrinter.ppExpr ci.type",
        "    let ty := Format.pretty fmt 100000",
        '    IO.println s!"theorem {aliasName} : {ty} := by"',
        '    IO.println "  sorry"',
        '    IO.println ""',
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    theorem_names = json.loads(CONFIG.read_text(encoding="utf-8"))["theorem_names"]
    CHALLENGE.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["lake", "build", "FlowSinkhorn.KLProjection.StatementMap"],
        cwd=LEAN_DIR,
        check=True,
        text=True,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False, encoding="utf-8") as tmp:
        tmp.write(lean_generator_source(theorem_names))
        tmp_path = Path(tmp.name)

    try:
        proc = subprocess.run(
            ["lake", "env", "lean", str(tmp_path)],
            cwd=LEAN_DIR,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    CHALLENGE.write_text(add_paper_comments(proc.stdout, theorem_names), encoding="utf-8")
    print(f"wrote {CHALLENGE.relative_to(LEAN_DIR)} with {len(theorem_names)} theorem entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
