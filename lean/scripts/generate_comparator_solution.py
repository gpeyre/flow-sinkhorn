#!/usr/bin/env python3
"""Generate the Comparator solution module from the paper-facing statement map."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEAN_DIR = REPO / "lean"
AUDIT_DIR = LEAN_DIR / "audit"
CONFIG = AUDIT_DIR / "comparator-paper-config.template.json"
SOLUTION = LEAN_DIR / "FlowSinkhorn" / "Comparator" / "Solution.lean"


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
        '  IO.println "import FlowSinkhorn.KLProjection.StatementMap"',
        '  IO.println ""',
        '  IO.println "set_option linter.style.longLine false"',
        '  IO.println "set_option linter.unusedVariables false"',
        '  IO.println ""',
        '  IO.println "/-!"',
        '  IO.println "# Comparator solution module"',
        '  IO.println ""',
        '  IO.println "This file exposes the paper-facing theorem names required by"',
        '  IO.println "`leanprover/comparator` as actual theorem constants."',
        '  IO.println ""',
        '  IO.println "Important: this is only the untrusted solution side. The trusted challenge side"',
        '  IO.println "must be authored independently from the paper/blueprint statements; generating"',
        '  IO.println "it from this file would make the Comparator comparison circular."',
        '  IO.println "-/"',
        '  IO.println ""',
        "  for pair in comparatorAliases do",
        "    let aliasName := pair.fst",
        "    let n := pair.snd",
        "    let ci ← getConstInfo n",
        "    let fmt ← PrettyPrinter.ppExpr ci.type",
        "    let ty := Format.pretty fmt 100000",
        '    IO.println s!"theorem {aliasName} : {ty} := by"',
        '    IO.println s!"  exact FlowSinkhorn.KLProjection.StatementMap.{aliasName}"',
        '    IO.println ""',
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    theorem_names = json.loads(CONFIG.read_text(encoding="utf-8"))["theorem_names"]
    SOLUTION.parent.mkdir(parents=True, exist_ok=True)

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

    SOLUTION.write_text(proc.stdout, encoding="utf-8")
    print(f"wrote {SOLUTION.relative_to(LEAN_DIR)} with {len(theorem_names)} theorem entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
