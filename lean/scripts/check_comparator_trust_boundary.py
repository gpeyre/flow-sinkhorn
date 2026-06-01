#!/usr/bin/env python3
"""Check whether the bootstrap Comparator challenge leaks proof endpoints.

For a hardened Comparator setup, importing `FlowSinkhorn.Comparator.Challenge`
should expose only trusted statement vocabulary, not the canonical theorem
implementations listed in `comparator-paper-manifest.json`.

This check should PASS for the current hardened challenge surface: the generated
challenge imports statement-vocabulary modules, while the solution imports the
paper-facing implementation map.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEAN_DIR = REPO / "lean"
AUDIT_DIR = LEAN_DIR / "audit"
MANIFEST = AUDIT_DIR / "comparator-paper-manifest.json"


def full_name_from_source(source: str, target: str) -> str | None:
    path_part = source.split(":", 1)[0]
    src = REPO / path_part
    if not src.exists():
        return None

    namespace_stack: list[str] = []
    decl_pat = re.compile(rf"^\s*(?:theorem|lemma|def|abbrev)\s+{re.escape(target)}\b")
    for line in src.read_text(encoding="utf-8").splitlines():
        ns = re.match(r"^\s*namespace\s+(.+?)\s*$", line)
        if ns:
            namespace_stack.extend(ns.group(1).split())
            continue

        if decl_pat.match(line):
            return ".".join(namespace_stack + [target])

        end = re.match(r"^\s*end(?:\s+(.+?))?\s*$", line)
        if end and namespace_stack:
            if end.group(1):
                parts = end.group(1).split()
                for _ in parts:
                    if namespace_stack:
                        namespace_stack.pop()
            else:
                namespace_stack.pop()

    return None


def visible_after_import(full_names: list[str]) -> set[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False, encoding="utf-8") as tmp:
        tmp.write("import FlowSinkhorn.Comparator.Challenge\n")
        for full_name in full_names:
            tmp.write(f"#check {full_name}\n")
        tmp_path = Path(tmp.name)
    try:
        proc = subprocess.run(
            ["lake", "env", "lean", str(tmp_path)],
            cwd=LEAN_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        unknown = set(re.findall(r"Unknown identifier `([^`]+)`", proc.stdout))
        return {name for name in full_names if name not in unknown}
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    full_names: list[tuple[str, str]] = []
    unresolved: list[tuple[str, str]] = []

    for item in manifest.get("paper_statements", []):
        target = item["target"]
        full = full_name_from_source(item["source"], target)
        if full is None:
            unresolved.append((item["label"], target))
            continue
        full_names.append((item["label"], full))

    if unresolved:
        print("Could not resolve implementation target names:")
        for label, target in unresolved:
            print(f"- {label}: {target}")
        return 2

    visible = visible_after_import([full for _, full in full_names])
    leaked = [(label, full) for label, full in full_names if full in visible]

    if leaked:
        print("TRUST BOUNDARY CHECK FAILED")
        print(f"Visible implementation endpoints: {len(leaked)}/{len(full_names)}")
        print("Importing FlowSinkhorn.Comparator.Challenge exposes proof endpoints:")
        for label, full in leaked:
            print(f"- {label}: {full}")
        print()
        print("This indicates that the challenge import surface is not hardened enough for Comparator.")
        return 1

    print("TRUST BOUNDARY CHECK PASSED")
    print("No manifest implementation theorem is visible after importing the Challenge module.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
