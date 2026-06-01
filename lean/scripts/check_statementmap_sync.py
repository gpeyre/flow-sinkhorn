#!/usr/bin/env python3
"""Check sync between neurips paper labels/numbering and Lean StatementMap aliases.

Checks:
1. Every theorem/proposition/lemma/corollary label in neurips/paper.aux has a label-key alias
   in StatementMap.lean.
2. Every such label has a numbered alias consistent with the compiled numbering (e.g. 4.2 -> prop_4_2).
3. The label-key alias and numbered alias point to the same Lean target constant.
4. Prints target source file for each label alias.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[2]
AUX = REPO / "neurips" / "paper.aux"
STATEMENT_MAP = REPO / "lean" / "FlowSinkhorn" / "KLProjection" / "StatementMap.lean"
LEAN_ROOT = REPO / "lean" / "FlowSinkhorn" / "KLProjection"
PAPER_ROOT = REPO / "lean" / "FlowSinkhorn" / "Paper"

LABEL_PREFIXES = ("prop:", "lem:", "thm:", "cor:", "app-prop:", "app-lem:", "app-cor:")


@dataclass(frozen=True)
class LabelInfo:
    label: str
    number: str
    kind: str
    canonical: str


def parse_aux_labels(aux_text: str) -> List[LabelInfo]:
    out: List[LabelInfo] = []
    for line in aux_text.splitlines():
        m = re.search(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", line)
        if not m:
            continue
        label, num = m.group(1), m.group(2)
        if not label.startswith(LABEL_PREFIXES):
            continue
        base = label[4:] if label.startswith("app-") else label
        kind, rest = base.split(":", 1)
        out.append(LabelInfo(label=label, number=num, kind=kind, canonical=rest.replace("-", "_")))
    return out


def parse_statement_map(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for m in re.finditer(r"^abbrev\s+([A-Za-z0-9_]+)\s*:=\s*@([A-Za-z0-9_.]+)", text, re.M):
        out[m.group(1)] = m.group(2)
    return out


def parse_paper_facade_aliases() -> Dict[str, Tuple[str, Path]]:
    out: Dict[str, Tuple[str, Path]] = {}
    if not PAPER_ROOT.exists():
        return out
    for fp in PAPER_ROOT.glob("*.lean"):
        if fp.name == "StatementMap.lean":
            continue
        for alias, target in parse_statement_map(fp.read_text(encoding="utf-8")).items():
            out[alias] = (target, fp)
    return out


def expected_label_alias(info: LabelInfo) -> str:
    return f"{info.kind}_{info.canonical}"


def expected_number_alias(info: LabelInfo) -> str:
    if "." not in info.number:
        raise ValueError(f"Unexpected numbering format: {info.number} for {info.label}")
    major, minor = info.number.split(".", 1)
    return f"{info.kind}_{major}_{minor}"


def resolve_target_file(target: str) -> str:
    name = target.split(".")[-1]
    pat = re.compile(rf"^(theorem|lemma|abbrev)\s+{re.escape(name)}\b", re.M)
    for fp in LEAN_ROOT.rglob("*.lean"):
        txt = fp.read_text(encoding="utf-8")
        m = pat.search(txt)
        if m:
            line = txt[: m.start()].count("\n") + 1
            rel = fp.relative_to(REPO)
            return f"{rel}:{line}"
    return "NOT_FOUND"


def main() -> int:
    if not AUX.exists():
        print(f"ERROR: missing {AUX}", file=sys.stderr)
        print(
            "Hint: regenerate it with "
            "`cd neurips && pdflatex -interaction=nonstopmode -halt-on-error paper.tex`.",
            file=sys.stderr,
        )
        return 2
    if not STATEMENT_MAP.exists():
        print(f"ERROR: missing {STATEMENT_MAP}", file=sys.stderr)
        return 2

    labels = parse_aux_labels(AUX.read_text(encoding="utf-8"))
    aliases = parse_statement_map(STATEMENT_MAP.read_text(encoding="utf-8"))
    facade_aliases = parse_paper_facade_aliases()

    errors: List[str] = []

    print("Label | Num | LabelAlias | NumberAlias | Target | Lean file")
    print("---|---|---|---|---|---")

    for info in labels:
        a_label = expected_label_alias(info)
        a_num = expected_number_alias(info)

        t_label = aliases.get(a_label)
        t_num = aliases.get(a_num)

        if t_label is None:
            errors.append(f"Missing label alias: {a_label} (for {info.label})")
        if t_num is None:
            errors.append(f"Missing numbered alias: {a_num} (for {info.label} = {info.number})")

        if t_label is not None and t_num is not None and t_label != t_num:
            errors.append(
                f"Alias target mismatch for {info.label}: {a_label} -> {t_label}, {a_num} -> {t_num}"
            )

        target = t_label or t_num or "MISSING"
        src = resolve_target_file(target) if target != "MISSING" else "MISSING"
        print(f"{info.label} | {info.number} | {a_label} | {a_num} | {target} | {src}")

        for alias_name, expected_target in ((a_label, t_label), (a_num, t_num)):
            if expected_target is None:
                continue
            facade = facade_aliases.get(alias_name)
            if facade is None:
                continue
            facade_target, facade_file = facade
            if facade_target != expected_target:
                errors.append(
                    "Paper facade alias drift: "
                    f"{facade_file.relative_to(REPO)}::{alias_name} -> {facade_target}, "
                    f"canonical -> {expected_target}"
                )

    if errors:
        print("\nSYNC CHECK FAILED:", file=sys.stderr)
        for e in errors:
            print(f"- {e}", file=sys.stderr)
        return 1

    print("\nSYNC CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
