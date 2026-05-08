#!/usr/bin/env python3
"""Audit paper-facing certification endpoints.

Checks:
1) Every theorem/proposition/lemma/corollary label in neurips/paper.aux has both
   - a label-key alias and
   - a numbering alias in StatementMap.lean.
2) Both aliases resolve to the same target.
3) Target theorem/lemma exists in Lean sources.
4) Flag suspicious pass-through endpoint bodies:
   - direct `:= foo ...` (no `by`)
   - first proof step `exact foo ...`

This is a structural audit (synchronization and endpoint-form), not a semantic proof checker.
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


def expected_label_alias(info: LabelInfo) -> str:
    return f"{info.kind}_{info.canonical}"


def expected_number_alias(info: LabelInfo) -> str:
    if "." not in info.number:
        raise ValueError(f"Unexpected numbering format: {info.number} for {info.label}")
    major, minor = info.number.split(".", 1)
    return f"{info.kind}_{major}_{minor}"


def find_decl_block(target: str) -> Tuple[Path, int, List[str]] | None:
    name = target.split(".")[-1]
    pat = re.compile(rf"^(theorem|lemma)\s+{re.escape(name)}\b", re.M)
    for fp in LEAN_ROOT.rglob("*.lean"):
        txt = fp.read_text(encoding="utf-8")
        m = pat.search(txt)
        if not m:
            continue
        lines = txt.splitlines()
        start_idx = txt[: m.start()].count("\n")
        end_idx = start_idx + 1
        while end_idx < len(lines) and not re.match(r"^(theorem|lemma)\s+", lines[end_idx]):
            end_idx += 1
        return fp, start_idx + 1, lines[start_idx:end_idx]
    return None


def classify_block(lines: List[str]) -> str:
    if not lines:
        return "unknown"
    head = lines[0]
    if ":=" in head and ":= by" not in head:
        return "direct_forward"
    body = [ln.strip() for ln in lines[1:10] if ln.strip()]
    if body and body[0].startswith("exact "):
        return "by_exact_forward"
    return "structured"


def main() -> int:
    if not AUX.exists() or not STATEMENT_MAP.exists():
        print("ERROR: missing required files", file=sys.stderr)
        return 2

    labels = parse_aux_labels(AUX.read_text(encoding="utf-8"))
    aliases = parse_statement_map(STATEMENT_MAP.read_text(encoding="utf-8"))

    errors: List[str] = []
    warnings: List[str] = []

    print("Label | Num | Target | Location | EndpointForm")
    print("---|---|---|---|---")

    for info in labels:
        a_label = expected_label_alias(info)
        a_num = expected_number_alias(info)

        t_label = aliases.get(a_label)
        t_num = aliases.get(a_num)

        if t_label is None:
            errors.append(f"Missing label alias: {a_label} ({info.label})")
        if t_num is None:
            errors.append(f"Missing numbered alias: {a_num} ({info.label}={info.number})")

        if t_label is not None and t_num is not None and t_label != t_num:
            errors.append(f"Alias mismatch: {info.label}: {a_label}->{t_label}, {a_num}->{t_num}")

        target = t_label or t_num
        if target is None:
            print(f"{info.label} | {info.number} | MISSING | MISSING | MISSING")
            continue

        decl = find_decl_block(target)
        if decl is None:
            errors.append(f"Target not found in Lean sources: {target}")
            print(f"{info.label} | {info.number} | {target} | NOT_FOUND | missing")
            continue

        fp, line, block = decl
        kind = classify_block(block)
        if kind in {"direct_forward", "by_exact_forward"}:
            warnings.append(f"Potential thin endpoint form for {info.label}: {target} ({kind})")
        rel = fp.relative_to(REPO)
        print(f"{info.label} | {info.number} | {target} | {rel}:{line} | {kind}")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"- {w}")

    if errors:
        print("\nAUDIT FAILED:", file=sys.stderr)
        for e in errors:
            print(f"- {e}", file=sys.stderr)
        return 1

    print("\nAUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
