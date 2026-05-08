#!/usr/bin/env python3
"""Audit paper-facing certification endpoints.

This audit intentionally separates three notions:
1) Endpoint coverage: every compiled paper label has label-key and numbered aliases.
2) Endpoint structure: the mapped Lean target exists and is not an obvious thin forwarder.
3) Internalization risk: structural flags for endpoints whose paper-facing statement still packages
   substantial model-specific facts as hypotheses.

The script is a structural audit, not a semantic proof checker.  A clean run means the paper labels
are covered by build-checked Lean endpoints; it does not, by itself, certify that every manuscript
intermediate derivation has been fully internalized from first principles.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO = Path(__file__).resolve().parents[2]
AUX = REPO / "neurips" / "paper.aux"
STATEMENT_MAP = REPO / "lean" / "FlowSinkhorn" / "KLProjection" / "StatementMap.lean"
LEAN_ROOT = REPO / "lean" / "FlowSinkhorn" / "KLProjection"
GAP_LEDGER = REPO / "lean" / "AUDIT_INTERNALIZATION_GAPS.md"

LABEL_PREFIXES = ("prop:", "lem:", "thm:", "cor:", "app-prop:", "app-lem:", "app-cor:")
ASSUMPTION_TARGET_SUFFIXES = ("_of_assumption", "_of_assumptions")
ASSUMPTION_HEAVY_THRESHOLD = 4
ASSUMPTION_SIGNAL_TOKENS = (
    "≤",
    "<",
    "=",
    "∀",
    "∃",
    "Monotone",
    "Antitone",
    "SeminormNonexpansive",
    "IsTopical",
    "hGammaKappaBudget",
)


@dataclass(frozen=True)
class LabelInfo:
    label: str
    number: str
    kind: str
    canonical: str


@dataclass(frozen=True)
class DeclInfo:
    path: Path
    line: int
    block: List[str]


@dataclass(frozen=True)
class EndpointAudit:
    label: LabelInfo
    target: str | None
    decl: DeclInfo | None
    endpoint_form: str
    hypothesis_count: int
    flags: Tuple[str, ...]


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
    # The target may be wrapped onto the next line for long paper aliases.
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


def find_decl_block(target: str) -> DeclInfo | None:
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
        return DeclInfo(fp, start_idx + 1, lines[start_idx:end_idx])
    return None


def signature_lines(lines: List[str]) -> List[str]:
    sig: List[str] = []
    for ln in lines:
        if ":=" in ln:
            sig.append(ln.split(":=", 1)[0])
            break
        sig.append(ln)
    return sig


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


def count_assumption_like_hypotheses(lines: List[str]) -> int:
    sig = "\n".join(signature_lines(lines))
    count = 0
    for group in parenthesized_groups(sig):
        if any(tok in group for tok in ASSUMPTION_SIGNAL_TOKENS):
            count += 1
    return count


def parenthesized_groups(text: str) -> List[str]:
    """Return top-level parenthesized binder groups from a Lean signature."""
    groups: List[str] = []
    depth = 0
    start = -1
    for idx, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                groups.append(text[start : idx + 1])
                start = -1
    return groups


def endpoint_flags(label_alias: str, target: str, endpoint_form: str, hypothesis_count: int) -> Tuple[str, ...]:
    flags: List[str] = []
    target_base = target.split(".")[-1]
    if target_base.endswith(ASSUMPTION_TARGET_SUFFIXES):
        flags.append("assumption-target")
    if label_alias.endswith(ASSUMPTION_TARGET_SUFFIXES):
        flags.append("assumption-alias")
    if endpoint_form in {"direct_forward", "by_exact_forward"}:
        flags.append("thin-forwarder")
    if hypothesis_count >= ASSUMPTION_HEAVY_THRESHOLD:
        flags.append(f"assumption-heavy:{hypothesis_count}")
    if target_base.endswith("_as_stated"):
        flags.append("paper-as-stated-wrapper")
    return tuple(flags)


def tier2_status(endpoint_form: str) -> str:
    return "WARN" if endpoint_form in {"direct_forward", "by_exact_forward"} else "PASS"


def tier3_status(flags: Iterable[str]) -> str:
    flag_list = list(flags)
    if not flag_list:
        return "candidate"
    if any(f in {"assumption-target", "assumption-alias"} for f in flag_list):
        return "gap"
    if any(f.startswith("assumption-heavy") for f in flag_list):
        return "needs-review"
    return "needs-review"


def audit_labels(labels: List[LabelInfo], aliases: Dict[str, str]) -> Tuple[List[EndpointAudit], List[str], List[str]]:
    errors: List[str] = []
    sync_warnings: List[str] = []
    audits: List[EndpointAudit] = []

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
            audits.append(EndpointAudit(info, None, None, "missing", 0, ("missing",)))
            continue

        decl = find_decl_block(target)
        if decl is None:
            errors.append(f"Target not found in Lean sources: {target}")
            audits.append(EndpointAudit(info, target, None, "missing", 0, ("missing-target",)))
            continue

        endpoint_form = classify_block(decl.block)
        hypothesis_count = count_assumption_like_hypotheses(decl.block)
        flags = endpoint_flags(a_label, target, endpoint_form, hypothesis_count)
        if endpoint_form in {"direct_forward", "by_exact_forward"}:
            sync_warnings.append(f"Potential thin endpoint form for {info.label}: {target} ({endpoint_form})")
        audits.append(EndpointAudit(info, target, decl, endpoint_form, hypothesis_count, flags))

    return audits, errors, sync_warnings


def print_endpoint_table(audits: List[EndpointAudit]) -> None:
    print("Label | Num | Target | Location | Tier1 endpoint | Tier2 shape | Tier3 internalization | Flags")
    print("---|---|---|---|---|---|---|---")
    for item in audits:
        info = item.label
        target = item.target or "MISSING"
        if item.decl is None:
            loc = "MISSING"
        else:
            loc = f"{item.decl.path.relative_to(REPO)}:{item.decl.line}"
        t1 = "PASS" if item.target and item.decl else "FAIL"
        t2 = tier2_status(item.endpoint_form) if item.decl else "FAIL"
        t3 = tier3_status(item.flags) if item.decl else "gap"
        flags = ",".join(item.flags) if item.flags else "-"
        print(f"{info.label} | {info.number} | {target} | {loc} | {t1} | {t2} | {t3} | {flags}")


def print_summary(audits: List[EndpointAudit], errors: List[str], warnings: List[str]) -> None:
    total = len(audits)
    covered = sum(1 for a in audits if a.target and a.decl)
    tier2_pass = sum(1 for a in audits if a.decl and tier2_status(a.endpoint_form) == "PASS")
    tier3_candidates = sum(1 for a in audits if a.decl and tier3_status(a.flags) == "candidate")
    gaps = [a for a in audits if a.decl and tier3_status(a.flags) != "candidate"]
    assumption_targets = [a for a in audits if "assumption-target" in a.flags or "assumption-alias" in a.flags]
    heavy = [a for a in audits if any(f.startswith("assumption-heavy") for f in a.flags)]

    print("\nSUMMARY")
    print(f"- Tier 1 endpoint coverage: {covered}/{total} labels covered")
    print(f"- Tier 2 endpoint shape: {tier2_pass}/{total} labels pass structural shape checks")
    print(f"- Tier 3 internalization candidates: {tier3_candidates}/{total} labels have no structural gap flags")
    print(f"- Internalization gap/review flags: {len(gaps)} labels")
    print(f"- `_of_assumption` endpoint/alias flags: {len(assumption_targets)} labels")
    print(f"- Assumption-heavy endpoint flags (threshold >= {ASSUMPTION_HEAVY_THRESHOLD}): {len(heavy)} labels")
    print(f"- Gap ledger: {GAP_LEDGER.relative_to(REPO)}")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"- {w}")

    if errors:
        print("\nAUDIT FAILED:", file=sys.stderr)
        for e in errors:
            print(f"- {e}", file=sys.stderr)
    else:
        print("\nAUDIT PASSED")


def main() -> int:
    if not AUX.exists() or not STATEMENT_MAP.exists():
        print("ERROR: missing required files", file=sys.stderr)
        return 2

    labels = parse_aux_labels(AUX.read_text(encoding="utf-8"))
    aliases = parse_statement_map(STATEMENT_MAP.read_text(encoding="utf-8"))
    audits, errors, warnings = audit_labels(labels, aliases)

    print_endpoint_table(audits)
    print_summary(audits, errors, warnings)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
