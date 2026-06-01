#!/usr/bin/env python3
"""Generate a paper-to-Comparator challenge review dossier.

This script does not prove semantic equivalence between LaTeX and Lean.  Its
purpose is to make the final trusted-review step explicit and reproducible:
for every paper label, it records the LaTeX statement source and the exact Lean
challenge theorem statement that Comparator checks against the solution.
"""

from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEAN_DIR = REPO / "lean"
AUDIT_DIR = LEAN_DIR / "audit"
PAPER = REPO / "neurips" / "paper.tex"
MANIFEST = AUDIT_DIR / "comparator-paper-manifest.json"
CHALLENGE = LEAN_DIR / "FlowSinkhorn" / "Comparator" / "Challenge.lean"
SOLUTION = LEAN_DIR / "FlowSinkhorn" / "Comparator" / "Solution.lean"
REVIEW_JSON = AUDIT_DIR / "comparator-paper-review.json"
REVIEW_MD = AUDIT_DIR / "comparator-paper-review.md"
AUDIT_JSON = AUDIT_DIR / "comparator-challenge-audit.json"
CHALLENGE_LOCK = AUDIT_DIR / "comparator-challenge-lock.json"


def clean_latex(text: str) -> str:
    text = re.sub(r"%.*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compact_lean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_imports(text: str) -> list[str]:
    return re.findall(r"^import\s+([A-Za-z0-9_.]+)\s*$", text, re.M)


def extract_paper_environments(text: str) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    env_re = re.compile(
        r"\\begin\{(theorem|proposition|lemma|corollary)\}"
        r"(?:\[([^\]]*)\])?(.*?)\\end\{\1\}",
        re.S,
    )
    for m in env_re.finditer(text):
        kind = m.group(1)
        title = m.group(2) or ""
        body = m.group(3)
        labels = re.findall(r"\\label\{([^}]+)\}", body)
        line = text[: m.start()].count("\n") + 1
        for label in labels:
            out[label] = {
                "kind": kind,
                "title": title,
                "line": line,
                "body": body.strip(),
                "body_compact": clean_latex(body),
            }
    return out


def extract_challenge_statements(text: str) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    theorem_re = re.compile(r"^theorem\s+([A-Za-z0-9_]+)\s*:\s*(.*?)\s*:=\s*by\n", re.M | re.S)
    for m in theorem_re.finditer(text):
        name = m.group(1)
        statement = m.group(2).strip()
        line = text[: m.start()].count("\n") + 1
        out[name] = {
            "line": line,
            "statement": statement,
            "statement_compact": compact_lean(statement),
        }
    return out


def md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def load_independent_audit() -> tuple[dict[str, dict[str, object]], dict[str, int], str | None]:
    if not AUDIT_JSON.exists():
        return {}, {}, None
    data = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    entries = {
        str(item["label"]): item
        for item in data.get("entries", [])
        if "label" in item
    }
    counts = {
        str(key): int(value)
        for key, value in data.get("verdict_counts", {}).items()
    }
    status_date = data.get("status_date")
    return entries, counts, str(status_date) if status_date is not None else None


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paper_envs = extract_paper_environments(PAPER.read_text(encoding="utf-8"))
    challenge_text = CHALLENGE.read_text(encoding="utf-8")
    solution_text = SOLUTION.read_text(encoding="utf-8")
    challenge = extract_challenge_statements(challenge_text)
    solution = extract_challenge_statements(solution_text)
    independent_audit, independent_counts, independent_status_date = load_independent_audit()

    entries: list[dict[str, object]] = []
    errors: list[str] = []

    for item in manifest["paper_statements"]:
        label = item["label"]
        alias = item["label_alias"]
        paper_info = paper_envs.get(label)
        challenge_info = challenge.get(alias)
        if paper_info is None:
            errors.append(f"missing paper environment for {label}")
            continue
        if challenge_info is None:
            errors.append(f"missing challenge theorem for {alias}")
            continue
        solution_info = solution.get(alias)
        if solution_info is None:
            errors.append(f"missing solution theorem for {alias}")
            continue
        paper_statement_compact = str(paper_info["body_compact"])
        challenge_statement_compact = str(challenge_info["statement_compact"])
        solution_statement_compact = str(solution_info["statement_compact"])
        audit_entry = independent_audit.get(label)
        review_status = "bootstrap-extracted; requires independent paper-to-Lean review"
        if audit_entry is not None:
            review_status = f"independent audit: {audit_entry.get('verdict')}"

        entry = {
            "label": label,
            "number": item["number"],
            "kind": item["kind"],
            "paper_env_kind": paper_info["kind"],
            "title": paper_info["title"],
            "paper_source": f"neurips/paper.tex:{paper_info['line']}",
            "paper_statement_compact": paper_statement_compact,
            "paper_statement_sha256": sha256_text(paper_statement_compact),
            "challenge_alias": alias,
            "challenge_source": f"lean/FlowSinkhorn/Comparator/Challenge.lean:{challenge_info['line']}",
            "challenge_statement": challenge_info["statement"],
            "challenge_statement_sha256": sha256_text(challenge_statement_compact),
            "solution_source": f"lean/FlowSinkhorn/Comparator/Solution.lean:{solution_info['line']}",
            "solution_statement_sha256": sha256_text(solution_statement_compact),
            "challenge_solution_statement_match": challenge_statement_compact == solution_statement_compact,
            "target": item["target"],
            "implementation_source": item["source"],
            "review_status": review_status,
        }
        if audit_entry is not None:
            entry.update(
                {
                    "independent_audit_verdict": audit_entry.get("verdict"),
                    "independent_audit_reviewer": audit_entry.get("reviewer"),
                    "independent_audit_note": audit_entry.get("note"),
                }
            )
        entries.append(entry)

    if errors:
        raise SystemExit("\n".join(errors))

    REVIEW_JSON.write_text(
        json.dumps(
            {
                "generated_from": [
                    "neurips/paper.tex",
                    "lean/audit/comparator-paper-manifest.json",
                    "lean/FlowSinkhorn/Comparator/Challenge.lean",
                    "lean/FlowSinkhorn/Comparator/Solution.lean",
                ],
                "challenge_imports": extract_imports(challenge_text),
                "solution_imports": extract_imports(solution_text),
                "independent_audit_file": str(AUDIT_JSON.relative_to(REPO))
                if AUDIT_JSON.exists()
                else None,
                "challenge_lock_file": str(CHALLENGE_LOCK.relative_to(REPO))
                if CHALLENGE_LOCK.exists()
                else None,
                "entries": entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lines: list[str] = [
        "# Comparator Paper-to-Challenge Review",
        "",
        f"Status date: {independent_status_date or '2026-05-29'}.",
        "",
        "This dossier is the audit surface for the current frozen Comparator challenge.",
        "It lists every paper theorem/proposition/lemma/corollary label, the LaTeX source location,",
        "the generated Comparator challenge theorem, and the canonical implementation target.",
        "It also records SHA-256 fingerprints of the compacted paper excerpt, challenge statement,",
        "and solution statement, plus a direct challenge-vs-solution statement match check.",
        "",
        "Trust boundary: the current `Challenge` module imports only Mathlib and the canonical",
        "`FlowSinkhorn.Comparator.Vocabulary.*` statement-language layer rather than proof-bearing",
        "implementation modules.  This is now",
        "checked by `scripts/check_comparator_scaffold.py` and by",
        "`scripts/check_comparator_trust_boundary.py`, which verifies that importing the",
        "challenge exposes no manifest implementation endpoint.",
        "",
        "Challenge-freeze status: `lean/audit/comparator-challenge-lock.json` records the exact",
        "reviewed imports, comments, theorem order, and statement hashes.  Any change to",
        "`Challenge.lean` must be followed by a fresh independent paper-to-challenge review",
        "before regenerating the lock.  A final external certificate should run Comparator with",
        "real `landrun` on Linux after the independent semantic audit has no remaining",
        "qualified entries.",
        "",
        "Important: the table is generated from the LaTeX labels and Comparator files.",
        "When an independent audit verdict is available, the row status reports that verdict;",
        "otherwise it remains marked as bootstrap-extracted and still needing review.",
        "",
    ]
    if independent_audit:
        faithful_count = independent_counts.get("faithful", 0)
        mismatch_count = independent_counts.get("mismatch", 0)
        qualified_count = sum(
            count
            for verdict, count in independent_counts.items()
            if verdict not in {"faithful", "mismatch"}
        )
        all_reviewed = faithful_count == len(entries)
        status_sentence = (
            "The current independent audit therefore says that this frozen challenge is "
            "paper-faithful for every audited entry; the remaining caveat is only the "
            "external hardened Comparator run with real `landrun`."
            if all_reviewed
            else
            "The current independent audit therefore says that this frozen challenge is "
            "structurally trusted as a Comparator artifact, but still has non-faithful "
            "paper-to-challenge entries to remediate."
        )
        lines += [
            "Independent audit status:",
            "",
            f"- audit file: `{AUDIT_JSON.relative_to(REPO)}`",
            f"- faithful without qualification: `{faithful_count}`",
            f"- qualified entries: `{qualified_count}`",
            f"- mismatch: `{mismatch_count}`",
            "",
            status_sentence,
            "",
        ]
    lines += [
        "Challenge imports:",
        "",
        "```text",
        *extract_imports(challenge_text),
        "```",
        "",
        "Solution imports:",
        "",
        "```text",
        *extract_imports(solution_text),
        "```",
        "",
        "## Coverage Table",
        "",
        "| # | Paper label | Paper source | Challenge theorem | Implementation | Match | Independent audit | Status |",
        "|---:|---|---|---|---|---|---|---|",
    ]

    for idx, entry in enumerate(entries, 1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    f"`{entry['label']}`",
                    f"`{entry['paper_source']}`",
                    f"`{entry['challenge_alias']}` at `{entry['challenge_source']}`",
                    f"`{entry['target']}` at `{entry['implementation_source']}`",
                    "`yes`" if entry["challenge_solution_statement_match"] else "`NO`",
                    f"`{entry.get('independent_audit_verdict', 'not-run')}`",
                    md_escape(str(entry["review_status"])),
                ]
            )
            + " |"
        )

    lines += ["", "## Detailed Entries", ""]
    for idx, entry in enumerate(entries, 1):
        lines += [
            f"### {idx}. `{entry['label']}`",
            "",
            f"- Number: `{entry['number']}`",
            f"- LaTeX environment: `{entry['paper_env_kind']}`",
            f"- Title: {entry['title'] or '(untitled)'}",
            f"- Paper source: `{entry['paper_source']}`",
            f"- Challenge theorem: `{entry['challenge_alias']}`",
            f"- Challenge source: `{entry['challenge_source']}`",
            f"- Solution source: `{entry['solution_source']}`",
            f"- Implementation target: `{entry['target']}`",
            f"- Implementation source: `{entry['implementation_source']}`",
            f"- Challenge/Solution statement match: `{entry['challenge_solution_statement_match']}`",
            f"- Paper statement SHA-256: `{entry['paper_statement_sha256']}`",
            f"- Challenge statement SHA-256: `{entry['challenge_statement_sha256']}`",
            f"- Solution statement SHA-256: `{entry['solution_statement_sha256']}`",
            f"- Review status: {entry['review_status']}",
        ]
        if "independent_audit_verdict" in entry:
            lines += [
                f"- Independent audit verdict: `{entry['independent_audit_verdict']}`",
                f"- Independent audit reviewer: `{entry['independent_audit_reviewer']}`",
                f"- Independent audit note: {entry['independent_audit_note']}",
            ]
        lines += [
            "",
            "Paper statement excerpt:",
            "",
            "```text",
            str(entry["paper_statement_compact"])[:1400],
            "```",
            "",
            "Lean challenge statement:",
            "",
            "```lean",
            f"theorem {entry['challenge_alias']} : {entry['challenge_statement']} := by",
            "  sorry",
            "```",
            "",
        ]

    REVIEW_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(entries)} review entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
