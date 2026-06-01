#!/usr/bin/env python3
"""Freeze the current Comparator challenge statements.

The Comparator `Challenge` module is intentionally statement-only.  This lock
file records the exact imports, theorem order, comments, and statement hashes
that have been independently reviewed.  This script refuses to write the lock
unless the manifest, review dossier, and independent audit are synchronized and
all current audit verdicts are faithful.  Regenerate it only after a deliberate
paper-to-challenge audit pass.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEAN_DIR = REPO / "lean"
AUDIT_DIR = LEAN_DIR / "audit"
CONFIG = AUDIT_DIR / "comparator-paper-config.template.json"
MANIFEST = AUDIT_DIR / "comparator-paper-manifest.json"
REVIEW = AUDIT_DIR / "comparator-paper-review.json"
AUDIT = AUDIT_DIR / "comparator-challenge-audit.json"
CHALLENGE = LEAN_DIR / "FlowSinkhorn" / "Comparator" / "Challenge.lean"
LOCK = AUDIT_DIR / "comparator-challenge-lock.json"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def imports(text: str) -> list[str]:
    return re.findall(r"^import\s+([A-Za-z0-9_.]+)\s*$", text, re.M)


def is_trusted_challenge_import(module: str) -> bool:
    return module == "Mathlib" or module.startswith("FlowSinkhorn.Comparator.Vocabulary.")


def extract_challenge_entries(text: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    matches = list(re.finditer(r"^theorem\s+([A-Za-z0-9_]+)\s*:\s*", text, re.M))
    for idx, m in enumerate(matches):
        block_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[m.start() : block_end]
        proof = re.search(r"\s*:=\s*by\n\s+sorry\s*(?:\n|$)", block, re.S)
        if proof is None:
            raise SystemExit(f"Challenge theorem {m.group(1)} is not a statement-only sorry")

        prefix = text[: m.start()]
        comment_start = prefix.rfind("/--")
        comment_end = prefix.rfind("-/")
        if comment_start == -1 or comment_end < comment_start:
            raise SystemExit(f"Challenge theorem {m.group(1)} has no preceding doc comment")
        if prefix[comment_end + 2 :].strip():
            raise SystemExit(f"Challenge theorem {m.group(1)} doc comment is not immediate")

        comment = prefix[comment_start : comment_end + 2]
        statement = block[m.end() - m.start() : proof.start()].strip()
        out.append(
            {
                "name": m.group(1),
                "line": str(text[: m.start()].count("\n") + 1),
                "comment_sha256": sha256_text(compact(comment)),
                "statement_sha256": sha256_text(compact(statement)),
            }
        )
    return out


def counter(items: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return out


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    challenge_text = CHALLENGE.read_text(encoding="utf-8")
    challenge_imports = imports(challenge_text)
    if not all(is_trusted_challenge_import(module) for module in challenge_imports):
        raise SystemExit(
            "Challenge imports are outside the audited Mathlib/Comparator.Vocabulary boundary"
        )
    if review.get("challenge_imports") != challenge_imports:
        raise SystemExit("Review challenge_imports are stale relative to Challenge.lean")
    if review.get("solution_imports") != ["FlowSinkhorn.KLProjection.StatementMap"]:
        raise SystemExit("Review solution_imports are outside the audited solution boundary")

    entries = extract_challenge_entries(challenge_text)
    theorem_names = config["theorem_names"]
    if [entry["name"] for entry in entries] != theorem_names:
        raise SystemExit("Challenge theorem entries do not match comparator config")

    manifest_entries = manifest.get("paper_statements", [])
    review_entries = review.get("entries", [])
    audit_entries = audit.get("entries", [])
    manifest_aliases = [item.get("label_alias") for item in manifest_entries]
    review_aliases = [item.get("challenge_alias") for item in review_entries]
    if manifest_aliases != theorem_names:
        raise SystemExit("Manifest label_alias list does not match comparator config")
    if review_aliases != theorem_names:
        raise SystemExit("Review challenge_alias list does not match comparator config")

    manifest_labels = [item.get("label") for item in manifest_entries]
    review_labels = [item.get("label") for item in review_entries]
    audit_labels = [item.get("label") for item in audit_entries]
    if review_labels != manifest_labels:
        raise SystemExit("Review labels are stale relative to the manifest")
    if audit_labels != manifest_labels:
        raise SystemExit("Independent audit labels are stale relative to the manifest")

    audit_status_date = audit.get("status_date")
    if audit_status_date is None:
        raise SystemExit("Independent audit is missing status_date")
    audit_counts = counter([str(item.get("verdict")) for item in audit_entries])
    if audit.get("verdict_counts") != audit_counts:
        raise SystemExit("Independent audit verdict_counts are stale")
    if audit_counts != {"faithful": len(manifest_labels)}:
        raise SystemExit("Independent audit is not fully faithful; refusing to write lock")

    manifest_by_alias = {
        item["label_alias"]: item for item in manifest_entries
    }
    review_by_alias = {
        item["challenge_alias"]: item for item in review_entries
    }
    audit_by_label = {item["label"]: item for item in audit_entries}

    locked_entries = []
    for entry in entries:
        name = entry["name"]
        manifest_item = manifest_by_alias[name]
        review_item = review_by_alias[name]
        audit_item = audit_by_label[manifest_item["label"]]
        if review_item.get("target") != manifest_item["target"]:
            raise SystemExit(f"Review target is stale for {manifest_item['label']}")
        if review_item.get("implementation_source") != manifest_item["source"]:
            raise SystemExit(f"Review implementation source is stale for {manifest_item['label']}")
        if review_item.get("number") != manifest_item["number"]:
            raise SystemExit(f"Review numbering is stale for {manifest_item['label']}")
        if review_item.get("independent_audit_verdict") != audit_item["verdict"]:
            raise SystemExit(f"Review independent audit verdict is stale for {manifest_item['label']}")
        if review_item.get("review_status") != "independent audit: faithful":
            raise SystemExit(f"Review status is not fully faithful for {manifest_item['label']}")
        if review_item.get("challenge_solution_statement_match") is not True:
            raise SystemExit(
                f"Challenge/Solution statement mismatch for {manifest_item['label']}"
            )
        locked_entries.append(
            {
                **entry,
                "label": manifest_item["label"],
                "number": manifest_item["number"],
                "target": manifest_item["target"],
                "implementation_source": manifest_item["source"],
                "paper_source": review_item["paper_source"],
                "review_status": review_item["review_status"],
                "challenge_solution_statement_match": review_item[
                    "challenge_solution_statement_match"
                ],
            }
        )

    LOCK.write_text(
        json.dumps(
            {
                "status_date": audit_status_date,
                "purpose": (
                    "Frozen Comparator challenge lock.  It records the exact "
                    "trusted statement-only Challenge module reviewed against the paper."
                ),
                "generated_from": [
                    "lean/FlowSinkhorn/Comparator/Challenge.lean",
                    "lean/audit/comparator-paper-manifest.json",
                    "lean/audit/comparator-paper-review.json",
                    "lean/audit/comparator-challenge-audit.json",
                ],
                "challenge_module": config["challenge_module"],
                "solution_module": config["solution_module"],
                "full_file_sha256": sha256_text(challenge_text),
                "imports": challenge_imports,
                "theorem_names": theorem_names,
                "entries": locked_entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {LOCK.relative_to(LEAN_DIR)} with {len(locked_entries)} locked entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
