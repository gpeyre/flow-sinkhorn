#!/usr/bin/env python3
"""Check that the Comparator challenge matches its frozen audit lock."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEAN_DIR = REPO / "lean"
AUDIT_DIR = LEAN_DIR / "audit"
CHALLENGE = LEAN_DIR / "FlowSinkhorn" / "Comparator" / "Challenge.lean"
LOCK = AUDIT_DIR / "comparator-challenge-lock.json"
AUDIT = AUDIT_DIR / "comparator-challenge-audit.json"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def imports(text: str) -> list[str]:
    return re.findall(r"^import\s+([A-Za-z0-9_.]+)\s*$", text, re.M)


def extract_challenge_entries(text: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    matches = list(re.finditer(r"^theorem\s+([A-Za-z0-9_]+)\s*:\s*", text, re.M))
    for idx, m in enumerate(matches):
        block_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[m.start() : block_end]
        proof = re.search(r"\s*:=\s*by\n\s+sorry\s*(?:\n|$)", block, re.S)
        if proof is None:
            raise ValueError(f"Challenge theorem {m.group(1)} is not a statement-only sorry")

        prefix = text[: m.start()]
        comment_start = prefix.rfind("/--")
        comment_end = prefix.rfind("-/")
        if comment_start == -1 or comment_end < comment_start:
            raise ValueError(f"Challenge theorem {m.group(1)} has no preceding doc comment")
        if prefix[comment_end + 2 :].strip():
            raise ValueError(f"Challenge theorem {m.group(1)} doc comment is not immediate")

        comment = prefix[comment_start : comment_end + 2]
        statement = block[m.end() - m.start() : proof.start()].strip()
        out.append(
            {
                "name": m.group(1),
                "comment_sha256": sha256_text(compact(comment)),
                "statement_sha256": sha256_text(compact(statement)),
            }
        )
    return out


def fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    if not LOCK.exists():
        return fail("Missing comparator-challenge-lock.json; run generate_comparator_challenge_lock.py")
    if not AUDIT.exists():
        return fail("Missing comparator-challenge-audit.json; run the independent challenge audit")

    challenge_text = CHALLENGE.read_text(encoding="utf-8")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    if lock.get("status_date") != audit.get("status_date"):
        return fail(
            "Challenge lock status_date differs from comparator-challenge-audit.json; "
            "regenerate the lock after the audited status changes"
        )

    if sha256_text(challenge_text) != lock.get("full_file_sha256"):
        return fail(
            "Challenge file hash differs from frozen lock.  Re-run the independent "
            "paper-to-challenge audit before regenerating the lock."
        )
    if imports(challenge_text) != lock.get("imports"):
        return fail("Challenge imports differ from frozen lock")

    current_entries = extract_challenge_entries(challenge_text)
    locked_entries = lock.get("entries", [])
    if [entry["name"] for entry in current_entries] != lock.get("theorem_names"):
        return fail("Challenge theorem order differs from frozen lock")
    if len(current_entries) != len(locked_entries):
        return fail("Challenge theorem count differs from frozen lock")

    audit_entries = audit.get("entries", [])
    if len(audit_entries) != len(locked_entries):
        return fail("Independent audit entry count differs from frozen lock")

    for current, locked in zip(current_entries, locked_entries):
        name = current["name"]
        if name != locked.get("name"):
            return fail(f"Challenge theorem name differs from frozen lock near {name}")
        if current["statement_sha256"] != locked.get("statement_sha256"):
            return fail(f"Challenge statement hash differs from frozen lock for {name}")
        if current["comment_sha256"] != locked.get("comment_sha256"):
            return fail(f"Challenge comment hash differs from frozen lock for {name}")

    for locked, audited in zip(locked_entries, audit_entries):
        label = str(locked.get("label", "<unknown>"))
        if locked.get("label") != audited.get("label"):
            return fail(f"Challenge lock label differs from independent audit near {label}")
        if audited.get("verdict") != "faithful":
            return fail(f"Independent audit verdict is not faithful for {label}")
        if locked.get("review_status") != "independent audit: faithful":
            return fail(f"Challenge lock review_status is not fully faithful for {label}")
        if locked.get("challenge_solution_statement_match") is not True:
            return fail(f"Challenge/Solution statement match is not locked as true for {label}")

    print(f"COMPARATOR CHALLENGE LOCK CHECK PASSED ({len(current_entries)} locked entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
