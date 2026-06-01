#!/usr/bin/env python3
"""Check the repository-side Comparator scaffold.

This is a lightweight structural check.  It does not replace Comparator; it
guards the files that Comparator consumes.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEAN_DIR = REPO / "lean"
AUDIT_DIR = LEAN_DIR / "audit"
CONFIG = AUDIT_DIR / "comparator-paper-config.template.json"
MANIFEST = AUDIT_DIR / "comparator-paper-manifest.json"
REVIEW = AUDIT_DIR / "comparator-paper-review.json"
REVIEW_MD = AUDIT_DIR / "comparator-paper-review.md"
AUDIT = AUDIT_DIR / "comparator-challenge-audit.json"
AUDIT_MD = AUDIT_DIR / "comparator-challenge-audit.md"
COMPARATOR_DOC = AUDIT_DIR / "comparator.md"
REVIEW_GENERATOR = LEAN_DIR / "scripts" / "generate_comparator_review.py"
CHALLENGE_LOCK = AUDIT_DIR / "comparator-challenge-lock.json"
PAPER = REPO / "neurips" / "paper.tex"
CHALLENGE = LEAN_DIR / "FlowSinkhorn" / "Comparator" / "Challenge.lean"
SOLUTION = LEAN_DIR / "FlowSinkhorn" / "Comparator" / "Solution.lean"

LABEL_PREFIXES = ("prop:", "lem:", "thm:", "cor:", "app-prop:", "app-lem:", "app-cor:")
EXPECTED_CHALLENGE_IMPORTS = [
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
EXPECTED_VOCABULARY_IMPORTS = EXPECTED_CHALLENGE_IMPORTS[1:]
EXPECTED_VOCABULARY_UMBRELLA = "FlowSinkhorn.Comparator.Vocabulary"
EXPECTED_VOCABULARY_SHIMS = {
    "FlowSinkhorn.KLProjection.Legacy.Section2DualityVocabulary":
        "FlowSinkhorn.Comparator.Vocabulary.Legacy.Section2Duality",
    "FlowSinkhorn.KLProjection.UniformBoundVocabulary":
        "FlowSinkhorn.Comparator.Vocabulary.UniformBound",
    "FlowSinkhorn.KLProjection.TopicalVocabulary":
        "FlowSinkhorn.Comparator.Vocabulary.Topical",
    "FlowSinkhorn.KLProjection.BlockQuotientVocabulary":
        "FlowSinkhorn.Comparator.Vocabulary.BlockQuotient",
    "FlowSinkhorn.KLProjection.Setup.VariationGeometryVocabulary":
        "FlowSinkhorn.Comparator.Vocabulary.Setup.VariationGeometry",
    "FlowSinkhorn.KLProjection.SweepVocabulary":
        "FlowSinkhorn.Comparator.Vocabulary.Sweep",
    "FlowSinkhorn.KLProjection.PrimalDualBounds.Vocabulary":
        "FlowSinkhorn.Comparator.Vocabulary.PrimalDualBounds",
    "FlowSinkhorn.KLProjection.DualConvergence.Vocabulary":
        "FlowSinkhorn.Comparator.Vocabulary.DualConvergence",
    "FlowSinkhorn.KLProjection.Applications.OT.HGammaVocabulary":
        "FlowSinkhorn.Comparator.Vocabulary.Applications.OT.HGamma",
    "FlowSinkhorn.KLProjection.Applications.OT.ComplexityVocabulary":
        "FlowSinkhorn.Comparator.Vocabulary.Applications.OT.Complexity",
    "FlowSinkhorn.KLProjection.Applications.GraphW1.ClosedFormsVocabulary":
        "FlowSinkhorn.Comparator.Vocabulary.Applications.GraphW1.ClosedForms",
    "FlowSinkhorn.KLProjection.Applications.GraphW1.HGammaVocabulary":
        "FlowSinkhorn.Comparator.Vocabulary.Applications.GraphW1.HGamma",
    "FlowSinkhorn.KLProjection.Applications.GraphW1.ComplexityVocabulary":
        "FlowSinkhorn.Comparator.Vocabulary.Applications.GraphW1.Complexity",
}
EXPECTED_SOLUTION_IMPORTS = ["FlowSinkhorn.KLProjection.StatementMap"]
EXPECTED_PERMITTED_AXIOMS = ["propext", "Quot.sound", "Classical.choice"]
VALID_AUDIT_VERDICTS = {
    "faithful",
    "mismatch",
    "insufficient-to-judge",
}
TEXT_SUFFIXES_TO_SCAN = {
    ".bib",
    ".json",
    ".lean",
    ".md",
    ".py",
    ".sh",
    ".sty",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_TEXT_SCAN_DIRS = {
    ".git",
    ".lake",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}


def theorem_names(text: str) -> list[str]:
    return re.findall(r"^theorem\s+([A-Za-z0-9_]+)\s*:", text, re.M)


def imports(text: str) -> list[str]:
    return re.findall(r"^import\s+([A-Za-z0-9_.]+)\s*$", text, re.M)


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_trusted_vocabulary_module(module: str) -> bool:
    return module.startswith("FlowSinkhorn.Comparator.Vocabulary.")


def module_path(module: str) -> Path:
    return LEAN_DIR / (module.replace(".", "/") + ".lean")


def strip_lean_comments(text: str) -> str:
    """Remove Lean line/block comments while preserving line numbers."""
    out: list[str] = []
    idx = 0
    depth = 0
    while idx < len(text):
        if depth:
            if text.startswith("/-", idx):
                out.append("  ")
                depth += 1
                idx += 2
            elif text.startswith("-/", idx):
                out.append("  ")
                depth -= 1
                idx += 2
            else:
                out.append("\n" if text[idx] == "\n" else " ")
                idx += 1
            continue

        if text.startswith("--", idx):
            while idx < len(text) and text[idx] != "\n":
                out.append(" ")
                idx += 1
        elif text.startswith("/-", idx):
            out.append("  ")
            depth += 1
            idx += 2
        else:
            out.append(text[idx])
            idx += 1
    return "".join(out)


def forbidden_vocabulary_lines(text: str) -> list[str]:
    text = strip_lean_comments(text)
    forbidden: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if re.match(r"^\s*(?:theorem|lemma|axiom|opaque|example)\b", line):
            forbidden.append(f"{line_no}: proof declaration")
        if re.match(r"^\s*unsafe\b", line):
            forbidden.append(f"{line_no}: unsafe declaration")
        if re.search(r"\b(sorry|admit)\b", line):
            forbidden.append(f"{line_no}: proof hole")
    return forbidden


def check_vocabulary_umbrella() -> list[str]:
    errors: list[str] = []
    path = module_path(EXPECTED_VOCABULARY_UMBRELLA)
    if not path.exists():
        return [f"missing vocabulary umbrella module {path}"]
    text = path.read_text(encoding="utf-8")
    if imports(text) != EXPECTED_VOCABULARY_IMPORTS:
        errors.append("Comparator vocabulary umbrella imports do not match the audited allowlist")
    forbidden = forbidden_vocabulary_lines(text)
    if forbidden:
        errors.append(
            "Comparator vocabulary umbrella contains forbidden trusted-surface lines: "
            + "; ".join(forbidden[:8])
        )
    return errors


def check_vocabulary_compatibility_shims() -> list[str]:
    errors: list[str] = []
    code_decl_re = re.compile(
        r"^\s*(?:namespace|section|noncomputable|def|abbrev|structure|class|"
        r"inductive|instance|theorem|lemma|axiom|opaque|example|unsafe)\b",
        re.M,
    )
    for shim_module, target_module in EXPECTED_VOCABULARY_SHIMS.items():
        path = module_path(shim_module)
        if not path.exists():
            errors.append(f"missing vocabulary compatibility shim {path}")
            continue
        text = path.read_text(encoding="utf-8")
        code_text = strip_lean_comments(text)
        if imports(text) != [target_module]:
            errors.append(
                f"compatibility shim {shim_module} must import exactly {target_module}"
            )
        code_decls = code_decl_re.findall(code_text)
        if code_decls:
            errors.append(
                f"compatibility shim {shim_module} is not import-only; declarations found"
            )
        forbidden = forbidden_vocabulary_lines(text)
        if forbidden:
            errors.append(
                f"compatibility shim {shim_module} contains forbidden lines: "
                + "; ".join(forbidden[:8])
            )
    return errors


def check_challenge_import_vocabulary_layer(challenge_imports: list[str]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()

    def check_module(module: str, root: str) -> None:
        if module == "Mathlib":
            return
        if not module.startswith("FlowSinkhorn."):
            return
        if not is_trusted_vocabulary_module(module):
            errors.append(
                f"{root} imports {module}, which is outside the trusted Comparator vocabulary layer"
            )
            return
        if module in seen:
            return
        seen.add(module)

        path = module_path(module)
        if not path.exists():
            errors.append(f"challenge vocabulary import {module} does not resolve to {path}")
            return

        text = path.read_text(encoding="utf-8")
        forbidden_lines = forbidden_vocabulary_lines(text)
        if forbidden_lines:
            errors.append(
                f"challenge vocabulary module {module} contains forbidden trusted-surface lines: "
                + "; ".join(forbidden_lines[:8])
            )

        stale_imports = [
            imported
            for imported in imports(text)
            if imported.startswith("FlowSinkhorn.KLProjection.")
        ]
        if stale_imports:
            errors.append(
                f"challenge vocabulary module {module} imports implementation-side modules: "
                + ", ".join(stale_imports[:8])
            )

        for imported in imports(text):
            check_module(imported, module)

    for module in challenge_imports:
        if module != "Mathlib" and not is_trusted_vocabulary_module(module):
            errors.append(
                f"challenge import {module} is not in FlowSinkhorn.Comparator.Vocabulary.*"
            )
            continue
        check_module(module, "Challenge")
    return errors


def extract_challenge_blocks(text: str) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    matches = list(re.finditer(r"^theorem\s+([A-Za-z0-9_]+)\s*:\s*", text, re.M))
    for idx, m in enumerate(matches):
        block_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[m.start() : block_end]
        proof = re.search(r"\s*:=\s*by\n\s+sorry\s*(?:\n|$)", block, re.S)
        if proof is None:
            blocks.append({"name": m.group(1), "statement": "", "comment": "", "has_sorry": "false"})
            continue

        prefix = text[: m.start()]
        comment_start = prefix.rfind("/--")
        comment_end = prefix.rfind("-/")
        has_immediate_comment = (
            comment_start != -1
            and comment_end >= comment_start
            and not prefix[comment_end + 2 :].strip()
        )
        comment = prefix[comment_start : comment_end + 2] if has_immediate_comment else ""
        statement = block[m.end() - m.start() : proof.start()].strip()
        blocks.append(
            {
                "name": m.group(1),
                "statement": statement,
                "comment": comment,
                "has_sorry": "true",
            }
        )
    return blocks


def challenge_sorry_theorem_names(text: str) -> list[tuple[str, str]]:
    return [
        (block["name"], block["statement"])
        for block in extract_challenge_blocks(text)
        if block["has_sorry"] == "true"
    ]


def challenge_lock_entries(text: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for block in extract_challenge_blocks(text):
        if block["has_sorry"] != "true" or not block["comment"]:
            continue
        out.append(
            {
                "name": block["name"],
                "comment_sha256": sha256_text(compact(block["comment"])),
                "statement_sha256": sha256_text(compact(block["statement"])),
            }
        )
    return out


def paper_statement_labels(text: str) -> tuple[list[str], list[str]]:
    labels_out: list[str] = []
    errors: list[str] = []
    env_re = re.compile(
        r"\\begin\{(theorem|proposition|lemma|corollary)\}"
        r"(?:\[([^\]]*)\])?(.*?)\\end\{\1\}",
        re.S,
    )
    for m in env_re.finditer(text):
        env_kind = m.group(1)
        body = m.group(3)
        line = text[: m.start()].count("\n") + 1
        labels = [
            label
            for label in re.findall(r"\\label\{([^}]+)\}", body)
            if label.startswith(LABEL_PREFIXES)
        ]
        if not labels:
            errors.append(f"paper theorem-like environment at neurips/paper.tex:{line} has no label")
            continue
        for label in labels:
            labels_out.append(label)
    return labels_out, errors


def fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


def stale_qualified_verdict_phrases() -> list[str]:
    old = "accept" + "able"
    middle = "with"
    suffix = "inter" + "pretation"
    return [
        f"{old}-{middle}-{suffix}",
        f"{old} {middle} {suffix}",
        f"{old}-{middle} {suffix}",
        f"{old} {middle}-{suffix}",
    ]


def relevant_repo_text_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES_TO_SCAN:
            continue
        if any(part in SKIP_TEXT_SCAN_DIRS for part in path.relative_to(REPO).parts):
            continue
        files.append(path)
    return files


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if config.get("challenge_module") != manifest.get("challenge_module"):
        return fail("Config challenge_module does not match manifest")
    if config.get("solution_module") != manifest.get("solution_module"):
        return fail("Config solution_module does not match manifest")
    if config.get("permitted_axioms") != EXPECTED_PERMITTED_AXIOMS:
        return fail("Config permitted_axioms changed from the audited allowlist")
    if config.get("theorem_names") != manifest.get("theorem_names"):
        return fail("Config theorem_names do not match manifest")
    expected = config["theorem_names"]
    expected_labels = [item["label"] for item in manifest["paper_statements"]]
    manifest_entries = manifest["paper_statements"]
    paper_labels, paper_errors = paper_statement_labels(PAPER.read_text(encoding="utf-8"))
    if paper_errors:
        return fail("Paper label coverage errors: " + "; ".join(paper_errors))
    if paper_labels != expected_labels:
        return fail("Manifest paper labels do not match direct neurips/paper.tex statement labels")

    unresolved_manifest = [
        str(item.get("label", "<unknown>"))
        for item in manifest_entries
        if item.get("target") in (None, "", "MISSING")
        or item.get("source") in (None, "", "MISSING")
    ]
    if unresolved_manifest:
        return fail("Manifest entries have unresolved Lean targets: " + ", ".join(unresolved_manifest))

    challenge_text = CHALLENGE.read_text(encoding="utf-8")
    solution_text = SOLUTION.read_text(encoding="utf-8")

    stale_trust_boundary_phrases = [
        "imports implementation modules",
        "not for a hardened 100% trusted challenge",
    ]
    stale_phrases = [phrase for phrase in stale_trust_boundary_phrases if phrase in challenge_text]
    if stale_phrases:
        return fail(
            "Challenge contains stale trust-boundary wording: "
            + ", ".join(repr(phrase) for phrase in stale_phrases)
        )

    challenge_imports = imports(challenge_text)
    solution_imports = imports(solution_text)

    if challenge_imports != EXPECTED_CHALLENGE_IMPORTS:
        return fail("Challenge imports do not match the audited allowlist")
    vocabulary_layer_errors = check_challenge_import_vocabulary_layer(challenge_imports)
    if vocabulary_layer_errors:
        return fail("Challenge import vocabulary-layer errors: " + "; ".join(vocabulary_layer_errors))
    vocabulary_umbrella_errors = check_vocabulary_umbrella()
    if vocabulary_umbrella_errors:
        return fail("Comparator vocabulary umbrella errors: " + "; ".join(vocabulary_umbrella_errors))
    vocabulary_shim_errors = check_vocabulary_compatibility_shims()
    if vocabulary_shim_errors:
        return fail("Vocabulary compatibility shim errors: " + "; ".join(vocabulary_shim_errors))
    if solution_imports != EXPECTED_SOLUTION_IMPORTS:
        return fail("Solution imports do not match the audited allowlist")

    challenge_names = theorem_names(challenge_text)
    solution_names = theorem_names(solution_text)

    if challenge_names != expected:
        return fail("Challenge theorem list does not match Comparator config theorem_names")
    if solution_names != expected:
        return fail("Solution theorem list does not match Comparator config theorem_names")

    missing_challenge_comments = [
        name
        for name in expected
        if not re.search(
            rf"/--\n(?: \* .*\n)+-\/\ntheorem\s+{re.escape(name)}\s*:",
            challenge_text,
            re.M,
        )
    ]
    if missing_challenge_comments:
        return fail(
            "Challenge theorem entries are missing paper-facing explanatory comments: "
            + ", ".join(missing_challenge_comments)
        )

    solution_exact_targets = re.findall(
        r"^\s+exact\s+FlowSinkhorn\.KLProjection\.StatementMap\.([A-Za-z0-9_]+)\s*$",
        solution_text,
        re.M,
    )
    if solution_exact_targets != expected:
        return fail("Solution proofs are not exactly the expected StatementMap alias proofs")

    banned_challenge_imports = [
        "FlowSinkhorn.KLProjection.StatementMap",
        "FlowSinkhorn.Comparator.Solution",
    ]
    for banned in banned_challenge_imports:
        if re.search(rf"^import\s+{re.escape(banned)}\s*$", challenge_text, re.M):
            return fail(f"Challenge imports forbidden module {banned}")

    if re.search(r"\b(sorry|admit|axiom)\b", solution_text):
        return fail("Solution contains sorry/admit/axiom")

    challenge_sorries = len(re.findall(r"^\s+sorry\s*$", challenge_text, re.M))
    if challenge_sorries != len(expected):
        return fail(
            f"Challenge should have exactly one statement hole per theorem "
            f"({len(expected)} expected, found {challenge_sorries})"
        )
    challenge_sorry_names = [name for name, _ in challenge_sorry_theorem_names(challenge_text)]
    if challenge_sorry_names != expected:
        return fail("Challenge proofs are not exactly the expected statement-only sorry proofs")

    if not CHALLENGE_LOCK.exists():
        return fail(
            "Missing comparator-challenge-lock.json; run "
            "scripts/generate_comparator_challenge_lock.py after independent challenge review"
        )
    challenge_lock = json.loads(CHALLENGE_LOCK.read_text(encoding="utf-8"))
    if challenge_lock.get("full_file_sha256") != sha256_text(challenge_text):
        return fail(
            "Challenge file differs from frozen comparator-challenge-lock.json; "
            "regenerate the lock only after an independent paper-to-challenge audit"
        )
    if challenge_lock.get("imports") != challenge_imports:
        return fail("Challenge imports differ from frozen comparator-challenge-lock.json")
    if challenge_lock.get("theorem_names") != expected:
        return fail("Challenge lock theorem_names do not match Comparator config")
    locked_entries = challenge_lock.get("entries", [])
    current_lock_entries = challenge_lock_entries(challenge_text)
    if len(locked_entries) != len(current_lock_entries):
        return fail("Challenge lock entry count does not match current Challenge theorem count")
    for current, locked in zip(current_lock_entries, locked_entries):
        if current["name"] != locked.get("name"):
            return fail(f"Challenge lock theorem-order mismatch near {current['name']}")
        if current["statement_sha256"] != locked.get("statement_sha256"):
            return fail(f"Challenge statement differs from frozen lock for {current['name']}")
        if current["comment_sha256"] != locked.get("comment_sha256"):
            return fail(f"Challenge comment differs from frozen lock for {current['name']}")

    if not REVIEW.exists():
        return fail("Missing comparator-paper-review.json; run generate_comparator_review.py")
    if not REVIEW_MD.exists():
        return fail("Missing comparator-paper-review.md; run generate_comparator_review.py")

    review_md_text = REVIEW_MD.read_text(encoding="utf-8")
    stale_review_phrases = [phrase for phrase in stale_trust_boundary_phrases if phrase in review_md_text]
    if stale_review_phrases:
        return fail(
            "Review dossier contains stale trust-boundary wording: "
            + ", ".join(repr(phrase) for phrase in stale_review_phrases)
        )

    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    review_entries = review.get("entries", [])
    review_aliases = [item.get("challenge_alias") for item in review_entries]
    review_labels = [item.get("label") for item in review_entries]

    if review_aliases != expected:
        return fail("Review challenge_alias list does not match Comparator config theorem_names")
    if review_labels != expected_labels:
        return fail("Review label list does not match paper manifest labels")
    if review.get("challenge_imports") != EXPECTED_CHALLENGE_IMPORTS:
        return fail("Review challenge_imports do not match the audited allowlist")
    if review.get("solution_imports") != EXPECTED_SOLUTION_IMPORTS:
        return fail("Review solution_imports do not match the audited allowlist")

    manifest_by_label = {item["label"]: item for item in manifest_entries}
    stale_review_entries = [
        str(item.get("label", "<unknown>"))
        for item in review_entries
        if item.get("label") not in manifest_by_label
        or item.get("target") != manifest_by_label[item["label"]].get("target")
        or item.get("implementation_source") != manifest_by_label[item["label"]].get("source")
        or item.get("number") != manifest_by_label[item["label"]].get("number")
    ]
    if stale_review_entries:
        return fail("Review entries are stale relative to manifest: " + ", ".join(stale_review_entries))

    missing_sources = [
        str(item.get("label", "<unknown>"))
        for item in review_entries
        if not item.get("paper_source")
        or not item.get("challenge_source")
        or not item.get("solution_source")
        or not item.get("implementation_source")
        or not item.get("target")
        or item.get("implementation_source") == "MISSING"
        or item.get("target") == "MISSING"
    ]
    if missing_sources:
        return fail("Review entries are missing source metadata: " + ", ".join(missing_sources))

    missing_hashes = [
        str(item.get("label", "<unknown>"))
        for item in review_entries
        if not item.get("paper_statement_sha256")
        or not item.get("challenge_statement_sha256")
        or not item.get("solution_statement_sha256")
    ]
    if missing_hashes:
        return fail("Review entries are missing statement hashes: " + ", ".join(missing_hashes))

    mismatched_entries = [
        str(item.get("label", "<unknown>"))
        for item in review_entries
        if item.get("challenge_solution_statement_match") is not True
    ]
    if mismatched_entries:
        return fail("Challenge/Solution statement mismatch: " + ", ".join(mismatched_entries))

    if not AUDIT.exists():
        return fail("Missing comparator-challenge-audit.json; run the independent challenge audit")

    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if challenge_lock.get("status_date") != audit.get("status_date"):
        return fail(
            "Challenge lock status_date differs from independent audit; "
            "run scripts/generate_comparator_challenge_lock.py after updating the audit"
        )

    audit_entries = audit.get("entries", [])
    audit_labels = [item.get("label") for item in audit_entries]
    if audit_labels != expected_labels:
        return fail("Independent challenge audit labels do not match paper manifest labels")

    invalid_verdicts = [
        str(item.get("label", "<unknown>"))
        for item in audit_entries
        if item.get("verdict") not in VALID_AUDIT_VERDICTS
    ]
    if invalid_verdicts:
        return fail("Independent challenge audit has invalid verdicts: " + ", ".join(invalid_verdicts))

    audit_counts: dict[str, int] = {}
    for item in audit_entries:
        verdict = str(item.get("verdict"))
        audit_counts[verdict] = audit_counts.get(verdict, 0) + 1
    if audit.get("verdict_counts") != audit_counts:
        return fail("Independent challenge audit verdict_counts are stale")
    if audit_counts != {"faithful": len(expected_labels)}:
        return fail(
            "Independent challenge audit is not fully faithful: "
            + ", ".join(f"{key}={value}" for key, value in sorted(audit_counts.items()))
        )

    review_audit_mismatch = [
        str(item.get("label", "<unknown>"))
        for idx, item in enumerate(review_entries)
        if item.get("independent_audit_verdict") != audit_entries[idx].get("verdict")
    ]
    if review_audit_mismatch:
        return fail("Review independent-audit verdicts are stale: " + ", ".join(review_audit_mismatch))
    review_nonfaithful = [
        str(item.get("label", "<unknown>"))
        for item in review_entries
        if item.get("independent_audit_verdict") != "faithful"
    ]
    if review_nonfaithful:
        return fail(
            "Review independent-audit verdicts are not fully faithful: "
            + ", ".join(review_nonfaithful)
        )
    review_status_nonfaithful = [
        str(item.get("label", "<unknown>"))
        for item in review_entries
        if item.get("review_status") != "independent audit: faithful"
    ]
    if review_status_nonfaithful:
        return fail(
            "Review status fields are not fully faithful: "
            + ", ".join(review_status_nonfaithful)
        )
    stale_status_mentions: list[str] = []
    stale_phrases = stale_qualified_verdict_phrases()
    for path in relevant_repo_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if any(phrase in text for phrase in stale_phrases):
            stale_status_mentions.append(str(path.relative_to(REPO)))
    if stale_status_mentions:
        return fail(
            "Stale qualified-audit verdict wording found in: "
            + ", ".join(stale_status_mentions)
        )

    print(f"COMPARATOR SCAFFOLD CHECK PASSED ({len(expected)} theorem entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
