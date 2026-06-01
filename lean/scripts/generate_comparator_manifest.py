#!/usr/bin/env python3
"""Generate Comparator manifest/config templates for paper-facing Lean statements."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEAN_DIR = REPO / "lean"
AUDIT_DIR = LEAN_DIR / "audit"
AUX = REPO / "neurips" / "paper.aux"
PAPER = REPO / "neurips" / "paper.tex"
STATEMENT_MAP = LEAN_DIR / "FlowSinkhorn" / "KLProjection" / "StatementMap.lean"
LEAN_ROOT = LEAN_DIR / "FlowSinkhorn" / "KLProjection"

LABEL_PREFIXES = ("prop:", "lem:", "thm:", "cor:", "app-prop:", "app-lem:", "app-cor:")


def parse_aux_labels(aux_text: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for line in aux_text.splitlines():
        m = re.search(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", line)
        if not m:
            continue
        label, number = m.group(1), m.group(2)
        if not label.startswith(LABEL_PREFIXES):
            continue
        base = label[4:] if label.startswith("app-") else label
        kind, rest = base.split(":", 1)
        canonical = rest.replace("-", "_")
        major, minor = number.split(".", 1)
        out.append(
            {
                "label": label,
                "number": number,
                "kind": kind,
                "canonical": canonical,
                "label_alias": f"{kind}_{canonical}",
                "numbered_alias": f"{kind}_{major}_{minor}",
            }
        )
    return out


def parse_paper_statement_labels(paper_text: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    env_re = re.compile(
        r"\\begin\{(theorem|proposition|lemma|corollary)\}"
        r"(?:\[([^\]]*)\])?(.*?)\\end\{\1\}",
        re.S,
    )
    for m in env_re.finditer(paper_text):
        env_kind = m.group(1)
        title = m.group(2) or ""
        body = m.group(3)
        line = paper_text[: m.start()].count("\n") + 1
        labels = [
            label
            for label in re.findall(r"\\label\{([^}]+)\}", body)
            if label.startswith(LABEL_PREFIXES)
        ]
        if not labels:
            raise SystemExit(
                f"paper theorem-like environment at neurips/paper.tex:{line} has no paper label"
            )
        for label in labels:
            out.append(
                {
                    "label": label,
                    "paper_source": f"neurips/paper.tex:{line}",
                    "paper_env_kind": env_kind,
                    "paper_title": title,
                }
            )
    return out


def check_aux_matches_paper(aux_labels: list[dict[str, str]], paper_labels: list[dict[str, str]]) -> None:
    aux_order = [item["label"] for item in aux_labels]
    paper_order = [item["label"] for item in paper_labels]
    if aux_order == paper_order:
        return

    missing_from_aux = [label for label in paper_order if label not in aux_order]
    missing_from_paper = [label for label in aux_order if label not in paper_order]
    details = ["paper.aux theorem labels are not synchronized with neurips/paper.tex"]
    if missing_from_aux:
        details.append("missing from paper.aux: " + ", ".join(missing_from_aux))
    if missing_from_paper:
        details.append("missing from paper.tex: " + ", ".join(missing_from_paper))
    if not missing_from_aux and not missing_from_paper:
        details.append("same labels but different order")
    raise SystemExit("\n".join(details))


def parse_statement_map(text: str) -> dict[str, str]:
    return dict(
        re.findall(r"^abbrev\s+([A-Za-z0-9_]+)\s*:=\s*@([A-Za-z0-9_.]+)", text, re.M)
    )


def find_decl(target: str) -> str | None:
    name = target.split(".")[-1]
    pat = re.compile(rf"^(theorem|lemma)\s+{re.escape(name)}\b", re.M)
    for fp in LEAN_ROOT.rglob("*.lean"):
        txt = fp.read_text(encoding="utf-8")
        m = pat.search(txt)
        if m:
            line = txt[: m.start()].count("\n") + 1
            return f"lean/{fp.relative_to(LEAN_DIR)}:{line}"
    return None


def main() -> int:
    if not AUX.exists():
        print(f"ERROR: missing {AUX}", file=sys.stderr)
        print(
            "Hint: regenerate it with "
            "`cd neurips && pdflatex -interaction=nonstopmode -halt-on-error paper.tex`.",
            file=sys.stderr,
        )
        return 2
    labels = parse_aux_labels(AUX.read_text(encoding="utf-8"))
    paper_labels = parse_paper_statement_labels(PAPER.read_text(encoding="utf-8"))
    check_aux_matches_paper(labels, paper_labels)
    aliases = parse_statement_map(STATEMENT_MAP.read_text(encoding="utf-8"))

    paper_metadata = {item["label"]: item for item in paper_labels}
    for item in labels:
        item.update(paper_metadata[item["label"]])
        target = aliases.get(item["label_alias"]) or aliases.get(item["numbered_alias"])
        item["target"] = target or "MISSING"
        item["source"] = find_decl(target) if target else None

    manifest = {
        "generated_from": (
            "neurips/paper.tex + neurips/paper.aux + "
            "lean/FlowSinkhorn/KLProjection/StatementMap.lean"
        ),
        "paper_tex_label_count": len(paper_labels),
        "challenge_module": "FlowSinkhorn.Comparator.Challenge",
        "solution_module": "FlowSinkhorn.Comparator.Solution",
        "permitted_axioms": ["propext", "Quot.sound", "Classical.choice"],
        "enable_nanoda": False,
        "theorem_names": [item["label_alias"] for item in labels],
        "paper_statements": labels,
    }

    (AUDIT_DIR / "comparator-paper-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    config = {
        key: manifest[key]
        for key in [
            "challenge_module",
            "solution_module",
            "theorem_names",
            "permitted_axioms",
            "enable_nanoda",
        ]
    }
    (AUDIT_DIR / "comparator-paper-config.template.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    print(f"wrote {len(labels)} paper statement entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
