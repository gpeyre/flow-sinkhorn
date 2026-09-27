# Article

`paper.tex` is the canonical, de-anonymized manuscript. It uses the standard
LaTeX `article` class, not a conference style. The main text is in `sections/`,
proofs and extensions in `appendices/`, and notation in `notation_section.tex`.
`figures/` contains exactly the nine PDFs included in the article.

## Build

From the repository root:

```sh
make -C paper
```

This requires a normal TeX installation with pdfLaTeX, BibTeX, and the packages
listed in the preamble (including TikZ, natbib, geometry, longtable, and booktabs).
It creates `paper/paper.pdf`. No Python, Lean, dataset, external draft, or network
access is needed to compile the article from a complete checkout.

## arXiv source package

```sh
make -C paper arxiv
```

The resulting `paper-arxiv-source.tar.gz` includes the main source, included
sections/appendices, notation, bibliography and generated `.bbl`, the actual
TikZ fragment, and all required PDF figures. It excludes build logs, old
presentation files, benchmarks, reviews, and audit material. Extract it and
compile `paper.tex` as the root document. This command does not submit to arXiv.

## Audits

- `../modifications.md`: reviewer-by-reviewer mathematical and editorial changes.
- `audit/bibliography.md`: primary-source checks and disposition of all 91 old records.
- `audit/bibliography_inventory.json`: preserved original bibliographic records and lookup evidence.
- `audit/check_revision.py`: seven independent numerical checks of the revised formulas;
  run with Python and NumPy. These are regression checks, not mathematical proofs.
- `../lean/PAPER_COVERAGE.md`: explicit limits of the historical Lean certificates.

The plot PDFs are retained historical CPU/GPU runs. Their error is a centered,
sign-aligned dual-potential diagnostic, not an L2 error on primal flows. Newer
benchmark defaults need not reproduce their exact gamma selection.
