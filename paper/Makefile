.PHONY: all paper clean arxiv

all: paper

paper:
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	bibtex paper
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex

# The archive contains only the actual multi-file article dependencies.
arxiv: paper
	tar -czf paper-arxiv-source.tar.gz paper.tex paper.bbl references.bib notation_section.tex sections appendices blueprint/blueprint_diagram_tikz.tex figures

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc paper-arxiv-source.tar.gz
