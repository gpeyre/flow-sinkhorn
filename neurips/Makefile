.PHONY: all paper clean

all: paper

paper:
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	bibtex paper
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc
