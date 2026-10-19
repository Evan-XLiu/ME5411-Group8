LATEXMK ?= latexmk

.PHONY: pdf watch clean distclean

pdf:
	cd report && $(LATEXMK) -xelatex -interaction=nonstopmode -halt-on-error main.tex

watch:
	cd report && $(LATEXMK) -xelatex -pvc -interaction=nonstopmode main.tex

clean:
	cd report && $(LATEXMK) -c main.tex

distclean:
	cd report && $(LATEXMK) -C main.tex

