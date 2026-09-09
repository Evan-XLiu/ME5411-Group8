LATEXMK ?= latexmk

.PHONY: pdf watch clean distclean

pdf:
	cd report && $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

watch:
	cd report && $(LATEXMK) -pdf -pvc -interaction=nonstopmode main.tex

clean:
	cd report && $(LATEXMK) -c main.tex

distclean:
	cd report && $(LATEXMK) -C main.tex

