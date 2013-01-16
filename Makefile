rungui:
	./run_mlo.sh --gui

runterminal:
	./run_mlo.sh -f fitness_script.py -c configuration_script.py

pep:
	pep8 --statistic --exclude=pyXGPR,Old_Files .

.PHONY: presentation
presentation:
	pdflatex -output-directory=Presentation Presentation/presentation.tex
	pdflatex -output-directory=Presentation Presentation/presentation.tex

clean:
	find . -name "*.pyc" -delete

