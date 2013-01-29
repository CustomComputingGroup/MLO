rungui:
	./run_mlo.sh --gui

run_example1:
	./run_mlo.sh -f examples/artificial_continous_function/fitness_script.py -c examples/artificial_continous_function/configuration_script.py

run_example2:
	./run_mlo.sh -f examples/quadrature_method_based_app/fitness_script.py -c examples/quadrature_method_based_app/configuration_script.py

run_example3:
	./run_mlo.sh -f examples/reconfigurable_radio/fitness_script.py -c examples/reconfigurable_radio/configuration_script.py
    
pep:
	pep8 --statistic --exclude=pyXGPR,Old_Files .

.PHONY: presentation
presentation:
	pdflatex -output-directory=Presentation Presentation/presentation.tex
	pdflatex -output-directory=Presentation Presentation/presentation.tex

clean:
	find . -name "*.pyc" -delete
	find . -name "*~" -delete
    
    


