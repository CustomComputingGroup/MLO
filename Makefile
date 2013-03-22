rungui:
	./run_mlo.sh --gui

run_example1:
	./run_mlo.sh -f examples/artificial_continous_function/fitness_script.py -c examples/artificial_continous_function/configuration_script.py

run_example2:
	./run_mlo.sh -f examples/quadrature_method_based_app/fitness_script.py -c examples/quadrature_method_based_app/configuration_script.py

run_example3:
	./run_mlo.sh -f examples/reconfigurable_radio/fitness_script.py -c examples/reconfigurable_radio/configuration_script.py

run_example4:
	./run_mlo.sh -f examples/xinyu_rtm/fitness_script.py -c examples/xinyu_rtm/configuration_script.py

install_gui:
	easy_install numpy
	easy_install scipy
	easy_install scikit-learn
	east_install matplotlib
	easy_install pisa
	easy_install pypdf
	easy_install wxPython pyt
	easy_install deap
	easy_install html5lib
	easy_isntall reportlab

    
install_terminal: 
	easy_install numpy
	easy_install scipy
	easy_install scikit-learn
	easy_install matplotlib
	easy_install pisa
	easy_install pypdf
	easy_install deap
	easy_install html5lib
	easy_isntall reportlab

restart:
	./run_mlo.sh --restart
    
pep:
	pep8 --statistic --exclude=pyXGPR,Old_Files .

.PHONY: presentation
presentation:
	pdflatex -output-directory=Presentation Presentation/presentation.tex
	pdflatex -output-directory=Presentation Presentation/presentation.tex

clean:
	rm profile
	find . -name "*.pyc" -delete
	find . -name "*~" -delete
    
    


