clean: clean-build clean-pyc clean-test

<<<<<<< HEAD
clean-build: check-package
	rm -fr $(PACKAGE)/build/
	rm -fr $(PACKAGE)/dist/
	rm -fr $(PACKAGE)/.eggs/
=======
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
>>>>>>> master
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	# find . -name '*~' -exec rm -f {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/

<<<<<<< HEAD
build: clean check-package
	cd $(PACKAGE) && python setup.py sdist bdist_wheel
	ls -l $(PACKAGE)/dist
	twine check $(PACKAGE)/dist/*

upload:
	twine upload --repository-url https://test.pypi.org/legacy/ $(PACKAGE)/dist/*

upload-production:
	twine upload $(PACKAGE)/dist/*


check-package:
ifndef PACKAGE
	$(error PACKAGE is undefined)
endif
=======
build-catanatron: clean
	python catanatron_core/setup.py sdist bdist_wheel
	ls -l dist
	twine check dist/*


build-catanatron-gym: clean
	python catanatron_gym/setup.py sdist bdist_wheel
	ls -l dist
	twine check dist/*

upload:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload-production:
	twine upload dist/*
>>>>>>> master
