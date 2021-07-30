.PHONY: docs

clean: clean-build clean-pyc clean-test

clean-build: check-package
	rm -fr $(PACKAGE)/build/
	rm -fr $(PACKAGE)/dist/
	rm -fr $(PACKAGE)/.eggs/
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


docs:
	sphinx-apidoc -o docs/source catanatron_core/catanatron
	sphinx-build -b html docs/source/ docs/build/html