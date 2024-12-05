.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-autobuild -b html --watch src/asgardpy/ docs/source/ docs/build/

.PHONY : run-checks
run-checks :
	isort --check src/
	black --check src/
	ruff check src/
	mypy src/
	codespell src/asgardpy
	CUDA_VISIBLE_DEVICES='' pytest -v --color=yes --doctest-modules tests/ src/asgardpy/
