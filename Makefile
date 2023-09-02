.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-autobuild -b html --watch asgardpy/ docs/source/ docs/build/

.PHONY : run-checks
run-checks :
	isort --check .
	black --check .
	ruff check .
	mypy .
	codespell src/asgardpy
	CUDA_VISIBLE_DEVICES='' pytest -v --color=yes --doctest-modules tests/ src/asgardpy/
