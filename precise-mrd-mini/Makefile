.PHONY: setup fmt lint test demo clean

setup:
	conda env create -f environment.yml || conda env update -f environment.yml
	pre-commit install

fmt:
	black src tests
	isort src tests

lint:
	black --check src tests
	isort --check-only src tests
	mypy src/mrd

test:
	pytest

demo:
	snakemake -j 2 --directory workflow

clean:
	rm -rf tmp/* reports/auto_report.md reports/auto_report.html reports/figures/*.png reports/figures/*.csv simulated_reads data/ground_truth/variants.csv
