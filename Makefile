.PHONY: clean fix imports sort

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf condor_logs/*
	find run_scripts/* -delete
	find logs/* -delete
fix:
	black src common scripts_method scripts_data
sort:
	isort src common scripts_method scripts_data --wrap-length=1 --combine-as --trailing-comma --use-parentheses
imports:
	autoflake -i -r --remove-all-unused-imports src common scripts_method scripts_data
