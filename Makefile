.PHONY: init clean data symlink lint requirements sync_raw_data env test

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
SHARED_IO = /io

#################################################################################
# COMMANDS                                                                      #
#################################################################################

# Initialize project (requirements + io/ folder)
init: requirements symlink

# Create a symlink from the shared folder to ./io/
symlink:
	@/bin/sh -c "if [ ! -d ./io ]; then \
			ln -s ${SHARED_IO} .; \
		fi"

## Install Python dependencies
requirements:
ifdef VIRTUAL_ENV
	python3 -m pip install -U pip setuptools wheel
	python3 -m pip install -r requirements.txt
	python3 -m pip install -e .
else
	@echo "Please create your virtual environment and activate it first (make env; source env/bin/activate)."
endif

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
ifdef VIRTUAL_ENV
	flake8 --per-file-ignores="__init__.py:F401" src
else
	@echo "Please create your virtual environment and activate it first (make env; source env/bin/activate)."
endif

## Sync data from Google Bucket
sync_raw_data:
	#TODO: aws s3 sync s3://$(BUCKET)/data/ /io/data/raw

## Set up Python environment
env:
ifndef VIRTUAL_ENV
	python3 -m venv env
else
	@echo "You are already in a virtual environment."
endif

# Run unit tests on src/ folder
test:
ifdef VIRTUAL_ENV
	coverage run --source=src setup.py test
	coverage report
else
	@echo "Please create your virtual environment and activate it first (make env; source env/bin/activate)."
endif

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
