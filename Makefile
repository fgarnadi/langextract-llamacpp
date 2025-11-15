.PHONY: help sync sync-update check fmt
.DEFAULT_GOAL := help

help: ## display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

sync: ## sync dependencies of all workspace
	@uv sync --all-packages

sync-update: ## sync dependencies of all workspace with update
	@uv sync -U --all-packages

check: ## check for lint and format errors
	@ruff check || true
	@ruff format --check || true

fmt: ## fix lint and format 
	@ruff check --fix
	@ruff format