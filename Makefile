.PHONY: help venv install install-dev test sync index ask search clean format

VENV_DIR=venv
PYTHON=$(VENV_DIR)/bin/python
PIP=$(VENV_DIR)/bin/pip
SYNCED_NOTES_DIR=synced-notes
NOTES_DB=$(HOME)/.notes_analyzer/chromadb
OLLAMA_MODEL=smollm2:135m

help:
	@echo "Available commands:"
	@echo "  make venv         - Create virtual environment"
	@echo "  make install     - Install package"
	@echo "  make install-dev - Install package with development dependencies"
	@echo "  make test        - Run tests"
	@echo "  make sync        - Sync Bear notes to synced-notes directory"
	@echo "  make index       - Index synced notes in ChromaDB"
	@echo "  make search      - Search your notes for relevant documents"
	@echo "  make ask         - Ask the AI to generate a response to your prompt"
	@echo "  make clean       - Remove generated files and directories"

venv:
	python3.11 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install .

install-dev: venv
	$(PIP) install -e '.[test]'

test: install-dev
	$(PYTHON) -m pytest

sync: 
	@if [ ! -d "$(SYNCED_NOTES_DIR)" ]; then \
		mkdir -p $(SYNCED_NOTES_DIR); \
	fi
	@echo "Syncing Bear notes to directory: $(shell pwd)/$(SYNCED_NOTES_DIR)"
	$(PYTHON) -m notes_analyzer sync

index:
	@echo "Note: indexing takes time... (4k notes was ~5 minutes with optimized processing)"
	@echo "Warning: This process uses significant memory and CPU. Close other applications first."
	@read -p "Do you want to continue? (y/n) " answer; \
	if [ "$$answer" != "y" ]; then \
		echo "Aborting."; \
		exit 1; \
	fi
	@if [ ! -d "$(SYNCED_NOTES_DIR)" ]; then \
		echo "Error: $(SYNCED_NOTES_DIR) directory not found. Run 'make sync' first."; \
		exit 1; \
	fi
	@echo "Indexing notes from directory: $(shell pwd)/$(SYNCED_NOTES_DIR)"
	$(PYTHON) -m notes_analyzer index $(SYNCED_NOTES_DIR)

init: install ollama-pull start-ollama sync index

ollama-pull:
	ollama pull $(OLLAMA_MODEL)

start-ollama:
	@if ! pgrep -x "ollama" > /dev/null; then \
		ollama serve & \
	else \
		echo "Ollama is already running"; \
	fi

search:
	@if [ ! -d "$(NOTES_DB)" ]; then \
		echo "Error: Database not found at $(NOTES_DB)"; \
		echo "Please run 'make sync' and 'make index' first to create and populate the index."; \
		exit 1; \
	fi
	@read -p "Enter your search query: " query; \
	read -p "Open results in web browser? (y/n): " web_answer; \
	if [ "$$web_answer" = "y" ]; then \
		$(PYTHON) -m notes_analyzer search "$$query" --web; \
	else \
		$(PYTHON) -m notes_analyzer search "$$query"; \
	fi

ask:
	@if [ ! -d "$(NOTES_DB)" ]; then \
		echo "Error: Database not found at $(NOTES_DB)"; \
		echo "Please run 'make sync' and 'make index' first to create and populate the index."; \
		exit 1; \
	fi
	@read -p "Enter your question: " question; \
	$(PYTHON) -m notes_analyzer ask "$$question"

clean:
	rm -rf $(VENV_DIR)
	rm -rf $(SYNCED_NOTES_DIR)
	rm -rf chromadb_db
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf __pycache__
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 

format:
	black .
	isort .
	$(PYTHON) -m ruff check . --fix
	$(PYTHON) -m ruff format .