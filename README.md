# notes-analyzer

[![PyPI](https://img.shields.io/pypi/v/notes-analyzer.svg)](https://pypi.org/project/notes-analyzer/)
[![Changelog](https://img.shields.io/github/v/release/program247365/notes-analyzer?include_prereleases&label=changelog)](https://github.com/program247365/notes-analyzer/releases)
[![Tests](https://github.com/program247365/notes-analyzer/actions/workflows/test.yml/badge.svg)](https://github.com/program247365/notes-analyzer/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/program247365/notes-analyzer/blob/master/LICENSE)

> Using [smollm2](https://ollama.com/library/smollm2:135m) to ask questions of my [Bear notes](https://bear.app).

## Requirements

- [Bear notes](https://bear.app) installed on your Mac
- [Ollama](https://ollama.com) installed on your Mac
- [Python](https://www.python.org) installed on your Mac
    - Python 3.11.1 is recommended

## Technology Used

- [smollm2](https://ollama.com/library/smollm2:135m) - a small language model to ask questions of Bear notes
- [nomic-embed-text](https://ollama.com/library/nomic-embed-text) - a small text embedding model to embed Bear notes
- [ChromaDB](https://www.chromadb.dev/) - an open-source vector database to store and query embeddings
- [Ollama](https://ollama.com) - an open-source language model

## Installation

Install this tool using `pip`:
```bash
pip install notes-analyzer
```
## Usage

For help, run:
```bash
notes-analyzer --help
```
You can also use:
```bash
python -m notes_analyzer --help
```
## Development

To contribute to this tool, first checkout the code. Then create a new virtual environment:
```bash
cd notes-analyzer
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
pip install -e '.[test]'
```
To run the tests:
```bash
python -m pytest
```
