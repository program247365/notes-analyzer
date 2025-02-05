# notes-analyzer

[![PyPI](https://img.shields.io/pypi/v/notes-analyzer.svg)](https://pypi.org/project/notes-analyzer/)
[![Changelog](https://img.shields.io/github/v/release/program247365/notes-analyzer?include_prereleases&label=changelog)](https://github.com/program247365/notes-analyzer/releases)
[![Tests](https://github.com/program247365/notes-analyzer/actions/workflows/test.yml/badge.svg)](https://github.com/program247365/notes-analyzer/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/program247365/notes-analyzer/blob/master/LICENSE)

> Using [smollm2](https://ollama.com/library/smollm2:135m) to interact with your [Bear notes](https://bear.app).

Search through and ask questions about your Bear notes using local AI models served by Ollama.

## Quickstart

Run the following:

```bash
make init # pulls model down, starts ollama, syncs your Bear notes to a folder, and then indexes them as embeddings into ChromaDB sqlite db
make search # search through your notes using semantic search
make ask # generate AI responses to your prompts
```

## Features

- **Semantic Search**: Use `make search` to find relevant notes based on meaning, not just keywords
- **AI Generation**: Use `make ask` to generate AI responses to your prompts
- **Bear Integration**: Automatically syncs with your Bear notes
- **Local Processing**: All processing happens on your machine using Ollama

## Requirements

- [Bear notes](https://bear.app) installed on your Mac
- [Ollama](https://ollama.com) installed on your Mac
- [Python](https://www.python.org) installed on your Mac
    - Python 3.11.1 is recommended

## Technology Used

- [smollm2](https://ollama.com/library/smollm2:135m) - a small language model for AI responses
- [nomic-embed-text](https://ollama.com/library/nomic-embed-text) - a small text embedding model for semantic search
- [ChromaDB](https://www.chromadb.dev/) - an open-source vector database to store and query embeddings
- [Ollama](https://ollama.com) - an open-source language model runner

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

Basic commands:
```bash
make sync   # Sync Bear notes to local directory
make index  # Create searchable index of your notes
make search # Search through your notes
make ask    # Generate AI responses to prompts
```

You can also use the Python module directly:
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
