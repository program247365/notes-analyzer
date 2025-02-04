# notes-analyzer

[![PyPI](https://img.shields.io/pypi/v/notes-analyzer.svg)](https://pypi.org/project/notes-analyzer/)
[![Changelog](https://img.shields.io/github/v/release/program247365/notes-analyzer?include_prereleases&label=changelog)](https://github.com/program247365/notes-analyzer/releases)
[![Tests](https://github.com/program247365/notes-analyzer/actions/workflows/test.yml/badge.svg)](https://github.com/program247365/notes-analyzer/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/program247365/notes-analyzer/blob/master/LICENSE)

Using deepseek to ask questions of Bear notes

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
