from setuptools import setup

setup(
    name="notes-analyzer",
    version="0.1.0",
    packages=["notes_analyzer"],
    install_requires=[
        "click",
        "chromadb",
        "requests",
        "markdown",
    ],
    entry_points={
        "console_scripts": [
            "notes-analyzer=notes_analyzer.cli:cli",
        ],
    },
    extras_require={
        "test": [
            "pytest",
            "black",
            "isort",
            "ruff",
        ],
    },
)
