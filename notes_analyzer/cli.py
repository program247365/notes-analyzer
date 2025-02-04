import concurrent.futures
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor

import chromadb
import click
import requests

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

BATCH_SIZE = 64  # Batch size for processing embeddings and ChromaDB operations

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------


def load_markdown_files(directory):
    """Read all markdown files in the directory and return a list of their contents."""
    texts = []
    filenames = []

    # First count total files for progress
    total_files = sum(
        1 for root, _, files in os.walk(directory) for f in files if f.endswith(".md")
    )

    with click.progressbar(
        length=total_files,
        label="Loading files",
        show_percent=True,
        show_pos=True,
        width=40,
        show_eta=True,
        item_show_func=lambda x: x if x else "",
    ) as bar:
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".md"):
                    filepath = os.path.join(root, filename)
                    relative_path = os.path.relpath(filepath, directory)

                    with open(filepath, "r", encoding="utf-8") as f:
                        texts.append(f.read())
                        filenames.append(relative_path)

                    bar.update(1, relative_path)

    click.echo(f"Found {len(texts)} files.")
    return texts, filenames


def get_embedding(text):
    """Use the nomic-embed-text model via Ollama to create an embedding for text."""
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text},
    )
    if response.status_code != 200:
        raise Exception(f"Failed to get embedding: {response.text}")
    return response.json()["embedding"]


def generate_answer(text):
    """Generate an answer using smollm2 via Ollama."""
    system_prompt = """You are a helpful research assistant. When answering questions:
    1. Be concise and direct
    2. If the context doesn't contain relevant information, say so
    3. Use bullet points when appropriate
    4. Always base your answers on the provided context
    5. If you're unsure, express your uncertainty
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "smollm2:135m",  # https://ollama.com/library/smollm2:135m
            "prompt": text,
            "system": system_prompt,
            "stream": False,
        },
    )
    if response.status_code != 200:
        raise Exception(f"Failed to generate answer: {response.text}")
    return response.json()["response"]


def get_db_path():
    """Get the absolute path to the ChromaDB database."""
    home = os.path.expanduser("~")
    db_path = os.path.join(home, ".notes_analyzer", "chromadb")
    os.makedirs(db_path, exist_ok=True)
    return db_path


def get_embeddings_batch(texts, batch_size=32):
    """Get embeddings for a batch of texts one at a time."""
    embeddings = []

    for text in texts:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": text,  # Send single text instead of array
            },
        )
        if response.status_code != 200:
            raise Exception(f"Failed to get embeddings: {response.text}")
        embeddings.append(response.json()["embedding"])

    return embeddings


def load_file(filepath):
    """Load a single file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# ------------------------------------------------------------------------------
# CLI group and commands
# ------------------------------------------------------------------------------


@click.group()
@click.version_option()
def cli():
    """Using nomic-embed-text and smollm2 via Ollama to ask questions of Bear notes"""


@cli.command()
@click.argument("notes_dir", type=click.Path(exists=True))
def index(notes_dir):
    """
    Extract markdown files from NOTES_DIR,
    compute embeddings with nomic-embed-text, and store them in ChromaDB.
    """
    click.echo(f"Loading markdown files from {notes_dir}...")

    # Collect all markdown files
    markdown_files = []
    for root, _, files in os.walk(notes_dir):
        for filename in files:
            if filename.endswith(".md"):
                filepath = os.path.join(root, filename)
                relative_path = os.path.relpath(filepath, notes_dir)
                markdown_files.append((filepath, relative_path))

    # Load files in parallel
    cpu_count = multiprocessing.cpu_count()
    thread_count = min(32, cpu_count * 2)  # Cap at 32 threads

    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        with click.progressbar(
            length=len(markdown_files),
            label="Loading files",
            show_percent=True,
            show_pos=True,
            width=40,
            show_eta=True,
        ) as bar:
            futures = []
            for filepath, _ in markdown_files:
                future = executor.submit(load_file, filepath)
                futures.append(future)

            texts = []
            for future in concurrent.futures.as_completed(futures):
                texts.append(future.result())
                bar.update(1)

    filenames = [f[1] for f in markdown_files]
    click.echo(f"Loaded {len(texts)} files.")

    # Create embeddings using batched requests
    click.echo("\nGenerating embeddings for your notes...")
    embeddings = []

    with click.progressbar(
        range(0, len(texts), BATCH_SIZE),
        label="Processing batches",
        show_percent=True,
        show_pos=True,
        width=40,
        show_eta=True,
    ) as bar:
        for i in bar:
            batch_texts = texts[i : i + BATCH_SIZE]
            batch_embeddings = get_embeddings_batch(batch_texts)
            embeddings.extend(
                list(
                    zip(
                        range(i, i + len(batch_texts)),
                        batch_embeddings,
                        batch_texts,
                        filenames[i : i + BATCH_SIZE],
                        strict=False,
                    )
                )
            )

    # Initialize ChromaDB
    click.echo("\nInitializing ChromaDB...")
    client = chromadb.PersistentClient(path=get_db_path())
    collection = client.get_or_create_collection("notes")

    # Store in larger batches
    click.echo("Storing documents in ChromaDB...")
    batch_count = (len(embeddings) + BATCH_SIZE - 1) // BATCH_SIZE

    with click.progressbar(
        range(0, len(embeddings), BATCH_SIZE),
        length=batch_count,
        label="Storing batches",
        show_percent=True,
        show_pos=True,
        width=40,
        show_eta=True,
    ) as bar:
        for start_idx in bar:
            end_idx = min(start_idx + BATCH_SIZE, len(embeddings))
            batch = embeddings[start_idx:end_idx]

            collection.add(
                ids=[str(i) for i, _, _, _ in batch],
                embeddings=[emb for _, emb, _, _ in batch],
                documents=[text for _, _, text, _ in batch],
                metadatas=[{"filename": filename} for _, _, _, filename in batch],
            )

    click.echo("\nIndexing complete! Your notes are now stored in ChromaDB.")


@cli.command()
@click.argument("query")
def ask(query):
    """
    Query the indexed notes with a question.
    This command retrieves relevant documents from ChromaDB and
    then uses mistral to generate an answer.
    """
    # Initialize the persistent ChromaDB client
    db_path = get_db_path()
    if not os.path.exists(db_path):
        click.echo(f"\nError: Database not found at {db_path}")
        click.echo(
            "Please run 'notes-analyzer index <directory>' first to create the index."
        )
        return

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection("notes")

    # Debug: Check if collection has any documents
    count = collection.count()
    if count == 0:
        click.echo("\nError: No documents found in the database.")
        click.echo(
            "Please run 'notes-analyzer index <directory>' "
            "to index your documents first."
        )
        return

    # First, get an embedding for the query
    click.echo(f"Found {count} documents in the database.")
    query_embedding = get_embedding(query)

    # click.echo(f"Query embedding: {query_embedding}")

    # Query ChromaDB for the top 3 relevant documents
    click.echo("Querying the index...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(3, count),  # Don't try to get more results than we have documents
        include=["metadatas", "documents", "distances"],  # Added 'documents' to include
    )

    # Debug output
    # click.echo(f"Results: {results}")

    if not results or not results["documents"] or len(results["documents"]) == 0:
        click.echo("\nNo relevant documents found for your query.")
        return

    # Show detailed results
    click.echo("\nMatched documents:")
    for i, (doc, meta, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=False,
        )
    ):
        click.echo(f"\n{i + 1}. {meta['filename']} (similarity: {1 - distance:.3f})")
        # Show a preview of the document (first 100 chars)
        preview = doc[:100] + "..." if len(doc) > 100 else doc
        click.echo(f"Preview: {preview}")

    # Continue with combining documents for context
    retrieved_docs = results["documents"][0]
    retrieved_files = [m["filename"] for m in results["metadatas"][0]]

    click.echo("\nFound relevant content in these files:")
    for filename in retrieved_files:
        click.echo(f"- {filename}")

    context = "\n\n".join(retrieved_docs)

    # Use the context in the prompt for generating an answer
    prompt = (
        f"Based on the following context, answer the question:\n\n"
        f"{context}\n\nQuestion: {query}"
    )
    answer = generate_answer(prompt)
    click.echo("\nAnswer:\n")
    click.echo(answer)


@cli.command()
def sync():
    """Sync Bear notes to the synced-notes directory."""
    from . import sync

    output_dir = "synced-notes"

    click.echo("Syncing Bear notes...")
    try:
        sync.sync(output_dir)
        click.echo("Done!")
    except FileNotFoundError as e:
        click.echo(f"Error: {e}")
        click.echo("Make sure Bear app is installed and accessible.")
    except Exception as e:
        click.echo(f"Error syncing notes: {e}")
