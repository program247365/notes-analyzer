import concurrent.futures
import logging
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor

import chromadb
import click
import requests

# Suppress ChromaDB's messages
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

BATCH_SIZE = 64  # Batch size for processing embeddings and ChromaDB operations
MAX_SEARCH_RESULTS = 100  # Maximum number of search results to return

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
@click.option("--web", is_flag=True, help="Open results in web browser")
def search(query, web):
    """
    Search the indexed notes for relevant documents matching the query.
    This command retrieves and displays relevant documents from ChromaDB.
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

    # Check if collection has any documents
    count = collection.count()
    if count == 0:
        click.echo("\nError: No documents found in the database.")
        click.echo(
            "Please run 'notes-analyzer index <directory>' "
            "to index your documents first."
        )
        return

    # Get an embedding for the query
    query_embedding = get_embedding(query)

    # Query ChromaDB for relevant documents
    click.echo("Querying the index...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(MAX_SEARCH_RESULTS, count),
        include=["metadatas", "documents", "distances"],
    )

    if not results or not results["documents"] or len(results["documents"]) == 0:
        msg = "\n🔍 No matching documents found for your query."
        if web:
            html_content = f"<h1>{msg}</h1>"
            _open_in_browser(html_content)
        else:
            click.secho(msg, fg="yellow")
        return

    if web:
        html_content = _generate_html_results(query, results)
        _open_in_browser(html_content)
    else:
        _display_cli_results(results)


def _generate_html_results(query, results):
    """Generate HTML content for search results."""
    # Dedupe results first
    deduped_results, duplicates = _dedupe_results(results)

    result_count = len(deduped_results)
    max_distance = max(r[2] for r in deduped_results) if deduped_results else 1.0

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Search Results: {query}</title>
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 
                    Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; 
                line-height: 1.6; 
                max-width: 800px; 
                margin: 0 auto; 
                padding: 20px; 
            }}
            .result {{ 
                border: 1px solid #eee; 
                padding: 20px; 
                margin: 20px 0; 
                border-radius: 8px; 
            }}
            .title {{ color: #2970ff; font-size: 1.2em; font-weight: bold; margin: 0; }}
            .relevance {{ color: #16a34a; font-size: 0.9em; margin: 5px 0; }}
            .preview {{ color: #374151; margin: 10px 0; }}
            .metadata {{ color: #6b7280; font-size: 0.9em; }}
            .bear-link {{ color: #6b7280; text-decoration: none; }}
            .bear-link:hover {{ text-decoration: underline; }}
            .header {{ margin-bottom: 30px; }}
            .tip {{ 
                background: #fef3c7; 
                padding: 10px; 
                border-radius: 6px; 
                margin-top: 20px; 
                color: #92400e; 
            }}
            .duplicate-link {{ 
                color: #6b7280;  /* Changed to match metadata color */
                font-size: 0.9em;
                margin-left: 10px;
                text-decoration: none;
            }}
            .duplicate-link:hover {{
                text-decoration: underline;
            }}
            .metadata-row {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 Found {result_count} unique documents</h1>
            <p>Search query: "{query}"</p>
        </div>
    """

    for doc, meta, distance in deduped_results:
        relevance = max(0, min(100, int(100 * (1 - (distance / max_distance)))))

        title = doc.split("\n")[0]
        if title.startswith("# "):
            title = title[2:]

        preview = doc.replace("\n", " ").strip()
        preview = preview[:150] + "..." if len(preview) > 150 else preview

        bear_url = f"bear://x-callback-url/open-note?title={title}&new_window=yes"

        html += f"""
        <div class="result">
            <h2 class="title">{title}</h2>
            <p class="relevance">Relevance: {relevance}%</p>
            <p class="preview">{preview}</p>
            <p class="metadata metadata-row">
                📝 {meta["filename"]}<br>
                <a href="{bear_url}" class="bear-link">🔗 Open in Bear</a>
                {
            _generate_duplicate_links_html(title, duplicates)
            if title in duplicates
            else ""
        }
            </p>
        </div>
        """

    html += """
        <div class="tip">
            💡 Tip: Use 'make ask "<prompt>"' to generate AI responses about 
            these topics
        </div>
    </body>
    </html>
    """

    return html


def _generate_duplicate_links_html(title, duplicates):
    """Generate HTML for duplicate links."""
    dupe_links = []
    for _, _, _ in duplicates[title]:
        bear_url = f"bear://x-callback-url/open-note?title={title}&new_window=yes"
        dupe_links.append(
            f'<a href="{bear_url}" class="duplicate-link">Duplicate note here</a>'
        )

    return f"• {', '.join(dupe_links)}" if dupe_links else ""


def _open_in_browser(html_content):
    """Write HTML content to temp file and open in browser."""
    import tempfile
    import webbrowser

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html") as f:
        f.write(html_content)
        webbrowser.open("file://" + f.name)


def _dedupe_results(results):
    """Dedupe results by title and track duplicates."""
    deduped = []
    duplicates = {}
    seen_titles = {}

    # Zip all result data together
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        strict=False,
    ):
        # Get title from document content
        title = doc.split("\n")[0]
        if title.startswith("# "):
            title = title[2:]

        if title in seen_titles:
            # Track duplicate
            if title not in duplicates:
                duplicates[title] = []
            duplicates[title].append((doc, meta, distance))
        else:
            # First occurrence
            seen_titles[title] = len(deduped)
            deduped.append((doc, meta, distance))

    return deduped, duplicates


def _display_cli_results(results):
    """Display results in CLI format."""
    # Dedupe results first
    deduped_results, duplicates = _dedupe_results(results)

    # Show search summary
    result_count = len(deduped_results)
    click.secho(f"\n🔍 Found {result_count} unique documents", fg="green", bold=True)
    click.secho("─" * 80, fg="bright_black")

    # Calculate max distance for normalization
    max_distance = max(r[2] for r in deduped_results) if deduped_results else 1.0

    for i, (doc, meta, distance) in enumerate(deduped_results):
        relevance = max(0, min(100, int(100 * (1 - (distance / max_distance)))))

        # Get title
        title = doc.split("\n")[0]
        if title.startswith("# "):
            title = title[2:]

        # Print result entry with styling
        click.echo()
        click.secho(f"{title}", fg="bright_blue", bold=True)

        # Show duplicate links if any exist
        if title in duplicates:
            dupe_links = []
            for _, dupe_meta, _ in duplicates[title]:
                dupe_links.append(
                    f"bear://x-callback-url/open-note?title={title}&new_window=yes"
                )
            dupe_text = "s" if len(dupe_links) > 1 else ""
            dupe_links_text = ", ".join(dupe_links)
            click.secho(
                f" (Duplicate note{dupe_text} here: {dupe_links_text})",
                fg="yellow",
                dim=True,
            )

        click.secho(f"Relevance: {relevance}%", fg="green", dim=True)

        # Preview text
        preview = doc.replace("\n", " ").strip()
        preview = preview[:150] + "..." if len(preview) > 150 else preview
        click.echo(preview)

        # Metadata and link
        click.secho(f"📝 {meta['filename']}", fg="bright_black")
        click.secho(
            f"🔗 bear://x-callback-url/open-note?title={title}&new_window=yes",
            fg="bright_black",
            underline=True,
        )

        if i < result_count - 1:
            click.secho("─" * 80, fg="bright_black", dim=True)

    # Footer
    click.echo()
    click.secho("─" * 80, fg="bright_black")
    click.secho(
        "💡 Tip: Use 'make ask \"<prompt>\"' to generate AI responses about\n"
        "    these topics",
        fg="yellow",
        dim=True,
    )


@cli.command()
@click.argument("prompt")
def ask(prompt):
    """
    Generate an answer to a prompt using the AI model.
    This command uses smollm2 to generate a response.
    """
    click.echo("Generating response...")
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
