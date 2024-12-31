# PDF Search for Zed

A document search extension for Zed that lets you semantically search through a
PDF document and use the results in Zed's AI Assistant.

## Structure

As a context server extension for Zed, this project has two main parts:

-   the MCP server - a Python implementation in the `pdf_rag` directory
-   the Zed extension functionality - `src`, `extension.toml` and `Cargo.toml`

## Prerequisites

This extension currently requires:

1. A `Qdrant` vector database instance (for storing document embeddings)
2. An `OpenAI` API key (for generating embeddings)
3. `uv` installed on your system

**Note:** While the current setup is not trivial and it requires external
services, we plan to simplify this in future versions by implementing
self-contained alternatives for both vector storage and embeddings generation.
Community feedback will help prioritize these improvements.

## Quick Start

1. Clone the repository

```bash
git clone https://github.com/freespirit/pdfsearch-zed.git
```

2. Set up the Python environment for the MCP server:

```bash
cd pdfsearch-zed/pdf_rag
uv venv
uv sync
```

3. [Install Dev Extension](https://zed.dev/docs/extensions/developing-extensions) in Zed

4. Build the search db

```bash
cd /path/to/pdfsearch-zed/pdf_rag
export OPENAI_API_KEY=<your_openai_api_key>
# This may take a couple of minutes, depending on the document size
uv run src/pdf_rag/rag.py build /path/to/file.pdf <qdrant_url>
```

5. Configure Zed

```json
"context_servers": {
    "pdfsearch-context-server": {
        "settings": {
            "pdf_path": "/path/to/file.pdf",
            "extension_path": "/path/to/pdfsearch-zed",
            "qdrant_url": "http://localhost:6333",
            "openai_api_key": "sk-..."
        }
    }
}
```

## Usage

1. Open Zed's AI Assistant panel
2. Type `/pdfsearch` followed by your search query
3. The extension will search the PDF and add relevant sections to the AI
   Assistant's context

## Future Improvements

-   [ ] Self-contained vector store and embeddings (no external dependencies)
-   [ ] Automated index building on first run
-   [ ] Configurable result size
-   [ ] Support for multiple PDFs
-   [ ] Optional: Additional file formats beyond PDF

## Project Structure

-   `pdf_rag/`: Python-based MCP server implementation
-   `src/`: Zed extension code
-   `extension.toml` and `Cargo.toml`: Zed extension configuration files

## Known Limitations

-   Manual index building is required before first use
-   Currently supports only single PDF documents
-   Requires external services (Qdrant and OpenAI)

-   TBD: Why do we need to provide the document path in the extension settings
    if we build the DB manually?
