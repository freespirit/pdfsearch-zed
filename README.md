# About

A document search (vector-based) MCP server extension for Zed.

# Structure

As a context server extension for Zed, this project has two main parts:

-   the MCP server - a Python implementation in the `pdf_rag` directory
-   the Zed extension functionality - `src`, `extension.toml` and `Cargo.toml`

# Prerequisites / Dependencies

In this initial version, the extension relies on `Qdrant` as a vectore store and
`OpenAI` for embeddings. This means you need:

-   a Qdrant instance and its url
-   an OpenAI API key

Unfortunately, this means the extension setup is not trivial yet. But if it
turns out to be a useful tool, there are options to be explored for both the
vector store and the embeddings.

# Installation

-   `git clone https://github.com/freespirit/pdfsearch-zed.git`
-   setup the environment of the MCP server:

```bash
cd /path/to/pdfsearch-zed # the local clone of the extension
cd pdf_rag
uv venv
uv sync
```

-   [Install Dev Extension](https://zed.dev/docs/extensions/developing-extensions) in Zed

# Usage

-   build the pdf search store
    -   TBD: Why do we need to provide the document path in the extension settings
        if we build the DB manually?

```bash
cd /path/to/pdfsearch-zed/pdf_rag
export OPENAI_API_KEY=...
# Here your QDrant instance should be running at localhost:6333
# This would split your document into chunks, embed them via OpenAI and
# store them in the Qdrant instance. It may take a couple of minutes,
# depending on the document size
uv run src/pdf_rag/rag.py build /path/to/file.pdf <qdrant_url>
```

-   add an entry in Zed's `context_servers` settings

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

-   In the Zed's AI Assistant panel, use `/pdfsearch` and some query to search
    the document and add the relevant information to the context.

# TODOs and future plans

-   make the extension self-sufficient by default
    -   search for built-in vector store
    -   search for built-in embeddings model (consider performance and
        dependencies)
-   make it more configurable
    -   length of returned context/string, it's currently fixed
    -   length of document chunks
-   other features
    -   configurable size of the result
        -   number of retrieved chunks
        -   size in characters or tokens of the returned text
    -   more than 1 document
    -   alternative document sources
        -   other file types
        -   web pages?

# Known issues

-   the vector DB is built manually and is a required step. It might be better
    to offload this to the extension's first run.
