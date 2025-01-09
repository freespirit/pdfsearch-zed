"""THe RAG backbone of this MCP server"""

import sys
from pathlib import Path
from typing import List

import tiktoken
from openai import OpenAI
from pypdf import PdfReader
from tqdm import tqdm

import libsql_experimental as libsql

VECTOR_COLLECTION_NAME = "document_chunks"
EMBEDDING_DIMENSIONS = 1024

QUERY_DROP = f"DROP TABLE IF EXISTS {VECTOR_COLLECTION_NAME}"

QUERY_CREATE = f"CREATE TABLE IF NOT EXISTS {VECTOR_COLLECTION_NAME} (" \
               f"  text TEXT," \
               f"  embedding F32_BLOB({EMBEDDING_DIMENSIONS})" \
               f");"

QUERY_INSERT = f"INSERT INTO {VECTOR_COLLECTION_NAME} (text, embedding) VALUES (?, vector32(?))"

QUERY_SEARCH = "SELECT text, vector_distance_cos(embedding, vector32(?)) " \
               f"  FROM {VECTOR_COLLECTION_NAME} " \
               f"  ORDER BY vector_distance_cos(embedding, vector32(?)) ASC " \
               f"  LIMIT 25;"


class RAG:
    """Basic document RAG system.

    Splits a document in chunks and later retrieves the most relevant chunks for a given query.
    """

    openai: OpenAI
    db_file: Path

    def __init__(self, pdf_file: str, db_path: str = "pdfsearch.sqlite"):
        self.pdf_file = pdf_file

        self.openai = OpenAI()
        self.db_file = Path(db_path)

    def build_search_db(self):
        pdf_text = ""
        with open(self.pdf_file, "rb") as pdf_file:
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                pdf_text += page.extract_text(extraction_mode="plain")

        chunks = chunkify(text=pdf_text)
        embeddings = [(embed(chunk, self.openai), chunk) for chunk in tqdm(chunks)]

        conn = libsql.connect(str(self.db_file.absolute()))

        conn.execute(QUERY_DROP)
        conn.execute(QUERY_CREATE)
        print(conn.commit())

        for i, (embedding, chunk) in tqdm(enumerate(embeddings)):
            params = (chunk, str(embedding))
            conn.execute(QUERY_INSERT, params)
        conn.commit()

        print(f"Inserted {len(embeddings)} chunks of text")

    def search(self, query: str) -> List[str]:
        embedding = embed(query, self.openai)

        conn = libsql.connect(str(self.db_file.absolute()))
        search_result = conn.execute(
            QUERY_SEARCH,
            (str(embedding), str(embedding)),
        ).fetchall()
        conn.commit()

        for row in search_result:
            print(row[0][:25], row[1])

        return [row[0] for row in search_result]


def chunkify(
        text: str, max_tokens: int = 512, overlap: int = 64, tokenizer=None
) -> List[str]:
    """Split text into token-based chunks with overlap."""
    if not text.strip():
        return []

    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    # Tokenize the entire text first
    tokenized_text = tokenizer.encode(text)
    num_tokens = len(tokenized_text)
    chunks = []

    # Iterate over the tokenized text in `max_tokens` increments with specified overlap
    start_idx = 0
    while start_idx < num_tokens:
        end_idx = min(start_idx + max_tokens, num_tokens)
        chunk_tokens = tokenized_text[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens)  # Decode the tokens back to text
        chunks.append(chunk_text)

        start_idx += max_tokens - overlap  # Move window forward with overlap

    return chunks


def embed(text: str, client: OpenAI):
    response = client.embeddings.create(
        input=text, model="text-embedding-3-small", dimensions=1024
    )
    return response.data[0].embedding


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else None

    if action == "build":
        pdf_file_path = sys.argv[2]
        rag = RAG(pdf_file_path)
        rag.build_search_db()
    elif action == "search":
        q = sys.argv[2]
        rag = RAG("")
        result = rag.search(q)
    elif action == "chunkify":
        pdf_doc_path = sys.argv[2]
        pdf_text = ""
        with open(pdf_doc_path, "rb") as pdf_file:
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                pdf_text += page.extract_text(extraction_mode="plain")

        chunks = chunkify(text=pdf_text)
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)
