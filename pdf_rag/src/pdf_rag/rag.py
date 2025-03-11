"""THe RAG backbone of this MCP server"""

import sys
from pathlib import Path
from typing import List

import libsql_experimental as libsql
import tiktoken
from openai import OpenAI
from pypdf import PdfReader
from tqdm import tqdm
from typing_extensions import Tuple

VECTOR_COLLECTION_NAME = "document_chunks"
EMBEDDING_DIMENSIONS = 1024

QUERY_DROP = f"DROP TABLE IF EXISTS {VECTOR_COLLECTION_NAME}"

QUERY_CREATE = (
    f"CREATE TABLE IF NOT EXISTS {VECTOR_COLLECTION_NAME} ("
    f"  text TEXT,"
    f"  embedding F32_BLOB({EMBEDDING_DIMENSIONS})"
    f");"
)

QUERY_INSERT = (
    f"INSERT INTO {VECTOR_COLLECTION_NAME} (text, embedding) VALUES (?, vector32(?))"
)

QUERY_SEARCH = (
    "SELECT text, vector_distance_cos(embedding, vector32(?)) "
    f"  FROM {VECTOR_COLLECTION_NAME} "
    f"  ORDER BY vector_distance_cos(embedding, vector32(?)) ASC "
    f"  LIMIT 10;"
)


class RAG:
    """Basic document RAG system.

    Splits a document in chunks and later retrieves the most relevant chunks for a given query.
    """

    openai: OpenAI
    db_file: Path

    def __init__(self, db_path: str = "pdfsearch.sqlite"):
        self.openai = OpenAI()
        self.db_file = Path(db_path)

    def build_search_db(self):
        conn = libsql.connect(str(self.db_file.absolute()))
        conn.execute(QUERY_DROP)
        conn.execute(QUERY_CREATE)
        conn.commit()

    def add_knowledge(self, text: str, embedding: List[float]):
        conn = libsql.connect(str(self.db_file.absolute()))
        conn.execute(QUERY_INSERT, (text, str(embedding)))
        conn.commit()

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


def embed(text: str, client: OpenAI) -> list[float]:
    response = client.embeddings.create(
        input=text, model="text-embedding-3-small", dimensions=1024
    )
    return response.data[0].embedding


def embed_pdf(
    file_path: Path, should_split: bool = True
) -> List[Tuple[str, List[float]]]:
    pdf_text = ""
    with open(file_path, "rb") as pdf_file:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            pdf_text += page.extract_text(extraction_mode="plain")

    chunks = chunkify(text=pdf_text) if should_split else [pdf_text]
    embeddings = [embed(chunk, OpenAI()) for chunk in tqdm(chunks, desc=f"Embedding {file_path.name}")]

    return list(zip(chunks, embeddings))


# TODO rename or split, it's ambiguous as it is now
def embed_text(
    text: str,
    should_split: bool = True,
) -> List[Tuple[str, List[float]]]:
    chunks = chunkify(text=text) if should_split else [text]
    embeddings = [embed(chunk, OpenAI()) for chunk in tqdm(chunks, desc=f"Embedding {text[:25]}")]

    return list(zip(chunks, embeddings))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF document RAG system")
    parser.add_argument(
        "action",
        choices=["build", "search", "chunkify"],
        help="Action to perform",
    )
    parser.add_argument(
        "inputs", nargs="+", help="Input files/directories or a search query"
    )
    args = parser.parse_args()

    if args.action == "build":
        rag = RAG()
        rag.build_search_db()

        # Process each input path
        total_chunks = 0
        for input_path in args.inputs:
            path = Path(input_path)

            # Handle PDF file
            if path.is_file() and path.suffix.lower() == ".pdf":
                embeddings = embed_pdf(path)
                for chunk, embedding in embeddings:
                    rag.add_knowledge(chunk, embedding)
                total_chunks += len(embeddings)

            # Handle directory of text files
            elif path.is_dir():
                for text_file in tqdm(path.glob("*.txt"), desc=f"Embedding files in {path.name}"):
                    text = text_file.read_text()
                    embeddings = embed(text, OpenAI())
                    rag.add_knowledge(text, embeddings)
                    total_chunks += 1

            # Assume a single file in other text format - txt, md...
            else:
                text = path.read_text()
                embeddings = embed_text(text)
                for t, e in embeddings:
                    rag.add_knowledge(t, e)
                    total_chunks += 1

        print(f"Inserted {total_chunks} chunks of text")

    elif args.action == "search":
        rag = RAG()
        result = rag.search(args.inputs[0])

    elif args.action == "chunkify":
        pdf_text = ""
        with open(args.inputs[0], "rb") as pdf_file:
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                pdf_text += page.extract_text(extraction_mode="plain")

        chunks = chunkify(text=pdf_text)
