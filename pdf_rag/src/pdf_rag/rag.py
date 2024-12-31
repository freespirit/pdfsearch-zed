"""THe RAG backbone of this MCP server"""

import sys
from typing import List

import tiktoken
from openai import OpenAI
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

VECTOR_COLLECTION_NAME = "Document Chunks"


class RAG:
    """Basic document RAG system.

    Splits a document in chunks and later retrieves the most relevant chunks for a given query.
    """

    openai: OpenAI
    qdrant: QdrantClient

    def __init__(self, pdf_file: str, qdrant_url: str):
        self.pdf_file = pdf_file

        self.openai = OpenAI()
        self.qdrant = QdrantClient(url=qdrant_url)

    def build_search_db(self):
        pdf_text = ""
        with open(self.pdf_file, "rb") as pdf_file:
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                pdf_text += page.extract_text(extraction_mode="plain")

        chunks = chunkify(text=pdf_text)
        embeddings = [(embed(chunk, self.openai), chunk) for chunk in tqdm(chunks)]

        if not self.qdrant.collection_exists(VECTOR_COLLECTION_NAME):
            self.qdrant.create_collection(
                collection_name=VECTOR_COLLECTION_NAME,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )

        vector_points = [
            PointStruct(id=i, vector=embedding, payload={"text": chunk})
            for i, (embedding, chunk) in tqdm(enumerate(embeddings))
        ]
        operation_info = self.qdrant.upsert(
            collection_name=VECTOR_COLLECTION_NAME,
            wait=True,
            points=vector_points,
        )

        print(f"Inserting {len(vector_points)} chunks of text: {operation_info.status}")

    def search(self, query: str) -> List[str]:
        embedding = embed(query, self.openai)

        search_result = self.qdrant.query_points(
            collection_name=VECTOR_COLLECTION_NAME,
            query=embedding,
            with_payload=True,
            limit=25,  # TODO make it configurable
        ).points

        return [point.payload["text"] for point in search_result]


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
        qdrant_url = sys.argv[3]
        rag = RAG(pdf_file_path, qdrant_url=qdrant_url)
        rag.build_search_db()
    elif action == "search":
        q = sys.argv[2]

        rag = RAG("", qdrant_url="http://localhost:6333")
        result = rag.search(q)
        for item in result:
            print(item)
            print("-----")
    elif action == "chunkify":
        pdf_doc_path = sys.argv[2]
        pdf_text = ""
        with open(pdf_doc_path, "rb") as pdf_file:
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                pdf_text += page.extract_text(extraction_mode="plain")

        chunks = chunkify(text=pdf_text)
        for chunk in chunks[:3]:
            print(chunk)
            print("-----")
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)
