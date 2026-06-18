import os
import json
from typing import List

import bm25s  # type: ignore

from src.models import Chunk
from src.retrieval.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """BM25 retrieval engine using the bm25s library."""

    def __init__(self) -> None:
        """Initialize the BM25 Retriever."""
        self.retriever = bm25s.BM25()
        self.chunks: List[Chunk] = []

    def index(self, chunks: List[Chunk]) -> None:
        """
        Index a list of chunks.

        Args:
            chunks (List[Chunk]): The text chunks to index.
        """
        self.chunks = chunks
        corpus_texts = [chunk.text for chunk in chunks]

        # Tokenize and index
        corpus_tokens = bm25s.tokenize(corpus_texts)
        self.retriever.index(corpus_tokens)

    def search(self, query: str, k: int = 5) -> List[Chunk]:
        """
        Search the indexed chunks for a query.

        Args:
            query (str): The search query.
            k (int, optional): The number of top chunks. Defaults to 5.

        Returns:
            List[Chunk]: The top k chunks retrieved.
        """
        if not self.chunks:
            raise ValueError("The index is empty. Please run indexing first.")

        # bm25s requires a list of queries for tokenization
        query_tokens = bm25s.tokenize([query])

        # Ensure we don't ask for more chunks than we have
        k_min = min(k, len(self.chunks))

        # retrieve returns a matrix of dimensions (n_queries, k)
        docs, scores = self.retriever.retrieve(
            query_tokens, corpus=self.chunks, k=k_min
        )

        # We take the first query's result
        result_docs = docs[0, :k_min].tolist() if hasattr(docs, "tolist") else list(docs[0])
        
        # We ensure they are valid chunks
        return [chunk for chunk in result_docs if isinstance(chunk, Chunk)]

    def save(self, save_dir: str) -> None:
        """
        Save the engine and chunks to disk.

        Args:
            save_dir (str): Directory where to save the files.
        """
        os.makedirs(save_dir, exist_ok=True)
        # bm25s saves the BM25 state
        self.retriever.save(save_dir)

        # Save chunks separately to preserve the Pydantic models structure
        chunks_data = [chunk.model_dump() for chunk in self.chunks]
        with open(os.path.join(save_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks_data, f)

    def load(self, load_dir: str) -> None:
        """
        Load the engine and chunks from disk.

        Args:
            load_dir (str): Directory from which to load.
        """
        # Load bm25s state (we skip token corpus loading to handle our custom Chunks)
        self.retriever = bm25s.BM25.load(load_dir, load_corpus=False)

        chunks_path = os.path.join(load_dir, "chunks.json")
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        self.chunks = [Chunk(**data) for data in chunks_data]
