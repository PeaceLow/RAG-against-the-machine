import os
import json
import re
from typing import List

# pyrefly: ignore [missing-import]
import bm25s

# pyrefly: ignore [missing-import]
import Stemmer

from src.models import Chunk
from src.retrieval.base import BaseRetriever


def tokenize_code(texts: List[str]) -> List[List[str]]:
    """
    Tokenize code while keeping original words, splitting snake_case,
    camelCase, applying English stemming, AND filtering stop words.
    """
    stemmer = Stemmer.Stemmer("english")

    # Liste de mots vides classiques en anglais
    STOP_WORDS = {
        "a", "about", "an", "and", "are", "as", "at", "be", "by", "for",
        "from", "how", "i", "in", "is", "it", "of", "on", "or", "that",
        "the", "this", "to", "was", "what", "when", "where", "who",
        "why", "will", "with", "can", "do", "does", "using", "use",
        "we", "you", "vllm", "python", "code", "example", "file",
        "error", "script", "please", "tell", "me", "show",
    }

    tokens_list = []
    for text in texts:
        base_tokens = re.split(r"[^\w.-]+", text)
        extended_tokens = []
        for t in base_tokens:
            if not t:
                continue

            # Ne pas étendre les stop words
            if t.lower() in STOP_WORDS:
                continue

            extended_tokens.append(t)
            if "_" in t:
                extended_tokens.extend(t.split("_"))

            # Split camelCase
            matches = re.finditer(
                r".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", t
            )
            camel_parts = [m.group(0) for m in matches]
            if len(camel_parts) > 1:
                extended_tokens.extend(camel_parts)

        # Stemming et lowercase, avec un dernier filtre de sécurité
        final_tokens = []
        for x in extended_tokens:
            lower_x = x.lower()
            if (
                len(lower_x) > 1
                and not lower_x.isdigit()
                and lower_x not in STOP_WORDS
            ):
                final_tokens.append(stemmer.stemWord(lower_x))

        tokens_list.append(final_tokens)
    return tokens_list


class BM25Retriever(BaseRetriever):
    """BM25 retrieval engine using the bm25s library."""

    def __init__(self) -> None:
        """Initialize the BM25 Retriever with optimized parameters for code."""
        # Actual perf : 89.69
        # Optimized for docs : k1 = 1.2, b=0.76
        self.retriever = bm25s.BM25(k1=1.2, b=0.76)
        self.chunks: List[Chunk] = []

    def index(self, chunks: List[Chunk]) -> None:
        """
        Index a list of chunks.
        """
        self.chunks = chunks

        corpus_texts = []
        for chunk in chunks:
            filename = chunk.file_path.split('/')[-1]
            boosted_metadata = f"{filename} " * 3
            super_text = f"{boosted_metadata} {chunk.file_path} {chunk.text}"
            corpus_texts.append(super_text)

        # Tokenize and index
        corpus_tokens = tokenize_code(corpus_texts)
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

        synonyms = {
            "error": "exception traceback",
            "config": "configuration setting parameter",
            "fake": "dummy mock test",
            "args": "arguments parameters kwargs",
            "init": "initialize setup __init__"
        }

        # On ajoute les synonymes à la volée si le mot est dans la requête
        expanded_query = query.lower()
        for word, syn in synonyms.items():
            if word in expanded_query:
                expanded_query += f" {syn}"

        # Custom tokenization for queries
        query_tokens = tokenize_code([expanded_query])

        # Ensure we don't ask for more chunks than we have
        k_min = min(k, len(self.chunks))

        # retrieve returns a matrix of dimensions (n_queries, k)
        docs, scores = self.retriever.retrieve(
            query_tokens, corpus=self.chunks, k=k_min
        )

        # We take the first query's result
        result_docs = (
            docs[0, :k_min].tolist()
            if hasattr(docs, "tolist")
            else list(docs[0])
        )

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
        with open(
            os.path.join(save_dir, "chunks.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(chunks_data, f)

    def load(self, load_dir: str) -> None:
        """
        Load the engine and chunks from disk.

        Args:
            load_dir (str): Directory from which to load.
        """
        # Load bm25s state (we skip token corpus loading
        # to handle our custom Chunks)
        self.retriever = bm25s.BM25.load(load_dir, load_corpus=False)

        chunks_path = os.path.join(load_dir, "chunks.json")
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        self.chunks = [Chunk(**data) for data in chunks_data]
