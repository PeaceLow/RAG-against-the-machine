from abc import ABC, abstractmethod
from typing import List

from src.models import Chunk


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""

    @abstractmethod
    def index(self, chunks: List[Chunk]) -> None:
        """
        Index a list of chunks.

        Args:
            chunks (List[Chunk]): The text chunks to index.
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Chunk]:
        """
        Search for the top k chunks for a given query.

        Args:
            query (str): The search query.
            k (int, optional): The number of top chunks to return.
                               Defaults to 5.

        Returns:
            List[Chunk]: The top k chunks.
        """
        pass

    @abstractmethod
    def save(self, save_dir: str) -> None:
        """
        Save the index to disk.

        Args:
            save_dir (str): Directory where the index will be saved.
        """
        pass

    @abstractmethod
    def load(self, load_dir: str) -> None:
        """
        Load the index from disk.

        Args:
            load_dir (str): Directory where the index is saved.
        """
        pass
