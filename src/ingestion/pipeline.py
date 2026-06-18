import os
from pathlib import Path
from typing import List
from tqdm import tqdm

from src.models import Chunk
from src.ingestion.chunker import chunk_python, chunk_markdown, chunk_text


def build_pipeline(repo_path: str, max_chunk_size: int = 2000) -> List[Chunk]:
    """
    Parcourt le dépôt, lit les fichiers et génère les chunks.

    Args:
        repo_path (str): Chemin vers le dossier à ingérer (ex: vllm-0.10.1)
        max_chunk_size (int): Taille maximale autorisée pour un chunk.

    Returns:
        List[Chunk]: Liste de tous les chunks générés.
    """
    all_chunks: List[Chunk] = []

    # Recueillir tous les chemins de fichiers que l'on veut parser
    file_paths = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            file_paths.append(os.path.join(root, file))

    print(f"🔍 Trouvé {len(file_paths)} fichiers dans {repo_path}. Début de l'ingestion...")

    for file_path in tqdm(file_paths, desc="Traitement des fichiers"):
        # Ignorer certains fichiers ou dossiers problématiques (ex: logs, pycache, .git)
        if ".git" in file_path or "__pycache__" in file_path:
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # Ignorer les binaire (.png, .so, etc.)
            continue
        except Exception as e:
            continue

        if not content.strip():
            continue

        # Découpage différencié
        ext = Path(file_path).suffix.lower()
        if ext == ".py":
            chunks = chunk_python(file_path, content, max_chunk_size=max_chunk_size)
        elif ext == ".md":
            chunks = chunk_markdown(file_path, content, max_chunk_size=max_chunk_size)
        else:
            # On ignore les autres types de fichiers pour l'instant (ou fallback textuel)
            # chunks = chunk_text(file_path, content, max_chunk_size=max_chunk_size)
            continue
            
        all_chunks.extend(chunks)

    print(f"✅ {len(all_chunks)} chunks générés avec succès.")
    return all_chunks
