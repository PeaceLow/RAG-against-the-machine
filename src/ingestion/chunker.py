import ast
import re
from typing import List

from src.models import Chunk
from src.config import MAX_CHUNK_SIZE


def _get_line_offsets(text: str) -> List[int]:
    """Helper to map line numbers to character indices."""
    offsets = [0]
    for m in re.finditer(r"\n", text):
        offsets.append(m.start() + 1)
    return offsets


def chunk_python(
    file_path: str, text: str, max_chunk_size: int = MAX_CHUNK_SIZE
) -> List[Chunk]:
    """
    Decoupe intelligemment un fichier Python en utilisant l'AST.
    Associe les noeuds de type FunctionDef, ClassDef et autres blocs pertinents.
    Si un bloc dépasse max_chunk_size, il est découpé par chunks plus petits selon la méthode par defaut.
    """
    chunks = []

    try:
        tree = ast.parse(text)
    except Exception:
        # En cas d'échec du parseur AST, utiliser une solution de repli (par ex: chunk texte par texte)
        return chunk_text(file_path, text, max_chunk_size)

    line_offsets = _get_line_offsets(text)

    current_chunk_text = ""
    current_start = 0
    current_end = 0

    def add_chunk(start_idx: int, end_idx: int) -> None:
        if end_idx <= start_idx:
            return

        chunk_text = text[start_idx:end_idx].strip()
        if not chunk_text:
            return

        if len(chunk_text) > max_chunk_size:
            # Fallback sur du text chunking simple si le noeud est gigantesque
            sub_chunks = chunk_text_by_size(
                file_path, text, start_idx, end_idx, max_chunk_size
            )
            chunks.extend(sub_chunks)
        else:
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=start_idx,
                    last_character_index=end_idx,
                    text=chunk_text,
                )
            )

    for node in tree.body:
        # Recuperer le start et end index si disponibles
        if (
            hasattr(node, "lineno")
            and hasattr(node, "col_offset")
            and hasattr(node, "end_lineno")
            and hasattr(node, "end_col_offset")
        ):
            node_start = line_offsets[node.lineno - 1] + node.col_offset
            # L'ast peut parfois donner un end_lineno plus loin que text len si c'est la fin du fichier
            try:
                node_end = line_offsets[int(node.end_lineno) - 1] + int(node.end_col_offset)  # type: ignore
            except IndexError:
                node_end = len(text)

            # Gérer la limite max_chunk_size en groupant les petits noeuds ou en les séparant
            node_length = node_end - node_start

            if current_chunk_text and (
                current_end
                - current_start
                + node_length
                + (node_start - current_end)
                > max_chunk_size
            ):
                add_chunk(current_start, current_end)
                current_start = node_start
                current_end = node_end
                current_chunk_text = text[current_start:current_end]
            else:
                if not current_chunk_text:
                    current_start = node_start
                current_end = node_end
                current_chunk_text = text[current_start:current_end]
        else:
            # Pour un noeud inattendu sans infos de lignes
            pass

    # Ajouter le reliquat
    if current_chunk_text:
        add_chunk(current_start, current_end)

    if not chunks:
        # S'il y a du texte non parsé/récupéré ou un code sans nodes reconnus
        chunks.extend(chunk_text(file_path, text, max_chunk_size))

    return chunks


def chunk_text_by_size(
    file_path: str,
    text: str,
    start_limit: int,
    end_limit: int,
    max_chunk_size: int,
) -> List[Chunk]:
    """Découpe un texte précis en morceaux d'une taille limite exacte."""
    chunks = []
    current_idx = start_limit
    while current_idx < end_limit:
        next_idx = min(current_idx + max_chunk_size, end_limit)
        chunk_text = text[current_idx:next_idx].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=current_idx,
                    last_character_index=next_idx,
                    text=chunk_text,
                )
            )
        current_idx = next_idx
    return chunks


def chunk_text(
    file_path: str, text: str, max_chunk_size: int = MAX_CHUNK_SIZE
) -> List[Chunk]:
    """Méthode de repli pour un texte sans logique particulière."""
    return chunk_text_by_size(file_path, text, 0, len(text), max_chunk_size)


def chunk_markdown(
    file_path: str, text: str, max_chunk_size: int = MAX_CHUNK_SIZE
) -> List[Chunk]:
    """
    Decoupe un fichier Markdown de maniere logique.
    Utilise les blocs séparés par '\n\n'.
    """
    chunks = []

    # Trouver tous les séparateurs de paragraphes
    matches = list(re.finditer(r"\n{2,}", text))

    block_boundaries = []
    last_end = 0

    for m in matches:
        if m.start() > last_end:
            block_boundaries.append((last_end, m.start()))
        last_end = m.end()

    if last_end < len(text):
        block_boundaries.append((last_end, len(text)))

    current_start = 0
    current_end = 0

    for start_idx, end_idx in block_boundaries:
        block_len = end_idx - start_idx

        # Si un seul bloc depasse la taille max
        if block_len > max_chunk_size:
            if current_end > current_start:
                chunks.append(
                    Chunk(
                        file_path=file_path,
                        first_character_index=current_start,
                        last_character_index=current_end,
                        text=text[current_start:current_end].strip(),
                    )
                )
            # On découpe ce très long bloc de façon naîve
            sub_chunks = chunk_text_by_size(
                file_path, text, start_idx, end_idx, max_chunk_size
            )
            chunks.extend(sub_chunks)
            current_start = end_idx
            current_end = end_idx
            continue

        # Si on depasse la limite en ajoutant le block courant
        if (end_idx - current_start) > max_chunk_size:
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=current_start,
                    last_character_index=current_end,
                    text=text[current_start:current_end].strip(),
                )
            )
            current_start = start_idx

        current_end = end_idx

    if current_end > current_start:
        chunk_str = text[current_start:current_end].strip()
        if chunk_str:
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=current_start,
                    last_character_index=current_end,
                    text=chunk_str,
                )
            )

    return chunks
