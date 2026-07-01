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
    Associe les noeuds de type FunctionDef, ClassDef et autres blocs
    pertinents.
    Si un bloc dépasse max_chunk_size, il est découpé par chunks plus
    petits selon la méthode par defaut.
    """
    chunks = []

    try:
        tree = ast.parse(text)
    except Exception:
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
        if (
            hasattr(node, "lineno")
            and hasattr(node, "col_offset")
            and hasattr(node, "end_lineno")
            and hasattr(node, "end_col_offset")
        ):
            node_start = line_offsets[node.lineno - 1] + node.col_offset
            try:
                node_end = line_offsets[int(node.end_lineno or 0) - 1] + int(
                    node.end_col_offset or 0
                )
            except IndexError:
                node_end = len(text)

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
            pass

    if current_chunk_text:
        add_chunk(current_start, current_end)

    if not chunks:
        chunks.extend(chunk_text(file_path, text, max_chunk_size))

    return chunks


def chunk_text_by_size(
    file_path: str,
    text: str,
    start_limit: int,
    end_limit: int,
    max_chunk_size: int,
    overlap: int = 200,
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
        if next_idx == end_limit:
            break
        current_idx = max(current_idx + 1, next_idx - overlap)
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

    matches = list(re.finditer(r"\n{2,}", text))

    block_boundaries = []
    last_end = 0

    for m in matches:
        if m.start() > last_end:
            block_boundaries.append((last_end, m.start()))
        last_end = m.end()

    if last_end < len(text):
        block_boundaries.append((last_end, len(text)))

    block_states = {}
    h1, h2, h3 = "", "", ""
    for start_idx, end_idx in block_boundaries:
        block_text = text[start_idx:end_idx].strip()
        first_line = block_text.split("\n")[0]
        if first_line.startswith("# "):
            h1 = first_line[2:].strip()
            h2, h3 = "", ""
        elif first_line.startswith("## "):
            h2 = first_line[3:].strip()
            h3 = ""
        elif first_line.startswith("### "):
            h3 = first_line[4:].strip()
        block_states[start_idx] = (h1, h2, h3)

    def _create_chunk(start: int, end: int) -> Chunk:
        s_h1, s_h2, s_h3 = block_states.get(start, ("", "", ""))
        headers = [h for h in (s_h1, s_h2, s_h3) if h]
        prefix = ""
        if headers:
            prefix = "Context: " + " > ".join(headers) + "\n\n"
        chunk_text = prefix + text[start:end].strip()
        return Chunk(
            file_path=file_path,
            first_character_index=start,
            last_character_index=end,
            text=chunk_text,
        )

    current_start = 0
    current_end = 0
    from typing import Tuple

    blocks_in_chunk: List[Tuple[int, int]] = []

    for start_idx, end_idx in block_boundaries:
        block_len = end_idx - start_idx

        s_h1, s_h2, s_h3 = block_states.get(current_start, ("", "", ""))
        headers = [h for h in (s_h1, s_h2, s_h3) if h]
        if headers:
            prefix_len = len("Context: " + " > ".join(headers) + "\n\n")
        else:
            prefix_len = 0
        current_chunk_real_size = prefix_len + (end_idx - current_start)

        if (prefix_len + block_len) > max_chunk_size:
            if current_end > current_start:
                chunks.append(_create_chunk(current_start, current_end))

            safe_max_size = max_chunk_size - prefix_len
            sub_chunks = chunk_text_by_size(
                file_path,
                text,
                start_idx,
                end_idx,
                safe_max_size,
                overlap=300,
            )
            if headers:
                prefix = "Context: " + " > ".join(headers) + "\n\n"
                for sc in sub_chunks:
                    sc.text = prefix + sc.text

            chunks.extend(sub_chunks)
            current_start = end_idx
            current_end = end_idx
            blocks_in_chunk = []
            continue

        if current_chunk_real_size > max_chunk_size:
            chunks.append(_create_chunk(current_start, current_end))
            target_start = current_end - 300
            for b_start, b_end in blocks_in_chunk:
                if b_start >= target_start:
                    current_start = b_start
                    break
            else:
                current_start = start_idx

            blocks_in_chunk = [
                (b_s, b_e)
                for b_s, b_e in blocks_in_chunk
                if b_s >= current_start
            ]

        blocks_in_chunk.append((start_idx, end_idx))
        current_end = end_idx

    if current_end > current_start:
        chunk_str = text[current_start:current_end].strip()
        if chunk_str:
            chunks.append(_create_chunk(current_start, current_end))

    final_chunks = []
    for c in chunks:
        chunk_len = c.last_character_index - c.first_character_index
        if chunk_len > max_chunk_size:
            c.last_character_index = c.first_character_index + max_chunk_size
            c.text = c.text[:max_chunk_size]

        final_chunks.append(c)

    return final_chunks
