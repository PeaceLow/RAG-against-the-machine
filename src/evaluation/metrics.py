from typing import List
from src.models import MinimalSource, Chunk


def _normalize_path(path: str) -> str:
    """Conserve uniquement la partie utile après vllm-0.10.1/."""
    try:
        # Trouver 'vllm-0.10.1/' et couper ce qui est avant
        idx = path.index("vllm-0.10.1/")
        return path[idx:]
    except ValueError:
        pass

    # Au cas où, on nettoie au moins data/raw/
    return path.replace("data/raw/", "")


def compute_overlap_percentage(
    ground_truth: MinimalSource, retrieved: Chunk
) -> float:
    """
    Calcule le pourcentage de chevauchement d'un chunk récupéré sur une source
    ground truth.
    Le pourcentage est par rapport à la taille de la source ground truth.
    """
    gt_path = _normalize_path(ground_truth.file_path)
    ret_path = _normalize_path(retrieved.file_path)

    if gt_path != ret_path:
        return 0.0

    gt_start = ground_truth.first_character_index
    gt_end = ground_truth.last_character_index
    gt_length = gt_end - gt_start

    if gt_length <= 0:
        return 0.0

    ret_start = retrieved.first_character_index
    ret_end = retrieved.last_character_index

    # Calcul de l'intersection des deux intervalles
    overlap_start = max(gt_start, ret_start)
    overlap_end = min(gt_end, ret_end)
    overlap_length = max(0, overlap_end - overlap_start)

    return overlap_length / gt_length


def compute_recall_at_k_for_question(
    ground_truth_sources: List[MinimalSource],
    retrieved_chunks: List[Chunk],
    k: int,
) -> float:
    """
    Calcule le Recall pour une seule question, en vérifiant si chaque source GT
    est partiellement couverte (>= 5%) par au moins un des top-k chunks.
    """
    if not ground_truth_sources:
        return 0.0

    top_k_chunks = retrieved_chunks[:k]
    found_count = 0

    for gt in ground_truth_sources:
        for chunk in top_k_chunks:
            if compute_overlap_percentage(gt, chunk) >= 0.05:
                found_count += 1
                break  # Source trouvée, on passe à la suivante

    return found_count / len(ground_truth_sources)
