import os
import json

# pyrefly: ignore [missing-import]
import fire
import time

from tqdm import tqdm

from src.ingestion.pipeline import build_pipeline
from src.retrieval.bm25_engine import BM25Retriever
from src.config import MAX_CHUNK_SIZE
from src.models import (
    RagDataset,
    StudentSearchResults,
    MinimalSearchResults,
    AnsweredQuestion,
    StudentSearchResultsAndAnswer,
    MinimalAnswer,
    Chunk,
)
from src.evaluation.metrics import compute_recall_at_k_for_question
from src.generation.llm_client import LLMClient

INDEX_SAVE_DIR = "data/processed/bm25_index"


class RAGCLI:
    """CLI principale pour le système RAG."""

    def index(
        self,
        repo_path: str = "vllm-0.10.1",
        max_chunk_size: int = MAX_CHUNK_SIZE,
    ) -> None:
        """
        Indexe le dépôt de code.

        Args:
            repo_path (str): Le chemin vers le dépôt à indexer.
            max_chunk_size (int): Taille maximale d'un chunk.
        """
        start_time = time.time()

        # 1. Pipeline d'ingestion (découpage)
        chunks = build_pipeline(repo_path, max_chunk_size=max_chunk_size)

        if not chunks:
            print(
                f"\033[91m\033[1mErreur\033[0m : Aucun chunk généré à partir de '{repo_path}'."  # noqa: E501
            )
            return

        # 2. Indexation BM25
        print(
            "\033[96m\033[1mIndexation avec BM25\033[0m (Cela peut prendre quelques instants)..."  # noqa: E501
        )
        retriever = BM25Retriever()
        retriever.index(chunks)

        # 3. Sauvegarde
        retriever.save(INDEX_SAVE_DIR)

        elapsed = time.time() - start_time
        print(
            f"\033[92m\033[1mIndexation terminée\033[0m et sauvegardée dans '{INDEX_SAVE_DIR}'. "  # noqa: E501
            f"Temps écoulé : {elapsed:.2f} secondes."
        )

    def search(self, query: str, k: int = 5) -> None:
        """
        Effectue une recherche textuelle rapide.

        Args:
            query (str): La requête ou question de l'utilisateur.
            k (int): Le nombre de chunks à retourner (Defaut: 5).
        """
        if not os.path.exists(INDEX_SAVE_DIR):
            print(
                "\033[91m\033[1mErreur\033[0m : L'index n'existe pas. Veuillez exécuter "  # noqa: E501
                "'python -m src.main index' au préalable."
            )
            return

        # Chargement
        print("\033[93m\033[1mChargement du moteur de recherche...\033[0m")
        start_time = time.time()
        retriever = BM25Retriever()
        retriever.load(INDEX_SAVE_DIR)
        print(
            f"\033[92m\033[1mMoteur chargé\033[0m en {time.time() - start_time:.2f}s"  # noqa: E501
        )

        # Recherche
        print(f"\033[96m\033[1mRecherche pour\033[0m : '{query}'...")
        results = retriever.search(query, k=k)

        print("\n" + "=" * 50)
        print(" RÉSULTATS DE RECHERCHE")
        print("=" * 50)
        for i, chunk in enumerate(results, 1):
            print(f"\n[{i}] Fichier : {chunk.file_path}")
            print(
                f"    Positions : {chunk.first_character_index} - "
                f"{chunk.last_character_index}"
            )
            text_preview = chunk.text[:200].replace("\n", " ")
            print(f"    Aperçu : {text_preview}...")

    def evaluate(self, dataset_path: str, k: int = 5) -> None:
        """
        Évalue le Recall@k sur un jeu de données (JSON).

        Args:
            dataset_path (str): Chemin vers le fichier JSON du dataset
                                (ground truth).
            k (int): Le k pour calculer le Recall@k.
        """
        if not os.path.exists(INDEX_SAVE_DIR):
            print(
                "\033[91m\033[1mErreur\033[0m : L'index n'existe pas. Veuillez exécuter "  # noqa: E501
                "'python -m src.main index' au préalable."
            )
            return

        print("\033[93m\033[1mChargement du moteur de recherche...\033[0m")
        retriever = BM25Retriever()
        retriever.load(INDEX_SAVE_DIR)

        print(
            f"\033[96m\033[1mChargement du dataset\033[0m depuis {dataset_path}..."  # noqa: E501
        )
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            dataset = RagDataset(**data)
        except Exception as e:
            print(
                f"\033[91m\033[1mErreur\033[0m lors de l'ouverture du dataset : {e}"  # noqa: E501
            )
            return

        total_recall = 0.0
        question_count = 0
        valid_q_count = 0

        doc_recall = 0.0
        doc_count = 0

        code_recall = 0.0
        code_count = 0

        print(
            f"\033[93m\033[1mDébut de l'évaluation\033[0m sur {len(dataset.rag_questions)} "  # noqa: E501
            f"questions (Recall@{k})..."
        )
        for q in tqdm(dataset.rag_questions, desc="Évaluation"):
            if not isinstance(q, AnsweredQuestion) or not q.sources:
                continue

            results = retriever.search(q.question, k=k)
            recall = compute_recall_at_k_for_question(q.sources, results, k)

            total_recall += recall
            question_count += 1
            valid_q_count += 1

            is_doc = any(s.file_path.endswith(".md") for s in q.sources)
            if is_doc:
                doc_recall += recall
                doc_count += 1
            else:
                code_recall += recall
                code_count += 1

        if valid_q_count == 0:
            print(
                "\033[91m\033[1mAucune question\033[0m avec sources (AnsweredQuestion) "  # noqa: E501
                "trouvée dans le dataset."
            )
            return

        mean_recall = (total_recall / question_count) * 100
        mean_doc_recall = (
            (doc_recall / doc_count) * 100 if doc_count > 0 else 0
        )
        mean_code_recall = (
            (code_recall / code_count) * 100 if code_count > 0 else 0
        )

        print("\n" + "=" * 50)
        print(f"\033[96m\033[1mRÉSULTATS DE L'ÉVALUATION\033[0m (Recall@{k})")
        print("=" * 50)
        print(
            f"Global       : {mean_recall:.2f}% ({question_count} questions)"
        )
        print(f"Docs (md)    : {mean_doc_recall:.2f}% ({doc_count} questions)")
        print(
            f"Code (py/..) : {mean_code_recall:.2f}% ({code_count} questions)"
        )
        print("=" * 50)

    def search_dataset(
        self, dataset_path: str, output_path: str, k: int = 5
    ) -> None:
        """
        Traite un fichier de questions et exporte les résultats de recherche.
        """
        if not os.path.exists(INDEX_SAVE_DIR):
            print("\033[91m\033[1mErreur\033[0m : L'index n'existe pas.")
            return

        retriever = BM25Retriever()
        retriever.load(INDEX_SAVE_DIR)

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(
                f"\033[91m\033[1mErreur\033[0m lors de l'ouverture du dataset : {e}"  # noqa: E501
            )
            return
        dataset = RagDataset(**data)

        results_list = []
        for q in tqdm(dataset.rag_questions, desc="Recherche en cours"):
            chunks = retriever.search(q.question, k=k)

            for chunk in chunks:
                if chunk.file_path.startswith("vllm-0.10.1"):
                    chunk.file_path = f"data/raw/{chunk.file_path}"

            results_list.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=chunks,  # type: ignore[arg-type]
                )
            )

        student_results = StudentSearchResults(
            search_results=results_list, k=k
        )
        out_dict = student_results.model_dump(by_alias=True)
        os.makedirs(output_path, exist_ok=True)

        input_filename = os.path.basename(dataset_path)
        final_file_path = os.path.join(output_path,
                                       f"results_{input_filename}")

        try:
            with open(final_file_path, "w", encoding="utf-8") as f:
                json.dump(out_dict, f, indent=4)
            print(
                f"\033[92m\033[1mRésultats\033[0m "
                f"sauvegardés dans {final_file_path}"
            )
        except Exception as e:
            print(f"\033[91m\033[1mErreur\033[0m de sauvegarde : {e}")

    def answer(self, query: str, k: int = 5, stream: bool = False) -> None:
        """
        Répond à une question en utilisant le contexte récupéré.
        """
        if not os.path.exists(INDEX_SAVE_DIR):
            print("\033[91m\033[1mErreur\033[0m : L'index n'existe pas.")
            return

        retriever = BM25Retriever()
        retriever.load(INDEX_SAVE_DIR)

        print(
            f"\033[96m\033[1mRecherche de contexte\033[0m pour : '{query}'..."
        )
        try:
            chunks = retriever.search(query, k=k)
        except Exception as e:
            print(
                f"\033[93m\033[1mAttention\033[0m : Erreur lors de la recherche : {e}"  # noqa: E501
            )
            chunks = []

        llm = LLMClient()
        try:
            if stream:
                print("\n" + "=" * 50)
                print("\033[96m\033[1mRÉPONSE GÉNÉRÉE\033[0m")
                print("=" * 50)
                answer_text = llm.generate_answer(query, chunks, stream=True)
                print("\n" + "=" * 50)
            else:
                answer_text = llm.generate_answer(query, chunks, stream=False)
                print("\n" + "=" * 50)
                print("\033[96m\033[1mRÉPONSE GÉNÉRÉE\033[0m")
                print("=" * 50)
                print(answer_text)
                print("=" * 50)
        except Exception as e:
            print(
                f"\033[93m\033[1mAttention\033[0m : Erreur lors de la génération : {e}"  # noqa: E501
            )

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """
        Génère les réponses de tout un dataset à partir des
        résultats de recherche.
        """
        try:
            with open(student_search_results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            search_results = StudentSearchResults(**data)
        except Exception as e:
            print(f"\033[91m\033[1mErreur\033[0m de chargement : {e}")
            return

        print(
            f"Loaded {len(search_results.search_results)} questions "
            f"from {student_search_results_path}"
        )
        llm = LLMClient()

        answered_results = []
        for result in tqdm(
            search_results.search_results, desc="Génération des réponses"
        ):
            chunks = []
            for src in result.retrieved_sources:
                try:
                    with open(src.file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                        text = content[
                            src.first_character_index: src.last_character_index  # noqa: E501
                        ]
                        chunks.append(
                            Chunk(
                                file_path=src.file_path,
                                first_character_index=(
                                    src.first_character_index
                                ),
                                last_character_index=src.last_character_index,
                                text=text,
                            )
                        )
                except Exception:
                    pass

            try:
                answer_text = llm.generate_answer(result.question, chunks)
            except Exception:
                answer_text = "Erreur."

            answered_results.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer_text,
                )
            )

        final_result = StudentSearchResultsAndAnswer(
            search_results=answered_results, k=search_results.k
        )

        os.makedirs(save_directory, exist_ok=True)
        filename = os.path.basename(student_search_results_path)
        output_path = os.path.join(save_directory, filename)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_result.model_dump(), f, indent=2)
            print(
                f"Processed {len(answered_results)} of "
                f"{len(answered_results)} questions"
            )
            print(f"Saved student_search_results_and_answer to {output_path}")
        except Exception as e:
            print(f"\033[91m\033[1mErreur\033[0m de sauvegarde : {e}")


if __name__ == "__main__":
    fire.Fire(RAGCLI)
