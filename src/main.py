import os
import json
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

    def index(self, repo_path: str = "vllm-0.10.1") -> None:
        """
        Indexe le dépôt de code.

        Args:
            repo_path (str): Le chemin vers le dépôt à indexer.
        """
        start_time = time.time()

        # 1. Pipeline d'ingestion (découpage)
        chunks = build_pipeline(repo_path, max_chunk_size=MAX_CHUNK_SIZE)
        
        if not chunks:
            print(f"❌ Erreur : Aucun chunk généré à partir de '{repo_path}'.")
            return

        # 2. Indexation BM25
        print(
            "⚙️ Indexation avec BM25 (Cela peut prendre quelques instants)..."
        )
        retriever = BM25Retriever()
        retriever.index(chunks)

        # 3. Sauvegarde
        retriever.save(INDEX_SAVE_DIR)

        elapsed = time.time() - start_time
        print(
            f"🎉 Indexation terminée et sauvegardée dans '{INDEX_SAVE_DIR}'. Temps écoulé : {elapsed:.2f} secondes."
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
                "❌ Erreur : L'index n'existe pas. Veuillez exécuter 'python -m src.main index' au préalable."
            )
            return

        # Chargement
        print("⏳ Chargement du moteur de recherche...")
        start_time = time.time()
        retriever = BM25Retriever()
        retriever.load(INDEX_SAVE_DIR)
        print(f"✅ Moteur chargé en {time.time() - start_time:.2f}s")

        # Recherche
        print(f"🔍 Recherche pour : '{query}'...")
        results = retriever.search(query, k=k)

        print("\n" + "=" * 50)
        print("🎯 RÉSULTATS DE RECHERCHE")
        print("=" * 50)
        for i, chunk in enumerate(results, 1):
            print(f"\n[{i}] Fichier : {chunk.file_path}")
            print(
                f"    Positions : {chunk.first_character_index} - {chunk.last_character_index}"
            )
            text_preview = chunk.text[:200].replace("\n", " ")
            print(f"    Aperçu : {text_preview}...")

    def evaluate(self, dataset_path: str, k: int = 5) -> None:
        """
        Évalue le Recall@k sur un jeu de données (JSON).

        Args:
            dataset_path (str): Chemin vers le fichier JSON du dataset (ground truth).
            k (int): Le k pour calculer le Recall@k.
        """
        if not os.path.exists(INDEX_SAVE_DIR):
            print(
                "❌ Erreur : L'index n'existe pas. Veuillez exécuter 'python -m src.main index' au préalable."
            )
            return

        print("⏳ Chargement du moteur de recherche...")
        retriever = BM25Retriever()
        retriever.load(INDEX_SAVE_DIR)

        print(f"📚 Chargement du dataset depuis {dataset_path}...")
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Erreur lors de l'ouverture du dataset : {e}")
            return

        # Parse Pydantic
        dataset = RagDataset(**data)

        total_recall = 0.0
        question_count = 0

        doc_recall = 0.0
        doc_count = 0

        code_recall = 0.0
        code_count = 0

        print(
            f"🏃 Début de l'évaluation sur {len(dataset.rag_questions)} questions (Recall@{k})..."
        )
        for q in tqdm(dataset.rag_questions, desc="Évaluation"):
            # On ne prend que les questions annotées (qui ont des sources)
            if not isinstance(q, AnsweredQuestion) or not q.sources:
                continue

            results = retriever.search(q.question, k=k)
            recall = compute_recall_at_k_for_question(q.sources, results, k)

            total_recall += recall
            question_count += 1

            # Simple heuristique pour séparer la doc du code (selon les sources)
            # Si au moins une source est du Markdown, on considère que c'est une question DOC, sinon CODE
            is_doc = any(s.file_path.endswith(".md") for s in q.sources)
            if is_doc:
                doc_recall += recall
                doc_count += 1
            else:
                code_recall += recall
                code_count += 1

        if question_count == 0:
            print(
                "❌ Aucune question avec sources (AnsweredQuestion) dans ce dataset."
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
        print(f"📊 RÉSULTATS DE L'ÉVALUATION (Recall@{k})")
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
            print("❌ Erreur : L'index n'existe pas.")
            return

        retriever = BM25Retriever()
        retriever.load(INDEX_SAVE_DIR)

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Erreur lors de l'ouverture du dataset : {e}")
            return
        dataset = RagDataset(**data)

        results_list = []
        for q in tqdm(dataset.rag_questions, desc="Recherche en cours"):
            chunks = retriever.search(q.question, k=k)
            # Conversion implict Chunk -> MinimalSource
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

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(student_results.model_dump(), f, indent=2)

        print(f"✅ Résultats de recherche sauvegardés dans {output_path}")

    def answer(self, query: str, k: int = 5, stream: bool = False) -> None:
        """
        Répond à une question en utilisant le contexte récupéré.
        """
        if not os.path.exists(INDEX_SAVE_DIR):
            print("❌ Erreur : L'index n'existe pas.")
            return

        retriever = BM25Retriever()
        retriever.load(INDEX_SAVE_DIR)

        print(f"🔍 Recherche de contexte pour : '{query}'...")
        try:
            chunks = retriever.search(query, k=k)
        except Exception as e:
            print(f"⚠️ Erreur lors de la recherche : {e}")
            chunks = []

        llm = LLMClient()
        try:
            if stream:
                print("\n" + "=" * 50)
                print("🎯 RÉPONSE GÉNÉRÉE")
                print("=" * 50)
                answer_text = llm.generate_answer(query, chunks, stream=True)
                print("\n" + "=" * 50)
            else:
                answer_text = llm.generate_answer(query, chunks, stream=False)
                print("\n" + "=" * 50)
                print("🎯 RÉPONSE GÉNÉRÉE")
                print("=" * 50)
                print(answer_text)
                print("=" * 50)
        except Exception as e:
            print(f"⚠️ Erreur lors de la génération : {e}")

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """
        Génère les réponses de tout un dataset à partir des résultats de recherche.
        """
        try:
            with open(student_search_results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            search_results = StudentSearchResults(**data)
        except Exception as e:
            print(f"❌ Erreur de chargement : {e}")
            return

        print(
            f"Loaded {len(search_results.search_results)} questions from {student_search_results_path}"
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
                            src.first_character_index : src.last_character_index
                        ]
                        chunks.append(
                            Chunk(
                                file_path=src.file_path,
                                first_character_index=src.first_character_index,
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
                f"Processed {len(answered_results)} of {len(answered_results)} questions"
            )
            print(f"Saved student_search_results_and_answer to {output_path}")
        except Exception as e:
            print(f"❌ Erreur de sauvegarde : {e}")


if __name__ == "__main__":
    fire.Fire(RAGCLI)
