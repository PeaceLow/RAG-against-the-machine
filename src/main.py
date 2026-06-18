import os
import json
import fire
import time
from typing import Optional
from tqdm import tqdm

from src.ingestion.pipeline import build_pipeline
from src.retrieval.bm25_engine import BM25Retriever
from src.config import MAX_CHUNK_SIZE
from src.models import RagDataset, StudentSearchResults, MinimalSearchResults, AnsweredQuestion
from src.evaluation.metrics import compute_recall_at_k_for_question

INDEX_SAVE_DIR = "data/processed/bm25_index"

class RAGCLI:
    """CLI principale pour le système RAG."""

    def index(self, repo_path: str = "vllm-0.10.1"):
        """
        Indexe le dépôt de code.
        
        Args:
            repo_path (str): Le chemin vers le dépôt à indexer.
        """
        start_time = time.time()
        
        # 1. Pipeline d'ingestion (découpage)
        chunks = build_pipeline(repo_path, max_chunk_size=MAX_CHUNK_SIZE)
        
        # 2. Indexation BM25
        print(f"⚙️ Indexation avec BM25 (Cela peut prendre quelques instants)...")
        retriever = BM25Retriever()
        retriever.index(chunks)
        
        # 3. Sauvegarde
        retriever.save(INDEX_SAVE_DIR)
        
        elapsed = time.time() - start_time
        print(f"🎉 Indexation terminée et sauvegardée dans '{INDEX_SAVE_DIR}'. Temps écoulé : {elapsed:.2f} secondes.")

    def search(self, query: str, k: int = 5):
        """
        Effectue une recherche textuelle rapide.
        
        Args:
            query (str): La requête ou question de l'utilisateur.
            k (int): Le nombre de chunks à retourner (Defaut: 5).
        """
        if not os.path.exists(INDEX_SAVE_DIR):
            print("❌ Erreur : L'index n'existe pas. Veuillez exécuter 'python -m src.main index' au préalable.")
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
        
        print("\n" + "="*50)
        print("🎯 RÉSULTATS DE RECHERCHE")
        print("="*50)
        for i, chunk in enumerate(results, 1):
            print(f"\n[{i}] Fichier : {chunk.file_path}")
            print(f"    Positions : {chunk.first_character_index} - {chunk.last_character_index}")
            text_preview = chunk.text[:200].replace('\n', ' ')
            print(f"    Aperçu : {text_preview}...")

    def evaluate(self, dataset_path: str, k: int = 5):
        """
        Évalue le Recall@k sur un jeu de données (JSON).
        
        Args:
            dataset_path (str): Chemin vers le fichier JSON du dataset (ground truth).
            k (int): Le k pour calculer le Recall@k.
        """
        if not os.path.exists(INDEX_SAVE_DIR):
            print("❌ Erreur : L'index n'existe pas. Veuillez exécuter 'python -m src.main index' au préalable.")
            return
            
        print("⏳ Chargement du moteur de recherche...")
        retriever = BM25Retriever()
        retriever.load(INDEX_SAVE_DIR)
        
        print(f"📚 Chargement du dataset depuis {dataset_path}...")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Parse Pydantic
        dataset = RagDataset(**data)
        
        total_recall = 0.0
        question_count = 0
        
        doc_recall = 0.0
        doc_count = 0
        
        code_recall = 0.0
        code_count = 0
        
        print(f"🏃 Début de l'évaluation sur {len(dataset.rag_questions)} questions (Recall@{k})...")
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
            is_doc = any(s.file_path.endswith('.md') for s in q.sources)
            if is_doc:
                doc_recall += recall
                doc_count += 1
            else:
                code_recall += recall
                code_count += 1
                
        if question_count == 0:
            print("❌ Aucune question avec sources (AnsweredQuestion) dans ce dataset.")
            return
            
        mean_recall = (total_recall / question_count) * 100
        mean_doc_recall = (doc_recall / doc_count) * 100 if doc_count > 0 else 0
        mean_code_recall = (code_recall / code_count) * 100 if code_count > 0 else 0
        
        print("\n" + "="*50)
        print(f"📊 RÉSULTATS DE L'ÉVALUATION (Recall@{k})")
        print("="*50)
        print(f"Global       : {mean_recall:.2f}% ({question_count} questions)")
        print(f"Docs (md)    : {mean_doc_recall:.2f}% ({doc_count} questions)")
        print(f"Code (py/..) : {mean_code_recall:.2f}% ({code_count} questions)")
        print("="*50)

    def search_dataset(self, dataset_path: str, output_path: str, k: int = 5):
        """
        Traite un fichier de questions et exporte les résultats de recherche.
        """
        if not os.path.exists(INDEX_SAVE_DIR):
            print("❌ Erreur : L'index n'existe pas.")
            return
            
        retriever = BM25Retriever()
        retriever.load(INDEX_SAVE_DIR)
        
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dataset = RagDataset(**data)
        
        results_list = []
        for q in tqdm(dataset.rag_questions, desc="Recherche en cours"):
            chunks = retriever.search(q.question, k=k)
            # Conversion implict Chunk -> MinimalSource
            results_list.append(MinimalSearchResults(
                question_id=q.question_id,
                question=q.question,
                retrieved_sources=chunks
            ))
            
        student_results = StudentSearchResults(search_results=results_list, k=k)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(student_results.model_dump(), f, indent=2)
            
        print(f"✅ Résultats de recherche sauvegardés dans {output_path}")


if __name__ == "__main__":
    fire.Fire(RAGCLI)
