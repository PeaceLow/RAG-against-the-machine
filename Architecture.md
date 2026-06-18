## 📂 L'Arborescence des Fichiers (src/)
```text
src/
├── __init__.py
├── main.py                  # Point d'entrée unique de l'application (Python Fire)
├── config.py                # Centralisation des constantes (ex: max_chunk_size)
├── models.py                # Modèles Pydantic (MinimalSource, RagDataset, etc.)
├── utils.py                 # Gestionnaires de contexte (fichiers), barres tqdm
│
├── ingestion/
│   ├── __init__.py
│   ├── pipeline.py          # Coordonne la lecture du repo et la sauvegarde de l'index
│   └── chunker.py           # Logique de découpage (AST pour Python, Markdown parser)
│
├── retrieval/
│   ├── __init__.py
│   ├── base.py              # Classe abstraite ou interface pour le Retriever
│   └── bm25_engine.py       # Implémentation de BM25 (ou TF-IDF)
│
├── generation/
│   ├── __init__.py
│   └── llm_client.py        # Chargement de Qwen3-0.6B et formatage du prompt
│
└── evaluation/
    ├── __init__.py
    └── metrics.py           # Calcul rigoureux du Recall@k (avec l'overlap de 5%)

```
## 🗺️ Ta Feuille de Route : L'Ordre d'Exécution
Ne fonce pas tête baissée dans le LLM. On avance par couches successives pour s'assurer que chaque brique est stable et validée par mypy.
### Étape 1 : Les Fondations et les Contrats de Données (models.py & config.py)
Avant de coder la moindre fonction, on doit définir nos structures.
 * **Quoi faire :** Recode exactement les classes Pydantic imposées par le sujet (MinimalSource, UnansweredQuestion, AnsweredQuestion, etc.).
 * **Pourquoi :** Ça va forcer tout le reste de ton code à utiliser les bons types dès le départ. Si tu as besoin d'ajouter des champs optionnels pour ton debugging, fais-le ici.
### Étape 2 : Le Découpeur Intelligent (ingestion/chunker.py)
C'est la pièce maîtresse pour tes scores de Recall.
 * **Quoi faire :** Écris deux stratégies distinctes.
   1. Pour Markdown : Découpe par blocs logiques (sections, paragraphes).
   2. Pour Python : Utilise le module ast pour extraire des fonctions ou classes complètes.
 * **Règle d'or :** Chaque chunk généré doit renvoyer le texte, mais aussi capturer minutieusement le first_character_index et le last_character_index par rapport au fichier d'origine. Le tout bridé à 2000 caractères max (paramétrable).
### Étape 3 : Le Moteur de Recherche (retrieval/)
 * **Quoi faire :** Crée ton moteur bm25_engine.py (je te conseille d'utiliser bm25s comme suggéré dans le sujet). Il doit prendre tes chunks, indexer le texte, et être capable de sauvegarder/charger cet index sur le disque dans data/processed/.
 * **Objectif :** Implémenter la fonction qui prend une requête textuelle et te ressort le top-k des meilleurs chunks.
### Étape 4 : La CLI de Base et l'Évaluation (main.py & evaluation/metrics.py)
C'est le moment de valider la première moitié du projet ("l'infrastructure").
 * **Quoi faire :** Code la commande index et la commande evaluate dans ta CLI avec Python Fire. Développe le calcul du Recall@k dans metrics.py en vérifiant bien la règle des 5% d'overlap minimum pour qu'une source soit considérée comme trouvée.
 * **Milestone :** À la fin de cette étape, tu dois pouvoir indexer le repo vLLM en moins de 5 minutes et sortir ton score de Recall@5 sur le dataset public. Si tu n'as pas 80% sur les docs et 50% sur le code, tu retournes bosser ton chunker.py avant d'aller plus loin.
### Étape 5 : L'Intégration du LLM (generation/llm_client.py)
Une fois que ton moteur de recherche est une horloge suisse, on passe à la génération.
 * **Quoi faire :** Configure le chargement de Qwen/Qwen3-0.6B. Écris le code qui prend la question de l'utilisateur, récupère les meilleurs chunks via ton Retriever, emballe le tout dans un prompt propre ("Voici le contexte exclusif pour répondre..."), et envoie ça au modèle.
 * **Attention :** Assure-toi de tronquer proprement si le contexte dépasse la limite de tokens du modèle. La sortie doit être formater pour remplir ton modèle Pydantic MinimalAnswer.
### Étape 6 : Finalisation de la CLI, Nettoyage et Makefile
 * **Quoi faire :** Ajoute les commandes manquantes (search, search_dataset, answer, answer_dataset) dans main.py. Sécurise tout le code avec des blocs try-except pour qu'aucune exception non gérée ne vienne faire crasher ton programme pendant la soutenance.
 * **Vérification :** Passe un coup de flake8 et assure-toi que mypy --strict ne renvoie aucune erreur. Configure ton Makefile pour automatiser tout ça (install, run, lint, etc.).
Allez, commence par poser tes modèles Pydantic et ton architecture de dossiers. Si tu as un doute sur la manière d'attaquer l'AST Python pour le chunker, tu me demandes. Au boulot !
