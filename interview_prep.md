# Préparation à la Soutenance RAG

Voici les 5 questions obligatoires de la grille d'évaluation (partie *Student Understanding*), accompagnées des réponses adaptées à notre implémentation pour que tu sois prêt(e) le jour J !

---

### 1. "What is Retrieval-Augmented Generation (RAG) and why is it useful?"
**Explication simple :**
Le RAG (Génération Augmentée par la Recherche) est une technique qui permet de donner une mémoire ou un contexte à un LLM (Grand Modèle de Langage) en lui fournissant des documents pertinents externes avant qu'il ne réponde.

**Pourquoi c'est utile ?**
* **Réduit les hallucinations :** Au lieu d'inventer, le LLM s'appuie sur les documents qu'on lui donne.
* **Accès aux données privées / récentes :** Un LLM classique n'a pas accès au code source interne d'une entreprise ou à la documentation récente (ici `vllm-0.10.1`). Le RAG permet d'interroger ces données spécifiques sans avoir à ré-entraîner (fine-tuner) le modèle (ce qui est très coûteux).

---

### 2. "Walk me through your complete RAG pipeline, from raw documents to a generated answer."
**Ton Pipeline de A à Z :**
1. **Ingestion (Lecture des données) :** On parcourt le dossier contenant le code et la doc (`vllm-0.10.1`), on lit chaque fichier (Python, Markdown, etc.).
2. **Chunking (Découpage) :** Les fichiers sont découpés en "chunks" (morceaux) d'une taille limite (`MAX_CHUNK_SIZE = 2000`). Pour éviter de couper une phrase ou une fonction en deux, on utilise une logique de *chevauchement* (*overlap*) de 200 à 1000 caractères.
3. **Indexation :** On passe ces chunks à travers la fonction de tokenisation avancée (`tokenize_code`) qui gère le snake_case et camelCase, puis on indexe ces mots dans notre moteur de recherche BM25 (`bm25s`).
4. **Retrieval (Recherche) :** Quand l'utilisateur pose une question, la requête passe dans BM25 qui calcule un score mathématique pour trouver les 5 meilleurs chunks (le "top `k`") correspondants.
5. **Generation (Réponse) :** On prend ces 5 chunks, on les formate sous forme de contexte, et on les donne au LLM (Qwen 0.6B) avec un *prompt* clair. Le modèle lit le contexte et rédige la réponse finale en citant ses sources.

---

### 3. "What is the difference between TF-IDF and BM25 as retrieval methods?"
**Explication simple :**
* **TF-IDF (Term Frequency - Inverse Document Frequency) :** C'est la base. Plus un mot de la question apparaît souvent dans le document (TF), plus le score monte. Mais si c'est un mot banal ("le", "et") qui apparaît partout dans le projet (IDF faible), le score est neutralisé.
* **BM25 (Best Matching 25) :** C'est une amélioration moderne de TF-IDF.
    1. Il "plaque" (sature) le score de la fréquence (TF) via le paramètre `k1`. (Un mot qui apparaît 2 fois ou 20 fois n'a pas une différence infinie de score, pour éviter le "spam" de mots-clés).
    2. Il ajuste le score selon la **longueur du document** via le paramètre `b`. (Un mot trouvé dans un très petit document a plus de valeur que s'il est noyé au milieu d'un fichier géant de 10 000 lignes).

---

### 4. "What implementation choices did you make and why? What trade-offs did you consider?"
**Choix et Compromis (Trade-offs) réalisés :**
* **Recherche lexicale (BM25) vs Vectorielle (Embeddings) :**
  * *Choix :* Nous avons choisi BM25 pur.
  * *Pourquoi :* C'est très rapide, sans besoin de GPU pour indexer, et extrêmement efficace pour le **Code**. Les variables (ex: `max_tokens_vllm`) sont des mots-clés exacts que les Embeddings gèrent parfois mal.
* **Tokenisation avancée :**
  * *Choix :* Ajouter une découpe du *camelCase* et du *snake_case* avec du "stemming" (enlever les terminaisons).
  * *Trade-off :* Cela ralentit très légèrement l'indexation, mais ça permet à l'utilisateur de chercher "token" et de trouver des variables appelées `total_tokens_count`.
* **Taille des Chunks bridée à 2000 :**
  * *Choix :* Respecter la taille stricte du sujet, mais la compenser avec un **énorme overlap** (jusqu'à 1000 caractères) pour le Markdown.
  * *Trade-off :* Le contexte est limité et un gros overlap augmente le nombre de chunks en mémoire, mais cela garantit qu'aucune information n'est scindée (Recall final énorme de >84% sur la doc).

---

### 5. "If you had more time, what would you improve in your system?"
**Ce que tu peux dire au correcteur :**
1. **Recherche Hybride :** Combiner BM25 (lexical) avec des Embeddings (vectoriel) pour que le moteur puisse comprendre les *synonymes* et les questions posées en langage naturel ("comment ça marche" -> "how does it work").
2. **Chunking intelligent (AST) :** Au lieu de découper bêtement par caractères, utiliser la bibliothèque `tree-sitter` pour découper le code proprement "fonction par fonction" ou "classe par classe".
3. **Re-Ranking :** Utiliser un modèle léger de type *Cross-Encoder* pour réordonner (re-ranker) plus intelligemment les 20 résultats trouvés par BM25, avant d'en garder seulement 5 pour le LLM.
4. **LLM Server (vLLM) :** Plutôt que de charger le LLM localement en mémoire à chaque requête via Transformers, déployer un serveur vLLM local. (Le LLM reste allumé et répondrait en quelques millisecondes).
