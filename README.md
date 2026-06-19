*This project has been created as part of the 42 curriculum by avauclai.*

# RAG Against The Machine: Will you answer my questions?

## Description
This project implements a Retrieval-Augmented Generation (RAG) system tailored for the `vLLM` codebase. The goal of this application is to index the vLLM repository, search for relevant code snippets and documentation based on a user's query, and dynamically generate an accurate, source-grounded answer using the local `Qwen/Qwen3-0.6B` Large Language Model.

## Instructions
1. **Requirements:** Python 3.10+, `uv` package manager.
2. **Installation:** Run `make install` to install all dependencies from `pyproject.toml` and `uv.lock`.
3. **Index Generation:** Run `uv run python -m src.main index` to process the data and generate the BM25 index.
4. **Search/Querying:** Run `uv run python -m src.main search "Your question here"` to retrieve context.
5. **Answer Generation:** Run `uv run python -m src.main answer "Your question here"` to retrieve context and generate an AI answer.
6. **Linting:** Run `make lint` or `make lint-strict` to verify type hinting and PEP 8 standard constraints.

## System Architecture
The RAG pipeline is composed of the following modules:
- **Ingestion (`chunker.py` / `pipeline.py`):** Uses the `ast` module to semantically parse Python code into functions and classes, and recursively chunks Markdown text while respecting a configurable character limit (default 2000).
- **Retrieval (`bm25_engine.py`):** Utilizes `bm25s` to index all the extracted chunks and perform fast keyword-based similarity search (BM25 algorithm).
- **Generation (`llm_client.py`):** Wraps `Qwen/Qwen3-0.6B` using `transformers` and `accelerate`. It injects the retrieved chunks into a strict system prompt to reduce hallucinations and enforce source citations.
- **Evaluation (`metrics.py`):** Evaluates the retrieval accuracy by computing the Recall@k against a ground-truth dataset.
- **CLI (`main.py`):** A single unified entrypoint powered by Python `fire`, providing all required commands (`index`, `search`, `answer`, `search_dataset`, `answer_dataset`, `evaluate`).

## Chunking Strategy
- **Python Code:** Parsed via the `ast` module. The chunker identifies top-level classes and functions, extracting their line numbers to generate contextually complete chunks. Fallbacks are used for lines outside of functions/classes.
- **Markdown / Text:** Parsed via `markdown_it`. The chunker recursively splits paragraphs based on standard Markdown block tokens. If a block is too large, it is split sequentially.

## Retrieval Method
We chose **BM25** (via the `bm25s` library), an advanced TF-IDF variant. It efficiently indexes words based on term frequency and inverse document frequency, making it excellent for matching specific variable names and syntax in codebases.

## Performance Analysis
- **Recall@5:** Evaluated to achieve above 80% on Markdown and >50% on Code, surpassing the minimum requirements.
- **Indexing Time:** BM25 indexing is rapid and lightweight, operating well within the 5-minute threshold.
- **Throughput:** Retrieving over the pre-built BM25 index enables fast inference (warm retrieval).

## Design Decisions
- Adopted Pydantic models for rigid JSON data structures and input validation.
- Decoupled `LLMClient` to handle fallback logic securely (if `Qwen/Qwen3-0.6B` model requires different Hugging Face topologies).
- `answer_dataset` dynamically reloads chunk texts from the local filesystem based on indices provided by `MinimalSource` to preserve memory efficiency and avoid huge JSON payloads.

## Challenges Faced
- Managing Python `ast` parse node inconsistencies (such as `None` line references).
- Adhering strictly to `flake8` standards without compromising code readability or disabling `E501` aggressively (handled via `.flake8` configurations).
- Resolving Mypy invariant covariance logic for inherited `List` Pydantic properties.

## Example Usage
```bash
# Index the vLLM repository
make run -- index --repo_path vllm-0.10.1

# Answer a query using top 5 context snippets
make run -- answer "How to configure OpenAI server?" --k 5

# Evaluate the retriever against ground truth
make run -- evaluate --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10
```

## Resources
- [Qwen Model Documentation](https://huggingface.co/Qwen)
- [BM25s Library](https://github.com/xhluca/bm25s)
- **AI Usage:** Generative models were utilized during the conception phase to understand the mathematics behind the BM25 formula, brainstorm Python AST edge cases, and refine the typing and linting workflows. All generated code was systematically tested and reviewed.
