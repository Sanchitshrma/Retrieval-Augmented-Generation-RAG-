# Retrieval-Augmented Generation (RAG) Pipeline

A hands-on, step-by-step RAG project built with **LangChain**, **ChromaDB** and the **Gemini API**. It starts with a basic ingestion and retrieval pipeline and builds up to advanced retrieval: history-aware queries, multi-query retrieval, Reciprocal Rank Fusion, hybrid (vector + BM25) search and reranking.

![RAG architecture](/rag_architecture.png)

---

## Features

- **Document ingestion:** load `.txt` files, chunk them, embed them with Gemini and persist to ChromaDB (cosine similarity)
- **Multiple chunking strategies:** Character, Recursive, Semantic and Agentic (LLM-driven) splitting
- **Vector retrieval:** top-k similarity search over ChromaDB
- **History-aware retrieval:** rewrites follow-up questions into standalone queries using chat history
- **Multi-query retrieval:** the LLM generates 3 query variations to widen recall
- **Reciprocal Rank Fusion (RRF):** merges ranked lists from multiple queries (k = 60)
- **Hybrid search:** vector search + BM25 keyword search through a weighted `EnsembleRetriever`
- **Reranking:** Cohere rerank (`rerank-english-v3.0`) applied to the hybrid results
- **Multimodal PDF ingestion:** Unstructured partitioning, title-based chunking and Gemini-generated summaries of text, tables and images
- **Grounded answers:** prompts restrict the model to the retrieved context and fall back to "I don't have enough information…" when the answer isn't there

---

## Architecture

**Indexing (offline)**

```
Source Documents -> Document Loading -> Chunking -> Embeddings -> ChromaDB
(PDF path)  Partitioning -> Title-Based Chunking -> AI Summaries -> Embeddings
```

**Query (online)**

```
User Query (+ Conversation History)
   -> History-Aware Rewrite -> Multi-Query Generation
   -> Vector Search (ChromaDB) + Keyword Search (BM25)
   -> Rank Fusion (RRF / ensemble) -> Reranking (Cohere)
   -> Retrieved Context -> Prompt Construction -> Gemini API -> Generated Answer
   -> Q&A appended to chat history
```

---

## Models used

| Purpose | Model |
|---|---|
| Answer generation, query rewriting, multi-query, PDF summaries | `gemini-2.5-flash` (via `ChatGoogleGenerativeAI`, temperature 0) |
| Embeddings | `gemini-embedding-001` (via `GoogleGenerativeAIEmbeddings`) |
| Agentic chunking demo | a Gemini Flash-Lite model (see `7_agentic_chunking.py`) |
| Reranking | Cohere `rerank-english-v3.0` |

---

## Project structure

```
.
├── docs/                                    # Source documents
│   ├── Google.txt
│   └── attention-is-all-you-need.pdf
├── 1_ingestion_pipeline.py                  # Load -> chunk (size 2000) -> embed -> persist to db/chroma_db
├── 2_retrieval_pipeline.py                  # Top-k vector retrieval + answer
├── 3_answer_generation.py                   # Retrieval + prompt + Gemini answer
├── 4_history_aware_generation.py            # Conversational RAG with query rewriting
├── 5_recursive_character_text_spliiter.py   # Recursive chunking demo
├── 6_semantic_chunking.py                   # Semantic chunking demo
├── 7_agentic_chunking.py                    # LLM-driven chunking demo
├── 8_multi_modal_rag.ipynb                  # PDF text/tables/images ingestion + answer
├── 9_retrieval_methods.py                   # Similarity / threshold / MMR retrieval options
├── 10_multi_query_retrieval.py              # Multi-query generation + retrieval
├── 11_reciprocal_rank_fusion.py             # Multi-query + RRF
├── 12_hybrid_search.ipynb                   # Vector + BM25 ensemble
├── 13_reranker.ipynb                        # Hybrid search + Cohere reranking
├── chunks_export.json                       # Sample output: AI-enhanced PDF chunks
└── rag_results.json                         # Sample output: retrieved chunks for a query
```

Running the multimodal notebook also creates `dbv1/` and `dbv2/` (ChromaDB stores for the raw and summarised PDF chunks).

---

## Tech stack

Python · LangChain · ChromaDB · BM25 (`rank-bm25`) · Cohere Rerank · Unstructured · Gemini API

---

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/Sanchitshrma/Retrieval-Augmented-Generation-RAG-.git
cd Retrieval-Augmented-Generation-RAG-

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install langchain langchain-community langchain-classic langchain-chroma \
            langchain-text-splitters langchain-experimental langchain-google-genai \
            langchain-cohere chromadb rank_bm25 python-dotenv pydantic
pip install "unstructured[all-docs]"     # only for the multimodal PDF notebook
```

The multimodal notebook also needs system packages: Poppler, Tesseract and libmagic (install commands are in the notebook).

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
COHERE_API_KEY=your_cohere_api_key      # only for 13_reranker.ipynb
```

Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/).

### 3. Run the pipeline

```bash
# Build the vector store from docs/*.txt (creates db/chroma_db)
python 1_ingestion_pipeline.py

# Retrieve and answer
python 2_retrieval_pipeline.py
python 3_answer_generation.py

# Conversational RAG (type 'quit' to exit)
python 4_history_aware_generation.py
```

Advanced retrieval:

```bash
python 10_multi_query_retrieval.py
python 11_reciprocal_rank_fusion.py
```

Open `12_hybrid_search.ipynb` and `13_reranker.ipynb` in Jupyter for hybrid search and reranking, and `8_multi_modal_rag.ipynb` for the PDF pipeline.

> `1_ingestion_pipeline.py` skips re-processing if `db/chroma_db` already exists. Delete that folder to rebuild. If you change the embedding model, rebuild the store, because vectors from different models are not compatible.

---

## How the retrieval stack works

| Stage | What it does | Where |
|---|---|---|
| History-aware rewrite | Turns a follow-up question into a standalone, searchable query | `4_history_aware_generation.py` |
| Multi-query | Generates 3 rephrasings to retrieve different relevant chunks | `10_multi_query_retrieval.py` |
| Vector search | Dense similarity search in ChromaDB (cosine) | `2_retrieval_pipeline.py` |
| BM25 | Sparse keyword matching for exact terms and names | `12_hybrid_search.ipynb` |
| RRF | Scores chunks by `1 / (k + rank)` summed across result lists (k = 60) | `11_reciprocal_rank_fusion.py` |
| Hybrid ensemble | Weighted combination of vector and BM25 results (0.5 / 0.5 in the hybrid demo, 0.7 / 0.3 in the reranker demo) | `12_hybrid_search.ipynb`, `13_reranker.ipynb` |
| Reranking | Re-scores the top 15 + 15 candidates and keeps the top 10 | `13_reranker.ipynb` |

---

## Notes

- Retrieval stages are separate, progressively more advanced scripts and notebooks rather than one combined entry point.
- The hybrid search and reranker notebooks run on in-notebook sample chunks, so they don't need `db/chroma_db`.
- `docs/` ships with `Google.txt` and the "Attention Is All You Need" PDF. Some example queries in `9_retrieval_methods.py`, `10_multi_query_retrieval.py` and `11_reciprocal_rank_fusion.py` ask about Microsoft and Tesla, so add matching `.txt` files to `docs/` (or edit the queries) to get meaningful results.
