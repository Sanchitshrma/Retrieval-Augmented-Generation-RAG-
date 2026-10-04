# Retrieval-Augmented Generation (RAG) Pipeline

A hands-on, step-by-step RAG project built with **LangChain** and **ChromaDB**, progressing from a basic ingestion + retrieval pipeline to advanced retrieval: history-aware queries, multi-query retrieval, Reciprocal Rank Fusion, hybrid (vector + BM25) search, and reranking, with the **Gemini API** as the generation layer.

![RAG architecture](assets/rag_architecture.png)

## Features

- **Document ingestion:** load `.txt` files, chunk them, embed them, and persist to ChromaDB (cosine similarity)
- **Multiple chunking strategies:** Character, Recursive, Semantic, and Agentic (LLM-driven) splitting
- **Vector retrieval:** top-k similarity search over ChromaDB
- **History-aware retrieval:** rewrites follow-up questions into standalone queries using chat history
- **Multi-query retrieval:** the LLM generates 3 query variations to widen recall
- **Reciprocal Rank Fusion (RRF):** merges ranked lists from multiple queries (k = 60)
- **Hybrid search:** vector search + BM25 keyword search via a weighted ensemble (0.7 / 0.3)
- **Reranking:** Cohere rerank (`rerank-english-v3.0`) on the fused candidates
- **Multimodal PDF ingestion:** Unstructured partitioning, title-based chunking, and AI-generated summaries of text, tables, and images
- **Grounded answers:** the prompt restricts the model to retrieved context and falls back to "I don't have enough information…" when the answer isn't there

---

## Architecture

**Indexing (offline)**

```
Source Documents -> Document Loading -> Chunking -> Embeddings -> ChromaDB
                    (PDF path) Partitioning -> Title-Based Chunking -> AI Summaries -> Embeddings
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

## Project structure

```
.
├── docs/                              # Source documents (.txt company files + a sample PDF)
│   ├── Google.txt  Microsoft.txt  Nvidia.txt  SpaceX.txt  Tesla.txt
│   └── attention-is-all-you-need.pdf
├── 1_ingestion_pipeline.py            # Load -> chunk -> embed -> persist to ChromaDB
├── 2_retrieval_pipeline.py            # Basic top-k vector retrieval
├── 3_answer_generation.py             # Retrieval + prompt + LLM answer
├── 4_history_aware_generation.py      # Conversational RAG with query rewriting
├── 5_recursive_character_text_spliiter.py   # Recursive chunking demo
├── 6_semantic_chunking.py             # Semantic chunking demo
├── 7_agentic_chunking.py              # LLM-driven chunking demo
├── 8_multi_modal_rag.ipynb            # PDF text/tables/images ingestion + answer
├── 9_retrieval_methods.py             # Similarity / threshold / MMR retrieval options
├── 10_multi_query_retrieval.py        # Multi-query generation + retrieval
├── 11_reciprocal_rank_fusion.py       # Multi-query + RRF
├── 12_hybrid_search.ipynb             # Vector + BM25 ensemble
├── 13_reranker.ipynb                  # Hybrid search + Cohere reranking
├── synthetic_questions.txt            # Sample test questions
└── requirements.txt
```

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
pip install -r requirements.txt
pip install langchain-experimental    # needed by 6_semantic_chunking.py
```

For the Gemini integration also install:

```bash
pip install langchain-google-genai
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
COHERE_API_KEY=your_cohere_api_key        # only for 13_reranker.ipynb
OPENAI_API_KEY=your_openai_api_key        # only while scripts still use OpenAI embeddings/LLM
```

### 3. Run the pipeline

```bash
# Build the vector store (creates db/chroma_db)
python 1_ingestion_pipeline.py

# Retrieve only
python 2_retrieval_pipeline.py

# Full RAG: retrieve + generate
python 3_answer_generation.py

# Conversational RAG (type 'quit' to exit)
python 4_history_aware_generation.py
```

Advanced retrieval:

```bash
python 10_multi_query_retrieval.py
python 11_reciprocal_rank_fusion.py
```

Open `12_hybrid_search.ipynb` and `13_reranker.ipynb` in Jupyter for hybrid search and reranking. The multimodal PDF pipeline lives in `8_multi_modal_rag.ipynb` and needs system dependencies (Poppler, Tesseract, libmagic); the notebook lists the install commands.

> `1_ingestion_pipeline.py` skips re-processing if `db/chroma_db` already exists. Delete that folder to rebuild.

---

## Migrating to Gemini

Swap the OpenAI classes for their Google equivalents:

```python
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Generation / query rewriting / multi-query / summaries
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)   # use a current Gemini model name

# Embeddings (re-run ingestion after changing the embedding model)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
```

Check Google's documentation for the current model names. If you change the embedding model, delete `db/chroma_db` and re-run `1_ingestion_pipeline.py`, because vectors from different embedding models are not compatible.

---

## How the retrieval stack works

| Stage | What it does | Where |
|---|---|---|
| History-aware rewrite | Turns a follow-up question into a standalone, searchable query | `4_history_aware_generation.py` |
| Multi-query | Generates 3 rephrasings to retrieve different relevant chunks | `10_multi_query_retrieval.py` |
| Vector search | Dense similarity search in ChromaDB (cosine) | `2_retrieval_pipeline.py` |
| BM25 | Sparse keyword matching for exact terms and names | `12_hybrid_search.ipynb` |
| RRF | Scores chunks by `1 / (k + rank)` summed across result lists | `11_reciprocal_rank_fusion.py` |
| Hybrid ensemble | Weighted combination of vector (0.7) and BM25 (0.3) results | `12_hybrid_search.ipynb` |
| Reranking | Re-scores candidates for relevance to the query | `13_reranker.ipynb` |

---

## Notes

- Retrieval stages are implemented as separate, progressively more advanced scripts and notebooks rather than a single combined entry point.
- The sample corpus is five company overview files, with questions in `synthetic_questions.txt`.

---

## License

Add a license of your choice (for example MIT) before publishing.
