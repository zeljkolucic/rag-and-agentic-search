# RAG and Agentic Search

This project was built following **Anthropic's RAG and Agentic Search course**. It covers the fundamentals of Retrieval Augmented Generation (RAG) through a series of Jupyter notebooks, progressing from basic text chunking to a full hybrid search pipeline.

## What is RAG?

Retrieval Augmented Generation (RAG) is a technique for working with documents too large to fit into a single prompt. Instead of passing an entire document to a model, RAG:

1. **Preprocesses** the document by splitting it into smaller chunks
2. **Indexes** those chunks so they can be searched efficiently
3. **Retrieves** only the most relevant chunks when answering a question
4. **Generates** a response using those targeted chunks as context

This trades implementation complexity for scalability — enabling you to work with document collections that would be impractical to stuff into a single prompt.

## Notebooks

### `001_chunking.ipynb` — Text Chunking Strategies

Breaking a document into chunks is a critical first step. A poor chunking strategy can cause the retriever to surface irrelevant context and produce wrong answers. Three strategies are implemented:

| Strategy | Description | Best for |
|---|---|---|
| **Size-based** (`chunk_by_char`) | Splits text into equal-length character segments with optional overlap | Any document type; reliable fallback |
| **Sentence-based** (`chunk_by_sentence`) | Splits on sentence boundaries and groups sentences with overlap | General prose |
| **Structure-based** (`chunk_by_section`) | Splits on Markdown section headers (`## `) | Well-structured documents you control |

Size-based chunking is the most universal because it works on any content — including code and PDFs with no structural markers — even if it can split mid-sentence. Adding overlap between chunks helps restore context lost at boundaries.

### `002_embeddings.ipynb` — Generating Embeddings with Voyage AI

An **embedding** is a numerical vector that represents the semantic meaning of a piece of text. Similar texts produce vectors that point in similar directions.

Since Anthropic does not provide an embedding API, this project uses [Voyage AI](https://www.voyageai.com/) (`voyage-3-large` model):

```python
def generate_embedding(text, model="voyage-3-large", input_type="query"):
    result = client.embed([text], model=model, input_type=input_type)
    return result.embeddings[0]
```

The function accepts either a single string or a list of strings for batch processing.

### `003_vectordb.ipynb` — Semantic Search with a Vector Index

Embeddings are stored in a custom `VectorIndex` that supports nearest-neighbour lookup. When a user asks a question, its embedding is compared against all stored chunk embeddings using **cosine distance**:

```
cosine_distance = 1 - cosine_similarity
```

Values near `0` indicate high similarity; values near `1` indicate dissimilarity. The index returns the `k` chunks with the lowest distance to the query.

**Five-step RAG pipeline:**

```
1. Chunk text by section
2. Generate embeddings for each chunk
3. Store embeddings in VectorIndex
4. Embed the user's question
5. Search the store and return the top-k chunks
```

The retrieved chunks are then inserted into the prompt sent to Claude.

### `004_bm25.ipynb` — Lexical Search with BM25

Semantic search can miss results when the user queries by a specific term (e.g., an incident ID like `INC-2023-Q4-011`) rather than by concept. **BM25 (Best Match 25)** is a classical lexical ranking algorithm that solves this by rewarding exact term matches, especially rare terms.

How BM25 scores a document for a query:

1. Tokenize the query into terms
2. Compute **Inverse Document Frequency (IDF)** — rare terms get higher weight
3. Compute **term frequency** in the candidate document, normalised by document length
4. Sum the weighted scores across all query terms

The `BM25Index` class implements this from scratch with configurable `k1` (term frequency saturation) and `b` (document length normalisation) parameters.

### `005_hybrid.ipynb` — Hybrid Search with Reciprocal Rank Fusion

Neither semantic nor lexical search alone is optimal. This notebook combines both inside a `Retriever` class using **Reciprocal Rank Fusion (RRF)** to merge their result lists.

```
RRF_score(d) = Σ ( 1 / (k + rank_i(d)) )
```

Each document is ranked by each index independently. RRF assigns a score based on the rank position from each source, then re-ranks by the combined score. Documents that appear near the top in both indexes rise to the top of the merged list.

The `Retriever` wraps any number of indexes that implement a common `SearchIndex` protocol:

```python
class SearchIndex(Protocol):
    def add_document(self, document: Dict[str, Any]) -> None: ...
    def search(self, query: Any, k: int = 1) -> List[Tuple[Dict, float]]: ...
```

This design makes the retriever easily extensible — adding a new search method only requires implementing the same two-method interface.

## Setup

**Prerequisites:** Python 3.9+, Jupyter

**Install dependencies:**

```bash
pip install voyageai python-dotenv
```

**Configure API keys** in a `.env` file:

```
ANTHROPIC_API_KEY="your_key_here"
VOYAGE_API_KEY="your_key_here"
```

Get a Voyage AI key at [voyageai.com](https://www.voyageai.com/) — a free tier is available.

**Run the notebooks** in order (`001` → `005`). Each notebook is self-contained and uses `report.md` as the source document.

## Project Structure

```
.
├── report.md              # Source document used across all notebooks
├── 001_chunking.ipynb     # Text chunking strategies
├── 002_embeddings.ipynb   # Embedding generation with Voyage AI
├── 003_vectordb.ipynb     # Semantic search with a custom VectorIndex
├── 004_bm25.ipynb         # Lexical search with BM25
└── 005_hybrid.ipynb       # Hybrid search with Reciprocal Rank Fusion
```
