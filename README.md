# Pinecone text client

Text utilities to work with Pinecone.

## Sparse encoding

To convert your own text corpus to sparse vectors, you can either use BM25 or Splade. 
For more information, see the [Pinecone documentation](https://docs.pinecone.io/docs/hybrid-search).

### BM25

```python
from pinecone_text.sparse import BM25

corpus = ["The quick brown fox jumps over the lazy dog",
          "The lazy dog is brown",
          "The fox is brown"]

# Initialize BM25 and fit the corpus
bm25 = BM25(tokenizer=lambda x: x.split())
bm25.fit(corpus)

# Encode a new document (for upsert to Pinecone index)
doc_sparse_vector = bm25.encode_documents("The brown fox is quick")
# {"indices": [102, 18, 12, ...], "values": [0.21, 0.38, 0.15, ...]}

# Encode a query (for search in Pinecone index)
query_sparse_vector = bm25.encode_queries("Which fox is brown?")
# {"indices": [102, 16, 18, ...], "values": [0.21, 0.11, 0.15, ...]}

# store BM25 params as json
bm25.store_params("bm25_params.json")

# load BM25 params from json
bm25.load_params("bm25_params.json")
```

### Splade

```python
from pinecone_text.sparse import SPLADE

corpus = ["The quick brown fox jumps over the lazy dog",
          "The lazy dog is brown",
          "The fox is brown"]

# Initialize Splade
splade = SPLADE()

# encode a batch of documents/queries
sparse_vectors = splade(corpus)
# [{"indices": [102, 18, 12, ...], "values": [0.21, 0.38, 0.15, ...]}, ...]
```