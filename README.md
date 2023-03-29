# Pinecone text client

:warning: This library is under heavy development and is subject to change.

This library provides tools for encoding text data for Pinecone. It includes: encoding text to sparse vectors using BM25 or Splade, and encoding text to dense vectors using Sentence Transformers. 

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
bm25.dump("bm25_params.json")

# load BM25 params from json
bm25.load("bm25_params.json")
```

#### Load default parameters
If you want to use the default parameters for BM25, you can call the `default` method.
The default parameters were fitted on the [MS MARCO](https://microsoft.github.io/msmarco/)  passage ranking dataset.
```python
bm25 = BM25.default()
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


## Dense Encoding with Sentence Transformers

This is a thin wrapper over the Sentence Transformers models hosted on hugging face. [See full list of models](https://huggingface.co/sentence-transformers)

```python
from pinecone_text.dense.sentence_transformer_encoder import SentenceTransformerEncoder

encoder = SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2")

encoder.encode_documents(["The quick brown fox jumps over the lazy dog"])

encoder.encode_queries(["Who jumped over the lazy dog?"])
```
