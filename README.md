<h1 align="center">
  <img src="https://avatars.githubusercontent.com/u/54333248?s=200&v=4">
    <br>
    Pinecone Text Client
    <br>
</h1>

A Python text utilities client designed for use with Pinecone. This package streamlines text processing and encoding, enabling seamless integration with Pinecone's hybrid search.

> **_⚠️ Warning_**
>
> This is a **public preview** ("Beta") version.   
> For any issues or requests, please reach out to our [support](support@pinecone.io) team.

## Installation

```bash
pip install -U pinecone-text
```

## Sparse encoding

To convert your own text corpus to sparse vectors, you can either use BM25 or Splade. 
For more information, see the [Pinecone documentation](https://docs.pinecone.io/docs/hybrid-search).

### BM25
To encode your documents and queries using BM25 as vector for dot product search, you can use the `BM25Encoder` class.

> **_📝 NOTE:_**
> 
> Our current implementation of BM25 supports only static document frequency (meaning that the document frequency values are precomputed and fixed, and do not change dynamically based on new documents added to the collection).
>

#### Usage

```python
from pinecone_text.sparse import BM25Encoder

corpus = ["The quick brown fox jumps over the lazy dog",
          "The lazy dog is brown",
          "The fox is brown"]

# Initialize BM25 and fit the corpus
bm25 = BM25Encoder()
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
If you want to use the default parameters for BM25Encoder, you can call the `default` method.
The default parameters were fitted on the [MS MARCO](https://microsoft.github.io/msmarco/)  passage ranking dataset.
```python
bm25 = BM25Encoder.default()
```

### Splade

```python
from pinecone_text.sparse import SpladeEncoder

corpus = ["The quick brown fox jumps over the lazy dog",
          "The lazy dog is brown",
          "The fox is brown"]

# Initialize Splade
splade = SpladeEncoder()

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
