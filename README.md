<h1 align="center">
  <img src="https://avatars.githubusercontent.com/u/54333248?s=200&v=4">
    <br>
    Pinecone Text Client
    <br>
</h1>

The Pinecone Text Client is a Python package that provides text utilities designed for seamless integration with Pinecone's [sparse-dense](https://docs.pinecone.io/docs/hybrid-search) (hybrid) semantic search.

> **_⚠️ Warning_**
>
> This is a **public preview** ("Beta") version.   
> For any issues or requests, please reach out to our [support](support@pinecone.io) team.
## Installation
To install the Pinecone Text Client, use the following command:
```bash
pip install pinecone-text
```

If you wish to use `SpladeEncoder`, you will need to install the `splade` extra:
```bash
pip install pinecone-text[splade]
```

If you wish to use `SentenceTransformerEncoder` dense encoder, you will need to install the `dense` extra:
```bash
pip install pinecone-text[dense]
```

If you wish to use `OpenAIEncoder` dense encoder, you will need to install the `openai` extra:
```bash
pip install pinecone-text[openai]
```

## Sparse Encoding

To convert your own text corpus to sparse vectors, you can either use [BM25](https://www.pinecone.io/learn/semantic-search/#bm25) or [SPLADE](https://www.pinecone.io/learn/splade/).

### BM25
To encode your documents and queries using BM25 as vector for dot product search, you can use the `BM25Encoder` class.

> **_📝 NOTE:_**
> 
> Our current implementation of BM25 supports only static document frequency (meaning that the document frequency values are precomputed and fixed, and do not change dynamically based on new documents added to the collection).
>
> When conducting a search, you may come across queries that contain terms not found in the training corpus but are present in the database. To address this scenario, BM25Encoder uses a default document frequency value of 1 when encoding such terms. 
#### Usage

For an end-to-end example, you can refer to our Quora dataset generation with BM25 [notebook](https://colab.research.google.com/github/pinecone-io/examples/blob/master/learn/search/semantic-search/sparse/bm25/bm25-vector-generation.ipynb).

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
# {"indices": [102, 18, 12, ...], "values": [0.22, 0.38, 0.15, ...]}

# Encode a query (for search in Pinecone index)
query_sparse_vector = bm25.encode_queries("Which fox is brown?")
# {"indices": [102, 16, 18, ...], "values": [0.22, 0.11, 0.15, ...]}

# store BM25 params as json
bm25.dump("bm25_params.json")

# load BM25 params from json
bm25.load("bm25_params.json")
```

#### Load Default Parameters
If you want to use the default parameters for `BM25Encoder`, you can call the `default` method.
The default parameters were fitted on the [MS MARCO](https://microsoft.github.io/msmarco/)  passage ranking dataset.
```python
from pinecone_text.sparse import BM25Encoder
bm25 = BM25Encoder.default()
```

#### BM25 Parameters
The `BM25Encoder` class offers configurable parameters to customize the encoding:

* `b`: Controls document length normalization (default: 0.75).
* `k1`: Controls term frequency saturation (default: 1.2).
* Tokenization Options: Allows customization of the tokenization process, including options for handling case, punctuation, stopwords, stemming, and language selection.

These parameters can be specified when initializing the BM25Encoder class. Please read the BM25Encoder documentation for more details.

### SPLADE

Currently the `SpladeEncoder` class supprts only the [naver/splade-cocondenser-ensembledistil](https://huggingface.co/naver/splade-cocondenser-ensembledistil) model, and follows [SPLADE V2](https://arxiv.org/abs/2109.10086) implementation.

> **_📝 NOTE:_**
> 
> Currently pinecone text not supoorts SPLADE with python 3.12 due to compatibility issues with pytorch
>

#### Usage

For an end-to-end example, you can refer to our Quora dataset generation with SPLADE [notebook](https://colab.research.google.com/github/pinecone-io/examples/blob/master/learn/search/semantic-search/sparse/splade/splade-vector-generation.ipynb).

Note: If cuda is available, the model will automatically run on GPU. You can explicitly override the device using the `device` parameter in the constructor.

```python
from pinecone_text.sparse import SpladeEncoder

# Initialize Splade
splade = SpladeEncoder()

# encode a batch of documents
documents = ["The quick brown fox jumps over the lazy dog",
             "The lazy dog is brown",
             "The fox is brown"]
document_vectors = splade.encode_documents(documents)
# [{"indices": [102, 18, 12, ...], "values": [0.21, 0.38, 0.15, ...]}, ...]

# encode a query
query = "Which fox is brown?"
query_vectors = splade.encode_queries(query)
# {"indices": [102, 18, 12, ...], "values": [0.21, 0.38, 0.15, ...]}
```


## Dense Encoding

For dense embedding we also provide a thin wrapper for the following models:
1. All Sentence Transformers models hosted on huggingface [See full list of models](https://huggingface.co/sentence-transformers)
2. All OpenAI API supported embedding models [See full list of models](https://platform.openai.com/docs/models/embeddings)

### Sentence Transformers models

When using `SentenceTransformerEncoder`, the models are downloaded from huggingface and run locally. Also, if cuda is available, the model will automatically run on GPU. You can explicitly override the device using the `device` parameter in the constructor.

> **_📝 NOTE:_**
> 
> Currently pinecone text not supoorts sentence transformers with python 3.12 due to compatibility issues with pytorch
>

#### Usage
```python
from pinecone_text.dense import SentenceTransformerEncoder

encoder = SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2")

encoder.encode_documents(["The quick brown fox jumps over the lazy dog"])
# [[0.21, 0.38, 0.15, ...]]

encoder.encode_queries(["Who jumped over the lazy dog?"])
# [[0.11, 0.43, 0.67, ...]]
```

### OpenAI models

When using the `OpenAIEncoder`, you need to provide an API key for the OpenAI API, and store it in the `OPENAI_API_KEY` environment variable before you import the encoder.

By default the encoder will use `text-embedding-3-small` as recommended by OpenAI. You can also specify a different model name using the `model_name` parameter.
#### Usage
```python
from pinecone_text.dense import OpenAIEncoder

encoder = OpenAIEncoder() # defaults to the recommended model - "text-embedding-3-small"

encoder.encode_documents(["The quick brown fox jumps over the lazy dog"])
# [[0.21, 0.38, 0.15, ...]]

encoder.encode_queries(["Who jumped over the lazy dog?"])
# [[0.11, 0.43, 0.67, ...]]
```

Pinecone text also supports Azure OpenAI API. To use it, you need to import the `AzureOpenAIEncoder` class instead of `OpenAIEncoder`. You also need to pass Azure specific environment variables to the constructor, along with your specific embeddings  deployment as the model name. For more information please follow the `AzureOpenAIEncoder` documentation.


### Jina AI models

When using the `JinaEncoder`, you need to provide an API key for the Jina Embeddings API, and store it in the `JINA_API_KEY` environment variable before you import the encoder.

By default the encoder will use `jina-embeddings-v2-base-en`. You can also specify a different model name using the `model_name` parameter.

#### Usage

```python
from pinecone_text.dense import JinaEncoder

encoder = JinaEncoder()

encoder.encode_documents(["The quick brown fox jumps over the lazy dog"])
# [[-0.62586284, -0.54578537, 0.5570845, ...]]

encoder.encode_queries(["Who jumped over the lazy dog?"])
# [[-0.43374294, -0.42069837, 0.773763, ...]]
```

## Combining Sparse and Dense Encodings for Hybrid Search
To combine sparse and dense encodings for hybrid search, you can use the `hybrid_convex_scale` method on your query.

This method receives both a dense vector and a sparse vector, along with a convex scaling parameter `alpha`. It returns a tuple consisting of the scaled dense and sparse vectors according to the following formula: `(alpha * dense_vector, (1 - alpha) * sparse_vector)`.
```python
from pinecone_text.hybrid import hybrid_convex_scale
from pinecone_text.sparse import SpladeEncoder
from pinecone_text.dense import SentenceTransformerEncoder

# Initialize Splade
splade = SpladeEncoder()

# Initialize Sentence Transformer
sentence_transformer = SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2")

# encode a query
sparse_vector = splade.encode_queries("Which fox is brown?")
dense_vector = sentence_transformer.encode_queries("Which fox is brown?")

# combine sparse and dense vectors
hybrid_dense, hybrid_sparse = hybrid_convex_scale(dense_vector, sparse_vector, alpha=0.8)
# ([-0.21, 0.38, 0.15, ...], {"indices": [102, 16, 18, ...], "values": [0.21, 0.11, 0.18, ...]})
```
