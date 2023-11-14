"""
# Sparse Vectorizers

Sparse vectorizers are used to convert a document into a sparse vector representation. This is useful for
indexing and searching large collections of documents. The sparse vector representation is a list of indices and
values. The indices are the token ids and the values are the token weights. The token weights are calculated
using the BM25 or SPLADE algorithms.

## BM25
Okapi BM25 is a probabilistic ranking function that is used to rank documents based on a query. BM25 is
a bag-of-words model that does not take into account the order of the words in the document. The BM25
algorithm is used to calculate the token weights in the sparse vector representation.

Important note:
Our BM25 implementation is not the same as the one in the original paper. We use a different TF-IDF representation
that are more suitable for vector representations. The BM25 implementation in this library is based work done by
Pinecone. For more information, see the [Pinecone documentation](https://docs.pinecone.io/docs/hybrid-search).

## SPLADE
SPLADE is a Transformer based encoder, that uses sophisticated expansion to encode documents and queries in a sparse representation.
This allows a semantic search to be performed on the sparse vectors. The SPLADE encoder is based on the work done by the research team in Naver Labs Europe.
For more information, see the [SPLADE paper](https://arxiv.org/abs/2109.10086). The SPLADE encoder is currently only available for inference only.
"""


from typing import Union, Dict, List

SparseVector = Dict[str, Union[List[int], List[float]]]

from .bm25_encoder import BM25Encoder  # noqa: F401
from .splade_encoder import SpladeEncoder  # noqa: F401
