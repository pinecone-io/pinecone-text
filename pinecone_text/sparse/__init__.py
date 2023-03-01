from typing import Union, Dict, List

SparseVector = Dict[str, Union[List[int], List[float]]]

from pinecone_text.sparse.bm25 import BM25
from pinecone_text.sparse.splade import SPLADE
