import json
import numpy as np
from scipy import sparse
from typing import List, Callable, Optional, Dict, Union, Tuple
from sklearn.feature_extraction.text import HashingVectorizer

from pinecone_text.sparse import SparseVector


class BM25:
    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        vocabulary_size: int = 2**18,
        b: float = 0.75,
        k1: float = 1.6,
        lowercase_tokens: bool = True,
    ):
        """OKapi BM25 with HashingVectorizer

        Args:
            tokenizer: A function to converts text to a list of tokens
            vocabulary_size: The hash size to which the tokens are mapped to
            b: The length normalization parameter
            k1: The term frequency normalization parameter
            lowercase_tokens: Whether to lowercase tokens before hashing
        """
        if vocabulary_size > 2**32 - 1:
            raise ValueError("vocabulary_size must be less than 2^32 - 1")
        elif vocabulary_size < 0:
            raise ValueError("vocabulary_size must be a positive integer")

        # Fixed params
        self.vocabulary_size: int = vocabulary_size
        self.b: float = b
        self.k1: float = k1

        self._tokenizer: Callable[[str], List[str]] = tokenizer
        self._vectorizer = HashingVectorizer(
            n_features=self.vocabulary_size,
            token_pattern=None,
            tokenizer=tokenizer,
            norm=None,
            alternate_sign=False,
            binary=True,
            lowercase=lowercase_tokens,
        )

        # Learned Params
        self.doc_freq: Optional[Dict[int, float]] = None
        self.n_docs: Optional[int] = None
        self.avgdl: Optional[float] = None

    def fit(self, corpus: List[str]) -> "BM25":
        """Fit BM25 by calculating document frequency over the corpus"""
        doc_tf_matrix = self._vectorizer.transform(corpus)
        self.avgdl = doc_tf_matrix.sum(1).mean().item()
        self.n_docs = doc_tf_matrix.shape[0]
        tf_vector = sparse.csr_matrix(doc_tf_matrix.sum(axis=0))
        self.doc_freq = {
            int(idx): float(val) for idx, val in zip(tf_vector.indices, tf_vector.data)
        }
        return self

    def encode_document(self, doc: str) -> dict[str, Union[list[int], list[float]]]:
        """Normalize document for BM25 scoring"""
        if self.doc_freq is None or self.n_docs is None or self.avgdl is None:
            raise ValueError("BM25 must be fit before encoding documents")

        doc_tf = self._vectorizer.transform([doc])
        norm_doc_tf = self._norm_doc_tf(doc_tf)
        return {
            "indices": [int(x) for x in doc_tf.indices],
            "values": [float(x) for x in norm_doc_tf.tolist()],
        }

    def encode_query(self, query: str) -> SparseVector:
        """Normalize query for BM25 scoring"""
        if self.doc_freq is None or self.n_docs is None or self.avgdl is None:
            raise ValueError("BM25 must be fit before encoding queries")

        query_tf = self._vectorizer.transform([query])
        indices, values = self._norm_query_tf(query_tf)
        return {
            "indices": [int(x) for x in indices],
            "values": [float(x) for x in values],
        }

    def store_params(self, path: str) -> None:
        """Store BM25 params to a file in JSON format"""
        with open(path, "w") as f:
            json.dump(self.get_params(), f)

    def load_params(self, path: str) -> "BM25":
        """Load BM25 params from a file in JSON format"""
        with open(path, "r") as f:
            params = json.load(f)
        return self.set_params(**params)

    def get_params(
        self,
    ) -> Dict[str, Union[int, float, Dict[str, List[Union[int, float]]]]]:
        if self.doc_freq is None or self.n_docs is None or self.avgdl is None:
            raise ValueError("BM25 must be fit before storing params")

        tf_pairs = list(self.doc_freq.items())
        return {
            "avgdl": self.avgdl,
            "ndocs": self.n_docs,
            "doc_freq": {
                "indices": [int(idx) for idx, _ in tf_pairs],
                "values": [float(val) for _, val in tf_pairs],
            },
            "b": self.b,
            "k1": self.k1,
            "vocabulary_size": self.vocabulary_size,
        }

    def set_params(self, **params: Union[int, float, Dict[str, List[int]]]) -> "BM25":
        self.avgdl = params["avgdl"]  # type: ignore
        self.n_docs = params["ndocs"]  # type: ignore
        self.doc_freq = {
            idx: val
            for idx, val in zip(
                params["doc_freq"]["indices"], params["doc_freq"]["values"]  # type: ignore
            )
        }
        self.b = params["b"]  # type: ignore
        self.k1 = params["k1"]  # type: ignore
        self.vocabulary_size = params["vocabulary_size"]  # type: ignore
        return self

    def _norm_doc_tf(self, doc_tf: sparse.csr_matrix) -> np.ndarray:
        """Calculate BM25 normalized document term-frequencies"""
        b, k1, avgdl = self.b, self.k1, self.avgdl
        tf = doc_tf.data
        norm_tf = tf / (k1 * (1.0 - b + b * (tf.sum() / avgdl)) + tf)
        return norm_tf

    def _norm_query_tf(
        self, query_tf: sparse.csr_matrix
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate BM25 normalized query term-frequencies"""
        tf = np.array([self.doc_freq.get(idx, 0) for idx in query_tf.indices])  # type: ignore
        idf = np.log((self.n_docs + 1) / (tf + 0.5))  # type: ignore
        return query_tf.indices, idf / idf.sum()
