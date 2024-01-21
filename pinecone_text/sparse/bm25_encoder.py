import json
import mmh3
import numpy as np
import tempfile
from pathlib import Path
from tqdm.auto import tqdm
import wget
from typing import List, Optional, Dict, Union, Tuple
from collections import Counter

from pinecone_text.sparse import SparseVector
from pinecone_text.sparse.base_sparse_encoder import BaseSparseEncoder
from pinecone_text.sparse.bm25_tokenizer import BM25Tokenizer


class BM25Encoder(BaseSparseEncoder):

    """OKAPI BM25 implementation for single fit to a corpus (no continuous corpus updates supported)"""

    def __init__(
        self,
        b: float = 0.75,
        k1: float = 1.2,
        lower_case: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        stem: bool = True,
        language: str = "english",
    ):
        """
        OKapi BM25 with mmh3 hashing

        Args:
            b: The length normalization parameter
            k1: The term frequency normalization parameter
            lower_case: Whether to lower case the tokens
            remove_punctuation: Whether to remove punctuation tokens
            remove_stopwords: Whether to remove stopwords tokens
            stem: Whether to stem the tokens (using SnowballStemmer)
            language: The language of the text (used for stopwords and stemmer)

        Example:

            ```python
            from pinecone_text.sparse import BM25

            bm25 = BM25Encoder()

            bm25.fit([ "The quick brown fox jumps over the lazy dog", "The lazy dog is brown"])

            bm25.encode_documents("The brown fox is quick") # {"indices": [102, 18, 12, ...], "values": [0.21, 0.38, 0.15, ...]}
            bm25.encode_queries("Which fox is brown?") # # {"indices": [102, 16, 18, ...], "values": [0.21, 0.11, 0.15, ...]}
            ```
        """
        # Fixed params
        self.b: float = b
        self.k1: float = k1

        self._tokenizer = BM25Tokenizer(
            lower_case=lower_case,
            remove_punctuation=remove_punctuation,
            remove_stopwords=remove_stopwords,
            stem=stem,
            language=language,
        )

        # Learned Params
        self.doc_freq: Optional[Dict[int, float]] = None
        self.n_docs: Optional[int] = None
        self.avgdl: Optional[float] = None

    def fit(self, corpus: List[str]) -> "BM25Encoder":
        """
        Fit BM25 by calculating document frequency over the corpus

        Args:
            corpus: list of texts to fit BM25 with
        """
        n_docs = 0
        sum_doc_len = 0
        doc_freq_counter: Counter = Counter()

        for doc in tqdm(corpus):
            if not isinstance(doc, str):
                raise ValueError("corpus must be a list of strings")

            indices, tf = self._tf(doc)
            if len(indices) == 0:
                continue
            n_docs += 1
            sum_doc_len += sum(tf)

            # Count the number of documents that contain each token
            doc_freq_counter.update(indices)

        self.doc_freq = dict(doc_freq_counter)
        self.n_docs = n_docs
        self.avgdl = sum_doc_len / n_docs
        return self

    def encode_documents(
        self, texts: Union[str, List[str]]
    ) -> Union[SparseVector, List[SparseVector]]:
        """
        encode documents to a sparse vector (for upsert to pinecone)

        Args:
            texts: a single or list of documents to encode as a string
        """
        if self.doc_freq is None or self.n_docs is None or self.avgdl is None:
            raise ValueError("BM25 must be fit before encoding documents")

        if isinstance(texts, str):
            return self._encode_single_document(texts)
        elif isinstance(texts, list):
            return [self._encode_single_document(text) for text in texts]
        else:
            raise ValueError("texts must be a string or list of strings")

    def _encode_single_document(self, text: str) -> SparseVector:
        indices, doc_tf = self._tf(text)
        tf = np.array(doc_tf)
        tf_sum = sum(tf)

        tf_normed = tf / (
            self.k1 * (1.0 - self.b + self.b * (tf_sum / self.avgdl)) + tf
        )
        return {
            "indices": indices,
            "values": tf_normed.tolist(),
        }

    def encode_queries(
        self, texts: Union[str, List[str]]
    ) -> Union[SparseVector, List[SparseVector]]:
        """
        encode query to a sparse vector

        Args:
            texts: a single or list of queries to encode as a string
        """
        if self.doc_freq is None or self.n_docs is None or self.avgdl is None:
            raise ValueError("BM25 must be fit before encoding queries")

        if isinstance(texts, str):
            return self._encode_single_query(texts)
        elif isinstance(texts, list):
            return [self._encode_single_query(text) for text in texts]
        else:
            raise ValueError("texts must be a string or list of strings")

    def _encode_single_query(self, text: str) -> SparseVector:
        indices, query_tf = self._tf(text)

        df = np.array([self.doc_freq.get(idx, 1) for idx in indices])  # type: ignore
        idf = np.log((self.n_docs + 1) / (df + 0.5))  # type: ignore
        idf_norm = idf / idf.sum()
        return {
            "indices": indices,
            "values": idf_norm.tolist(),
        }

    def dump(self, path: str) -> None:
        """
        Store BM25 params to a file in JSON format

        Args:
            path: full file path to save params in
        """
        with open(path, "w") as f:
            json.dump(self.get_params(), f)

    def load(self, path: str) -> "BM25Encoder":
        """
        Load BM25 params from a file in JSON format

        Args:
            path: full file path to load params from
        """
        with open(path, "r") as f:
            params = json.load(f)
        return self.set_params(**params)

    def get_params(
        self,
    ) -> Dict[str, Union[int, float, str, Dict[str, List[Union[int, float]]]]]:
        """Returns the BM25 params"""

        if self.doc_freq is None or self.n_docs is None or self.avgdl is None:
            raise ValueError("BM25 must be fit before storing params")

        tf_pairs = list(self.doc_freq.items())
        return {
            "avgdl": self.avgdl,
            "n_docs": self.n_docs,
            "doc_freq": {
                "indices": [int(idx) for idx, _ in tf_pairs],
                "values": [float(val) for _, val in tf_pairs],
            },
            "b": self.b,
            "k1": self.k1,
            "lower_case": self._tokenizer.lower_case,
            "remove_punctuation": self._tokenizer.remove_punctuation,
            "remove_stopwords": self._tokenizer.remove_stopwords,
            "stem": self._tokenizer.stem,
            "language": self._tokenizer.language,
        }

    def set_params(
        self,
        avgdl: float,
        n_docs: int,
        doc_freq: Dict[str, List[int]],
        b: float,
        k1: float,
        lower_case: bool,
        remove_punctuation: bool,
        remove_stopwords: bool,
        stem: bool,
        language: str,
    ) -> "BM25Encoder":
        """
        Set input parameters to BM25

        Args:
            avgdl: average document length in the corpus
            n_docs: number of documents in the corpus
            doc_freq: document frequency of each term in the corpus
            b: length normalization parameter
            k1: term frequency normalization parameter
            lower_case: whether to lower case the text
            remove_punctuation: whether to remove punctuation from the text
            remove_stopwords: whether to remove stopwords from the text
            stem: whether to stem the text
            language: language of the text for stopwords and stemmer
        """
        self.avgdl = avgdl  # type: ignore
        self.n_docs = n_docs  # type: ignore
        self.doc_freq = {
            idx: val
            for idx, val in zip(doc_freq["indices"], doc_freq["values"])  # type: ignore
        }
        self.b = b  # type: ignore
        self.k1 = k1  # type: ignore
        self._tokenizer = BM25Tokenizer(
            lower_case=lower_case,  # type: ignore
            remove_punctuation=remove_punctuation,  # type: ignore
            remove_stopwords=remove_stopwords,  # type: ignore
            stem=stem,  # type: ignore
            language=language,
        )  # type: ignore
        return self

    @staticmethod
    def default() -> "BM25Encoder":
        """Create a BM25 model from pre-made params for the MS MARCO passages corpus"""
        bm25 = BM25Encoder()
        url = "https://storage.googleapis.com/pinecone-datasets-dev/bm25_params/msmarco_bm25_params_v4_0_0.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir, "msmarco_bm25_params.json")
            wget.download(url, str(tmp_path))
            bm25.load(str(tmp_path))
        return bm25

    @staticmethod
    def _hash_text(token: str) -> int:
        """Use mmh3 to hash text to 32-bit unsigned integer"""
        return mmh3.hash(token, signed=False)

    def _tf(self, text: str) -> Tuple[List[int], List[int]]:
        """
        Calculate term frequency for a given text

        Args:
            text: a document to calculate term frequency for

        Returns: a tuple of two lists:
            indices: list of term indices
            values: list of term frequencies
        """
        counts = Counter((self._hash_text(token) for token in self._tokenizer(text)))

        items = list(counts.items())
        return [idx for idx, _ in items], [val for _, val in items]
