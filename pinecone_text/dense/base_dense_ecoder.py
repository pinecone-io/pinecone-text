from typing import List, Union
from abc import ABC, abstractmethod


class BaseDenseEncoder(ABC):
    @abstractmethod
    def encode_documents(
        self, texts: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        encode documents to a dense vector (for upsert to pinecone)

        Args:
            texts: a single or list of documents to encode as a string
        """
        pass  # pragma: no cover

    @abstractmethod
    def encode_queries(
        self, texts: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        encode queries to a dense vector

        Args:
            texts: a single or list of queries to encode as a string
        """
        pass  # pragma: no cover
