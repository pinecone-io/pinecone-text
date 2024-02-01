from typing import List, Union, Any, Optional
from abc import ABC, abstractmethod
from functools import cached_property


class BaseDenseEncoder(ABC):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._dimension: Optional[int] = None

    @cached_property
    def dimension(self) -> int:
        if self._dimension is None:
            return len(self.encode_documents("hello"))
        return self._dimension

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
