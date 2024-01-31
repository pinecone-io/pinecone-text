from typing import Union, List, Any, Optional
from pinecone_text.dense.base_dense_ecoder import BaseDenseEncoder
import os
import requests

JINA_API_URL: str = "https://api.jina.ai/v1/embeddings"


class JinaEncoder(BaseDenseEncoder):
    """
    JinaAI's text embedding wrapper. See https://jina.ai/embeddings/

    Note: You should provide an API key in the environment variable JINA_API_KEY.
          Or you can pass it as argument to the constructor as `api_key`.
    """

    def __init__(
        self,
        model_name: str = "jina-embeddings-v2-base-en",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize the OpenAI encoder.

        :param model_name: The name of the embedding model to use.
        :param kwargs: Additional arguments
        """
        if api_key is None:
            api_key = os.environ.get("JINA_API_KEY", None)

        if api_key is None:
            raise ValueError(
                "JinaEncoder requires an API key to work. Please provide `api_key` argument or set `JINA_API_KEY` environment variable"
            )
        super().__init__()
        self._model_name = model_name
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept-Encoding": "identity",
                "Content-type": "application/json",
            }
        )

    def encode_documents(
        self, texts: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        return self._encode(texts)

    def encode_queries(
        self, texts: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        return self._encode(texts)

    def _encode(
        self, texts: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        if isinstance(texts, str):
            texts_input = [texts]
        elif isinstance(texts, list):
            texts_input = texts
        else:
            raise ValueError(
                f"texts must be a string or list of strings, got: {type(texts)}"
            )

        resp = self._session.post(  # type: ignore
            JINA_API_URL, json={"input": texts_input, "model": self._model_name}
        ).json()
        if "data" not in resp:
            raise RuntimeError(resp["detail"])

        embeddings = resp["data"]

        # Sort resulting embeddings by index
        sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])  # type: ignore

        # Return just the embeddings
        res = [result["embedding"] for result in sorted_embeddings]

        if isinstance(texts, str):
            res = res[0]
        return res
