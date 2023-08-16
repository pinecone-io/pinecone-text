import openai
from typing import Union, List
from pinecone_text.dense.base_dense_ecoder import BaseDenseEncoder


class OpenAIEncoder(BaseDenseEncoder):
    """
    OpenAI's text embedding wrapper. See https://platform.openai.com/docs/guides/embeddings

    Note: You should provide an API key and organization in the environment variables OPENAI_API_KEY and OPENAI_ORG.
    """

    def __init__(self, model_name: str = "text-embedding-ada-002"):
        self._model_name = model_name

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
        texts_input = texts if isinstance(texts, list) else [texts]
        response = openai.Embedding.create(input=texts_input, model=self._model_name)
        if isinstance(texts, str):
            return response["data"][0]["embedding"]
        return [result["embedding"] for result in response["data"]]
