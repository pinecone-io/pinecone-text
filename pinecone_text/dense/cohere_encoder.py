from enum import Enum
from typing import Union, List, Any, Optional
from pinecone_text.dense.base_dense_ecoder import BaseDenseEncoder

try:
    import cohere
    from cohere.error import CohereError
except (OSError, ImportError, ModuleNotFoundError) as e:
    _cohere_installed = False
else:
    _cohere_installed = True


class CohereEncoderName(Enum):
    """
    Supported Cohere encoder models.
    """

    ENGLISH_V3 = "embed-english-v3.0"
    ENGLISH_LIGHT_V3 = "embed-english-light-3.0"
    MULTILINGUAL_V3 = "embed-multilingual-v3.0"
    MULTILINGUAL_LIGHT_V3 = "embed-multilingual-light-v3.0"

    @classmethod
    def list_models(cls) -> List[str]:
        """Method to get a list of all model names."""
        return [model.value for model in cls]


class CohereInputType(Enum):
    SEARCH_DOCUMENT = "search_document"
    SEARCH_QUERY = "search_query"
    # CLASSIFICATION = "classification"
    # CLUSTERING = "clustering"


class CohereEncoder(BaseDenseEncoder):
    """
    Cohere's text embedding wrapper. See https://docs.cohere.com/reference/embed

    Note: You should provide an API key as the environment variable `CO_API_KEY`.
          Or you can pass it as argument to the constructor as `api_key`.
    """

    def __init__(
        self,
        model_name: str = "embed-english-v3.0",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize the Cohere encoder.

        :param model_name: The name of the embedding model to use. See https://docs.cohere.com/reference/embed
        :param kwargs: Additional arguments to pass to the underlying cohere client. See https://github.com/openai/openai-python
        """
        if not _cohere_installed:
            raise ImportError(
                "Failed to import cohere. Make sure you install cohere extra "
                "dependencies by running: "
                "`pip install pinecone-text[cohere]"
            )
        if model_name not in CohereEncoderName.list_models():
            raise ValueError(
                f"Model '{model_name}' not supported. Please use one of:"
                + "\n"
                + "\n".join([f"- {x}" for x in CohereEncoderName.list_models()])
            )
        super().__init__()
        self._model_name = model_name
        self._client = cohere.Client(api_key=api_key, **kwargs)

    def encode_documents(
        self, texts: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        return self._encode(texts, CohereInputType.SEARCH_DOCUMENT.value)

    def encode_queries(
        self, texts: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        return self._encode(texts, CohereInputType.SEARCH_QUERY.value)

    def _encode(
        self, texts: Union[str, List[str]], input_type: str
    ) -> Union[List[float], List[List[float]]]:
        if isinstance(texts, str):
            texts_input = [texts]
        elif isinstance(texts, list):
            texts_input = texts
        else:
            raise ValueError(
                f"texts must be a string or list of strings, got: {type(texts)}"
            )

        try:
            response = self._client.embed(
                texts=texts_input,
                model=self._model_name,
                input_type=input_type,
            )
        except CohereError as e:
            # TODO: consider wrapping external provider errors
            raise e

        if isinstance(texts, str):
            return response.embeddings[0]
        return [embedding for embedding in response.embeddings]
