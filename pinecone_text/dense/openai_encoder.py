import os
from typing import Union, List, Any, Optional, Dict
from pinecone_text.dense.base_dense_ecoder import BaseDenseEncoder

try:
    import openai
    from openai import OpenAIError
except (OSError, ImportError, ModuleNotFoundError) as e:
    _openai_installed = False
else:
    _openai_installed = True


class OpenAIEncoder(BaseDenseEncoder):
    """
    OpenAI's text embedding wrapper. See https://platform.openai.com/docs/guides/embeddings

    Note: this method reflects the OpenAI client initialization behaviour (See https://github.com/openai/openai-python/blob/main/src/openai/_client.py)
      On initialization, You may explicitly pass any argument that the OpenAI client accepts, or use the following environment variables:

    - `OPENAI_API_KEY` as `api_key`
    - `OPENAI_ORG_ID` as `organization`
    - `OPENAI_BASE_URL` as `base_url`

    Example:
        Using environment variables:
        >>> import os
        >>> from pinecone_text.dense import OpenAIEncoder
        >>> os.environ['OPENAI_API_KEY'] = "sk-..."
        >>> encoder = OpenAIEncoder()
        >>> encoder.encode_documents(["some text", "some other text"])

        Passing arguments explicitly:
        >>> from pinecone_text.dense import OpenAIEncoder
        >>> encoder = OpenAIEncoder(api_key="sk-...")
    """  # noqa: E501

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        *,
        dimension: Optional[int] = None,
        **kwargs: Any,
    ):
        if not _openai_installed:
            raise ImportError(
                "Failed to import openai. Make sure you install openai extra "
                "dependencies by running: "
                "`pip install pinecone-text[openai]"
            )
        super().__init__()

        if dimension is not None:
            assert dimension > 0, "dimension must be a positive integer"

        self._model_name = model_name
        self._dimension = dimension
        self._client = self._create_client(**kwargs)

    @staticmethod
    def _create_client(**kwargs: Any) -> Union[openai.OpenAI, openai.AzureOpenAI]:
        return openai.OpenAI(**kwargs)

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

        try:
            params: Dict[str, Any] = dict(
                input=texts_input,
                model=self._model_name,
            )
            if self._dimension is not None:
                params["dimensions"] = self._dimension
            response = self._client.embeddings.create(**params)
        except OpenAIError as e:
            # TODO: consider wrapping external provider errors
            raise e

        if isinstance(texts, str):
            return response.data[0].embedding
        return [result.embedding for result in response.data]


class AzureOpenAIEncoder(OpenAIEncoder):
    """
    Initialize the Azure OpenAI encoder.

    Note: this method reflects the AzureOpenAI client initialization behaviour (See https://github.com/openai/openai-python/blob/main/src/openai/lib/azure.py).
           You may explicitly pass any argument that the AzureOpenAI client accepts, or use the following environment variables:

        - `AZURE_OPENAI_API_KEY` as `api_key`
        - `AZURE_OPENAI_ENDPOINT` as `azure_endpoint`
        - `OPENAI_API_VERSION` as `api_version`
        - `OPENAI_ORG_ID` as `organization`
        - `AZURE_OPENAI_AD_TOKEN` as `azure_ad_token`

    In addition, you must pass the `model_name` argument with the name of the deployment you wish to use in your own Azure account.

    Example:
        Using environment variables:
        >>> import os
        >>> from pinecone_text.dense import AzureOpenAIEncoder
        >>> os.environ['AZURE_OPENAI_API_KEY'] = "sk-..."
        >>> os.environ['AZURE_OPENAI_ENDPOINT'] = "https://.....openai.azure.com/"
        >>> os.environ['OPENAI_API_VERSION'] = "2023-12-01-preview"
        >>> encoder = AzureOpenAIEncoder(model_name="my-ada-002-deployment")
        >>> encoder.encode_documents(["some text", "some other text"])

        Passing arguments explicitly:
        >>> from pinecone_text.dense import AzureOpenAIEncoder
        >>> encoder = AzureOpenAIEncoder(api_key="sk-...", azure_endpoint="https://.....openai.azure.com/", api_version="2023-12-01-preview")
    """  # noqa: E501

    def __init__(self, model_name: str, **kwargs: Any):
        super().__init__(model_name=model_name, **kwargs)

    @staticmethod
    def _create_client(**kwargs: Any) -> openai.AzureOpenAI:
        return openai.AzureOpenAI(**kwargs)
