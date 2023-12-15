import os
from typing import Union, List, Any, Optional
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

    Note: You should provide an API key and organization in the environment variables OPENAI_API_KEY and OPENAI_ORG.
          Or you can pass them as arguments to the constructor as `api_key` and `organization`.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize the OpenAI encoder.

        :param model_name: The name of the embedding model to use. See https://beta.openai.com/docs/api-reference/embeddings
        :param kwargs: Additional arguments to pass to the underlying openai client. See https://github.com/openai/openai-python
        """
        if not _openai_installed:
            raise ImportError(
                "Failed to import openai. Make sure you install openai extra "
                "dependencies by running: "
                "`pip install pinecone-text[openai]"
            )
        self._model_name = model_name
        self._client = openai.OpenAI(
            api_key=api_key, organization=organization, base_url=base_url, **kwargs
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

        batch_size = 16  # Azure OpenAI limit as of 2023-11-27
        result = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = self._client.embeddings.create(
                    input=batch, model=self._model_name
                )
            except OpenAIError as e:
                # TODO: consider wrapping external provider errors
                raise e

            if isinstance(batch, str):
                result.extend(response.data[0].embedding)
            result.extend([result.embedding for result in response.data])

        return result


class AzureOpenAIEncoder(OpenAIEncoder):
    """
    Azure OpenAI's text embedding wrapper.
    See https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/understand-embeddings

    Note: You should provide an API key in the environment variables AZURE_OPENAI_API_KEY.
          Or you can pass it as an arguments to the constructor as `api_key`.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize the OpenAI encoder.

        :param model_name: The name of the embedding model to use. See https://beta.openai.com/docs/api-reference/embeddings
        :param kwargs: Additional arguments to pass to the underlying openai client. See https://github.com/openai/openai-python
        """
        if not _openai_installed:
            raise ImportError(
                "Failed to import openai. Make sure you install openai extra "
                "dependencies by running: "
                "`pip install pinecone-text[openai]"
            )
        self._model_name = model_name
        self._client = openai.AzureOpenAI(
            api_key=api_key,
            api_version="2023-05-15",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            base_url=base_url,
            **kwargs,
        )
