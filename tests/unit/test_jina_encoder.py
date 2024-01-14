import pytest
from unittest.mock import patch
import requests
import json
import os

from pinecone_text.dense import JinaEncoder


@pytest.mark.parametrize(
    "encoding_function",
    [
        ("encode_documents"),
        ("encode_queries"),
    ],
)
def test_encode_single_text(encoding_function):
    encoder = JinaEncoder(api_key="test_api_key")
    with patch("requests.sessions.Session.post") as mock_post:
        # Configure the mock to return a specific response
        mock_response = requests.Response()
        mock_response.status_code = 200
        mock_response._content = json.dumps(
            {
                "model": "jina-embeddings-v2-base-en",
                "object": "list",
                "usage": {"total_tokens": 6, "prompt_tokens": 6},
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}
                ],
            }
        ).encode()

        mock_post.return_value = mock_response
        result = getattr(encoder, encoding_function)("test text")
        assert result == [0.1, 0.2, 0.3]


@pytest.mark.parametrize(
    "encoding_function",
    [
        ("encode_documents"),
        ("encode_queries"),
    ],
)
def test_encode_multiple_text(encoding_function):
    encoder = JinaEncoder(api_key="test_api_key")
    with patch("requests.sessions.Session.post") as mock_post:
        # Configure the mock to return a specific response
        mock_response = requests.Response()
        mock_response.status_code = 200
        mock_response._content = json.dumps(
            {
                "model": "jina-embeddings-v2-base-en",
                "object": "list",
                "usage": {"total_tokens": 6, "prompt_tokens": 6},
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
                    {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]},
                ],
            }
        ).encode()

        mock_post.return_value = mock_response
        result = getattr(encoder, encoding_function)(["test text", "text 2"])
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_error_no_api_key():
    with pytest.raises(ValueError):
        _ = JinaEncoder()


@pytest.fixture()
def env_api_key():
    previous_jina_api_key = os.environ.get("JINA_API_KEY", None)
    if previous_jina_api_key is None:
        os.environ["JINA_API_KEY"] = "test_api_key"
    yield
    if previous_jina_api_key is None:
        os.environ.pop("JINA_API_KEY")
    else:
        os.environ["JINA_API_KEY"] = previous_jina_api_key


def test_init_with_api_key_from_environ(env_api_key):
    _ = JinaEncoder()
