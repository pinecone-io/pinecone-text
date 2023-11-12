import pytest
from pinecone_text.dense import OpenAIEncoder
from openai import OpenAIError


DEFAULT_DIMENSION = 1536


@pytest.fixture
def openai_encoder():
    return OpenAIEncoder()


def test_init_without_openai_installed():
    try:
        encoder = OpenAIEncoder()
        assert encoder is not None
    except ImportError:
        pytest.fail("Failed to initialize OpenAIEncoder due to missing OpenAI library")


def test_init_with_kwargs():
    client = OpenAIEncoder(api_key="test_api_key", organization="test_organization")
    assert client._client.api_key == "test_api_key"
    assert client._client.organization == "test_organization"


def test_encode_documents_single_text(openai_encoder):
    result = openai_encoder.encode_documents("test text")
    assert isinstance(result, list)
    assert len(result) == DEFAULT_DIMENSION


def test_encode_documents_multiple_texts(openai_encoder):
    result = openai_encoder.encode_documents(["text1", "text2"])
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(len(sub_result) == DEFAULT_DIMENSION for sub_result in result)


def test_encode_documents_invalid_input(openai_encoder):
    with pytest.raises(ValueError):
        openai_encoder.encode_documents(123)


def test_encode_documents_error_handling(openai_encoder):
    with pytest.raises(OpenAIError):
        openai_encoder.encode_documents("text is too long" * 10000)


def test_encode_queries_single_text(openai_encoder):
    result = openai_encoder.encode_queries("test text")
    assert isinstance(result, list)
    assert len(result) == DEFAULT_DIMENSION


def test_encode_queries_multiple_texts(openai_encoder):
    result = openai_encoder.encode_queries(["text1", "text2"])
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(len(sub_result) == DEFAULT_DIMENSION for sub_result in result)


def test_encode_queries_invalid_input(openai_encoder):
    with pytest.raises(ValueError):
        openai_encoder.encode_queries(123)


def test_encode_queries_error_handling(openai_encoder):
    with pytest.raises(OpenAIError):
        openai_encoder.encode_queries("text is too long" * 10000)
