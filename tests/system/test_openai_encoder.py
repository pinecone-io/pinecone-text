import pytest
from pinecone_text.dense import OpenAIEncoder
from openai import BadRequestError, AuthenticationError


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
    encoder = OpenAIEncoder(
        api_key="test_api_key", organization="test_organization", timeout=30
    )
    assert encoder._client.api_key == "test_api_key"
    assert encoder._client.organization == "test_organization"
    assert encoder._client.timeout == 30

    with pytest.raises(AuthenticationError):
        encoder.encode_documents("test text")


def encode_by_type(openai_encoder, encoding_function, test_input):
    if encoding_function == "encode_documents":
        return openai_encoder.encode_documents(test_input)
    elif encoding_function == "encode_queries":
        return openai_encoder.encode_queries(test_input)
    pytest.fail(f"Unknown encoding function: {encoding_function}")


@pytest.mark.parametrize(
    "encoding_function",
    [
        ("encode_documents"),
        ("encode_queries"),
    ],
)
def test_encode_single_text(openai_encoder, encoding_function):
    result = encode_by_type(openai_encoder, encoding_function, "test text")

    assert isinstance(result, list)
    assert len(result) == DEFAULT_DIMENSION


@pytest.mark.parametrize(
    "encoding_function",
    [
        ("encode_documents"),
        ("encode_queries"),
    ],
)
def test_encode_documents_multiple_texts(openai_encoder, encoding_function):
    result = encode_by_type(openai_encoder, encoding_function, ["text1", "text2"])
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(len(sub_result) == DEFAULT_DIMENSION for sub_result in result)


@pytest.mark.parametrize(
    "encoding_function",
    [
        ("encode_documents"),
        ("encode_queries"),
    ],
)
def test_encode_invalid_input(openai_encoder, encoding_function):
    with pytest.raises(ValueError):
        encode_by_type(openai_encoder, encoding_function, 123)


@pytest.mark.parametrize(
    "encoding_function",
    [
        ("encode_documents"),
        ("encode_queries"),
    ],
)
def test_encode_error_handling(openai_encoder, encoding_function):
    with pytest.raises(BadRequestError):
        encode_by_type(openai_encoder, encoding_function, "text is too long" * 10000)
