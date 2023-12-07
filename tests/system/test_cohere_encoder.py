import pytest
from pinecone_text.dense import CohereEncoder
from cohere import CohereAPIError


DEFAULT_DIMENSION = 1024


@pytest.fixture
def cohere_encoder():
    return CohereEncoder()


def test_init_with_kwargs():
    encoder = CohereEncoder(api_key=None)
    assert encoder._client.api_key is not None


def test_init_with_wrong_api_key():
    encoder = CohereEncoder(api_key="test_api_key")
    assert encoder._client.api_key == "test_api_key"

    with pytest.raises(CohereAPIError):
        encoder.encode_documents("test text")


def encode_by_type(cohere_encoder, encoding_function, test_input):
    func = getattr(cohere_encoder, encoding_function)
    return func(test_input)


@pytest.mark.parametrize(
    "encoding_function",
    [
        ("encode_documents"),
        ("encode_queries"),
    ],
)
def test_encode_single_text(cohere_encoder, encoding_function):
    result = encode_by_type(cohere_encoder, encoding_function, "test text")

    assert isinstance(result, list)
    assert len(result) == DEFAULT_DIMENSION


@pytest.mark.parametrize(
    "encoding_function",
    [
        ("encode_documents"),
        ("encode_queries"),
    ],
)
def test_encode_documents_multiple_texts(cohere_encoder, encoding_function):
    result = encode_by_type(cohere_encoder, encoding_function, ["text1", "text2"])
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
def test_encode_invalid_input(cohere_encoder, encoding_function):
    with pytest.raises(ValueError):
        encode_by_type(cohere_encoder, encoding_function, 123)
