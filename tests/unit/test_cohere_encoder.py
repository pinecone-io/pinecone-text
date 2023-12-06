import pytest
from unittest.mock import patch, Mock
from typing import List
from pinecone_text.dense import CohereEncoder


real_encoder = CohereEncoder()


def create_mock_response(embeddings):
    mock_response = Mock()
    mock_response.data = [Mock(embedding=embedding) for embedding in embeddings]
    return mock_response


mock_single_embedding = create_mock_response([[0.1, 0.2, 0.3]])
mock_multiple_embeddings = create_mock_response([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


@pytest.fixture
def cohere_encoder():
    with patch("pinecone_text.dense.cohere_encoder.cohere"):
        yield CohereEncoder()

@pytest.fixture
def real_encoder():
    yield CohereEncoder()

def test_init_without_cohere_installed():
    with patch("pinecone_text.dense.cohere_encoder._cohere_installed", False):
        with pytest.raises(ImportError):
            CohereEncoder()


def test_init_with_kwargs():
    with patch("pinecone_text.dense.cohere_encoder.cohere.Client") as mock_cohere:
        CohereEncoder(
            api_key="test_api_key", organization="test_organization", timeout=30
        )
        mock_cohere.assert_called_with(
            api_key="test_api_key",
            organization="test_organization",
            timeout=30,
        )


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
def test_encode_single_text(real_encoder, encoding_function):
    if encoding_function == "encode_documents":
        out = real_encoder.encode_documents("test text")
    else:
        out = real_encoder.encode_queries("test text")
    assert isinstance(out, list)


@pytest.mark.parametrize(
    "encoding_function",
    [
        ("encode_documents"),
        ("encode_queries"),
    ],
)
def test_encode_multiple_texts(real_encoder, encoding_function):
    if encoding_function == "encode_documents":
        out = real_encoder.encode_documents(["test text", "doc 2"])
    else:
        out = real_encoder.encode_queries(["test text", "doc 2"])
    assert isinstance(out, list)
    assert isinstance(out[0], list)


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


@pytest.mark.parametrize(
    "encoding_function",
    [
        ("encode_documents"),
        ("encode_queries"),
    ],
)
def test_encode_error_handling(cohere_encoder, encoding_function):
    with patch.object(
        cohere_encoder._client, "embeddings", create=True
    ) as mock_embeddings:
        mock_embeddings.create.side_effect = ValueError("Cohere API error")
        with pytest.raises(ValueError, match="Cohere API error"):
            encode_by_type(cohere_encoder, encoding_function, "test text")
