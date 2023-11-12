import pytest
from unittest.mock import patch, Mock
from pinecone_text.dense import OpenAIEncoder
from openai._exceptions import OpenAIError


def create_mock_response(embeddings):
    mock_response = Mock()
    mock_response.data = [Mock(embedding=embedding) for embedding in embeddings]
    return mock_response


mock_single_embedding = create_mock_response([[0.1, 0.2, 0.3]])
mock_multiple_embeddings = create_mock_response([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


@pytest.fixture
def openai_encoder():
    with patch("pinecone_text.dense.openai_encoder.openai") as mock_openai:
        yield OpenAIEncoder()


def test_init_without_openai_installed():
    with patch("pinecone_text.dense.openai_encoder._openai_installed", False):
        with pytest.raises(ImportError):
            OpenAIEncoder()


def test_init_with_kwargs():
    with patch("pinecone_text.dense.openai_encoder.openai") as mock_openai:
        OpenAIEncoder(api_key="test_api_key", organization="test_organization")
        mock_openai.OpenAI.assert_called_with(
            api_key="test_api_key", organization="test_organization"
        )


def test_encode_documents_single_text(openai_encoder):
    with patch.object(
        openai_encoder._client, "embeddings", create=True
    ) as mock_embeddings:
        mock_embeddings.create.return_value = mock_single_embedding
        result = openai_encoder.encode_documents("test text")
        assert result == [0.1, 0.2, 0.3]


def test_encode_documents_multiple_texts(openai_encoder):
    with patch.object(
        openai_encoder._client, "embeddings", create=True
    ) as mock_embeddings:
        mock_embeddings.create.return_value = mock_multiple_embeddings
        result = openai_encoder.encode_documents(["text1", "text2"])
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_encode_documents_invalid_input(openai_encoder):
    with pytest.raises(ValueError):
        openai_encoder.encode_documents(123)


def test_encode_documents_error_handling(openai_encoder):
    with patch.object(
        openai_encoder._client, "embeddings", create=True
    ) as mock_embeddings:
        mock_embeddings.create.side_effect = OpenAIError("OpenAI API error")
        with pytest.raises(OpenAIError, match="OpenAI API error"):
            openai_encoder.encode_queries("test text")


def test_encode_queries_single_text(openai_encoder):
    with patch.object(
        openai_encoder._client, "embeddings", create=True
    ) as mock_embeddings:
        mock_embeddings.create.return_value = mock_single_embedding
        result = openai_encoder.encode_queries("test text")
        assert result == [0.1, 0.2, 0.3]


def test_encode_queries_multiple_texts(openai_encoder):
    with patch.object(
        openai_encoder._client, "embeddings", create=True
    ) as mock_embeddings:
        mock_embeddings.create.return_value = mock_multiple_embeddings
        result = openai_encoder.encode_queries(["text1", "text2"])
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_encode_queries_invalid_input(openai_encoder):
    with pytest.raises(ValueError):
        openai_encoder.encode_queries(123)


def test_encode_queries_error_handling(openai_encoder):
    with patch.object(
        openai_encoder._client, "embeddings", create=True
    ) as mock_embeddings:
        mock_embeddings.create.side_effect = OpenAIError("OpenAI API error")
        with pytest.raises(OpenAIError, match="OpenAI API error"):
            openai_encoder.encode_queries("test text")
