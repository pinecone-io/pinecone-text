import pytest
from unittest.mock import patch, Mock
from pinecone_text.dense import OpenAIEncoder, AzureOpenAIEncoder


def create_mock_response(embeddings):
    mock_response = Mock()
    mock_response.data = [Mock(embedding=embedding) for embedding in embeddings]
    return mock_response


mock_single_embedding = create_mock_response([[0.1, 0.2, 0.3]])
mock_multiple_embeddings = create_mock_response([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


@pytest.fixture(params=[OpenAIEncoder, AzureOpenAIEncoder])
def encoder(request):
    encoder_class = request.param
    with patch("pinecone_text.dense.openai_encoder.openai"):
        return encoder_class(model_name="test_model_name")


def test_init_without_openai_installed():
    with patch("pinecone_text.dense.openai_encoder._openai_installed", False):
        with pytest.raises(ImportError):
            OpenAIEncoder()


@pytest.mark.parametrize("encoder_class", [OpenAIEncoder, AzureOpenAIEncoder])
def test_init_with_kwargs(encoder_class):
    with patch("pinecone_text.dense.openai_encoder.openai") as mock_openai:
        encoder_class(
            api_key="test_api_key",
            organization="test_organization",
            timeout=30,
            model_name="test_model_name",
        )
        if encoder_class == OpenAIEncoder:
            mock_openai.OpenAI.assert_called_with(
                api_key="test_api_key",
                organization="test_organization",
                timeout=30,
            )
        else:
            mock_openai.AzureOpenAI.assert_called_with(
                api_key="test_api_key",
                organization="test_organization",
                timeout=30,
            )


def encode_by_type(encoder, encoding_function, test_input):
    func = getattr(encoder, encoding_function)
    return func(test_input)


@pytest.mark.parametrize(
    "encoding_function",
    [
        ("encode_documents"),
        ("encode_queries"),
    ],
)
def test_encode_single_text(encoder, encoding_function):
    with patch.object(encoder._client, "embeddings", create=True) as mock_embeddings:
        mock_embeddings.create.return_value = mock_single_embedding
        result = encode_by_type(encoder, encoding_function, "test text")
        assert result == [0.1, 0.2, 0.3]


@pytest.mark.parametrize(
    "encoding_function",
    [
        ("encode_documents"),
        ("encode_queries"),
    ],
)
def test_encode_multiple_texts(encoder, encoding_function):
    with patch.object(encoder._client, "embeddings", create=True) as mock_embeddings:
        mock_embeddings.create.return_value = mock_multiple_embeddings
        result = encode_by_type(encoder, encoding_function, ["text1", "text2"])
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


@pytest.mark.parametrize(
    "encoding_function",
    [
        ("encode_documents"),
        ("encode_queries"),
    ],
)
def test_encode_invalid_input(encoder, encoding_function):
    with pytest.raises(ValueError):
        encode_by_type(encoder, encoding_function, 123)


@pytest.mark.parametrize(
    "encoding_function",
    [
        ("encode_documents"),
        ("encode_queries"),
    ],
)
def test_encode_error_handling(encoder, encoding_function):
    with patch.object(encoder._client, "embeddings", create=True) as mock_embeddings:
        mock_embeddings.create.side_effect = ValueError("OpenAI API error")
        with pytest.raises(ValueError, match="OpenAI API error"):
            encode_by_type(encoder, encoding_function, "test text")


def test_openai_encoder_invalid_dimension():
    with pytest.raises(AssertionError):
        OpenAIEncoder(model_name="text-embedding-3-small", dimension=0)
