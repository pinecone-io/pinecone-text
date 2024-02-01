import pytest
import os
from pinecone_text.dense import OpenAIEncoder, AzureOpenAIEncoder
from openai import BadRequestError, AuthenticationError

DEFAULT_DIMENSION = 1536


@pytest.fixture(params=[OpenAIEncoder, AzureOpenAIEncoder])
def openai_encoder(request):
    if request.param == OpenAIEncoder:
        return request.param(max_retries=3)
    else:
        model_name = os.environ.get("EMBEDDINGS_AZURE_OPENAI_DEPLOYMENT_NAME")
        return request.param(model_name=model_name, max_retries=3)


@pytest.mark.parametrize("encoder_class", [OpenAIEncoder, AzureOpenAIEncoder])
def test_init_with_kwargs(encoder_class):
    encoder = encoder_class(
        api_key="test_api_key",
        organization="test_organization",
        timeout=30,
        model_name="test_model_name",
    )
    assert encoder._client.api_key == "test_api_key"
    assert encoder._client.organization == "test_organization"
    assert encoder._client.timeout == 30

    with pytest.raises(AuthenticationError):
        encoder.encode_documents("test text")


def encode_by_type(openai_encoder, encoding_function, test_input):
    func = getattr(openai_encoder, encoding_function)
    return func(test_input)


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


def test_dimension(openai_encoder):
    assert openai_encoder.dimension == DEFAULT_DIMENSION


def test_openai_encoder_with_dimension():
    dimension = 10
    encoder = OpenAIEncoder(model_name="text-embedding-3-small", dimension=dimension)
    result = encoder.encode_documents("test text")

    assert encoder.dimension == dimension
    assert len(result) == dimension
