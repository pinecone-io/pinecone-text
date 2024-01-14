import pytest

from pinecone_text.dense import JinaEncoder

DEFAULT_DIMENSION = 768


@pytest.mark.parametrize(
    "encoding_function",
    [
        ("encode_documents"),
        ("encode_queries"),
    ],
)
def test_encode_single_text(encoding_function):
    encoder = JinaEncoder()
    result = getattr(encoder, encoding_function)("test text")
    assert isinstance(result, list)
    assert len(result) == DEFAULT_DIMENSION


@pytest.mark.parametrize(
    "encoding_function",
    [
        ("encode_documents"),
        ("encode_queries"),
    ],
)
def test_encode_multiple_text(encoding_function):
    encoder = JinaEncoder()
    result = getattr(encoder, encoding_function)(["test text", "text 2"])
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
def test_encode_invalid_input(encoding_function):
    encoder = JinaEncoder()

    with pytest.raises(ValueError):
        _ = getattr(encoder, encoding_function)(123)


@pytest.mark.parametrize(
    "encoding_function",
    [
        ("encode_documents"),
        ("encode_queries"),
    ],
)
def test_encode_too_many_tokens(encoding_function):
    encoder = JinaEncoder()
    with pytest.raises(RuntimeError, match=".+?cannot exceed 8192 tokens.+?"):
        _ = getattr(encoder, encoding_function)("text is too long" * 10000)


@pytest.mark.parametrize(
    "encoding_function",
    [
        ("encode_documents"),
        ("encode_queries"),
    ],
)
def test_encode_too_many_sentences(encoding_function):
    encoder = JinaEncoder()
    with pytest.raises(RuntimeError, match=".+?larger than the largest allowed.+?"):
        _ = getattr(encoder, encoding_function)(["too many inputs"] * 3000)
