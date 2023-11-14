import pytest
from unittest.mock import patch
from pinecone_text.dense.sentence_transformer_encoder import SentenceTransformerEncoder
import pinecone_text.dense.sentence_transformer_encoder

DEFAULT_DIMENSION = 384


class TestSentenceTransformerEncoder:
    def setup_method(self):
        self.corpus = [
            "The quick brown fox jumps over the lazy dog",
            "The lazy dog is brown",
            "The fox is brown",
            "The fox is quick",
            "The fox is brown and quick",
            "The fox is brown and lazy",
            "The fox is brown and quick and lazy",
            "The fox is brown and quick and lazy and jumps",
            "The fox is brown and quick and lazy and jumps and over",
        ]
        self.encoder = SentenceTransformerEncoder(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def test_encode_documents(self):
        encoded_docs = self.encoder.encode_documents(self.corpus)
        assert len(encoded_docs) == len(self.corpus)
        assert len(encoded_docs[0]) == DEFAULT_DIMENSION

    def test_encode_single_document(self):
        encoded_doc = self.encoder.encode_documents(self.corpus[0])
        assert len(encoded_doc) == DEFAULT_DIMENSION

    def test_encode_single_document_list(self):
        encoded_docs = self.encoder.encode_documents([self.corpus[0]])
        assert len(encoded_docs) == 1
        assert len(encoded_docs[0]) == DEFAULT_DIMENSION

    def test_encode_queries(self):
        encoded_queries = self.encoder.encode_queries(self.corpus)
        assert len(encoded_queries[0]) == DEFAULT_DIMENSION

    def test_encode_single_query(self):
        encoded_query = self.encoder.encode_queries(self.corpus[0])
        assert len(encoded_query) == DEFAULT_DIMENSION

    def test_separate_doc_query_encoders(self):
        encoder = SentenceTransformerEncoder(
            document_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            query_encoder_name="sentence-transformers/msmarco-distilbert-base-tas-b",
        )
        doc_encoded = encoder.encode_documents(self.corpus[0])
        assert len(doc_encoded) == DEFAULT_DIMENSION

        query_encoded = encoder.encode_queries(self.corpus[0])
        assert len(query_encoded) == 768

    def test_encode_single_queries_list(self):
        encoded_queries = self.encoder.encode_queries([self.corpus[0]])
        assert len(encoded_queries) == 1
        assert len(encoded_queries[0]) == DEFAULT_DIMENSION

    @pytest.mark.parametrize("cuda_available, device_input, expected_device", [
        (True, None, "cuda"),
        (False, None, "cpu"),
        (True, "cpu", "cpu"),
        (False, "cuda", "cuda")
    ])
    def test_init_cuda_available(self, cuda_available, device_input, expected_device):
        with patch('torch.cuda.is_available', return_value=cuda_available):
            encoder = SentenceTransformerEncoder(
                document_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
                device=device_input
            )
            assert str(encoder.document_encoder._target_device) == expected_device
            assert str(encoder.query_encoder._target_device) == expected_device
