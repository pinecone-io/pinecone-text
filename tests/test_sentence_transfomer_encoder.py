from pinecone_text.dense.sentence_transformer_encoder import SentenceTransformerEncoder


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
        assert len(encoded_docs[0]) == 384

    def test_encode_single_document(self):
        endoced_doc = self.encoder.encode_documents(self.corpus[0])
        assert len(endoced_doc) == 384

    def test_encode_queries(self):
        encoded_queries = self.encoder.encode_queries(self.corpus)
        encoded_docs = self.encoder.encode_documents(self.corpus)
        assert len(encoded_queries[0]) == 384

    def test_encode_single_query(self):
        encoded_query = self.encoder.encode_queries(self.corpus[0])
        assert len(encoded_query) == 384

    def test_separate_doc_query_encoders(self):
        encoder = SentenceTransformerEncoder(
            document_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            query_encoder_name="sentence-transformers/msmarco-distilbert-base-tas-b",
        )
        doc_encoded = encoder.encode_documents(self.corpus[0])
        assert len(doc_encoded) == 384

        query_encoded = encoder.encode_queries(self.corpus[0])
        assert len(query_encoded) == 768
