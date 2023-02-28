import numpy as np
from pinecone_text.src.sparse.bm25 import BM25


class TestBM25:

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
        self.bm25 = BM25(tokenizer=lambda x: x.split())
        self.bm25.fit(self.corpus)

    def get_token_hash(self, token, bm25: BM25):
        return bm25._vectorizer.transform([token]).indices[0].item()

    def test_fit_default_params(self):
        assert self.bm25.n_docs == len(self.corpus)
        expected_avgdl = np.mean([len(set([w.lower() for w in doc.split()])) for doc in self.corpus])
        assert self.bm25.avgdl == expected_avgdl

        assert self.bm25.doc_freq[self.get_token_hash('the', self.bm25)] == 9
        assert self.bm25.doc_freq[self.get_token_hash('quick', self.bm25)] == 6
        assert self.get_token_hash('notincorpus', self.bm25) not in self.bm25.doc_freq

    def test_encode_query(self):
        query = "The quick brown fox jumps over the lazy dog newword"
        encoded_query = self.bm25.encode_query(query)

        assert len(encoded_query['indices']) == len(encoded_query['values'])
        assert set(encoded_query['indices']) == set([self.get_token_hash(t, self.bm25) for t in query.split()])

        fox_value = encoded_query["values"][encoded_query['indices'].index(self.get_token_hash("fox", self.bm25))]
        assert abs(fox_value - 0.020173) < 0.0001

        newword_value = encoded_query["values"][encoded_query['indices'].index(
            self.get_token_hash("newword", self.bm25))]
        assert abs(newword_value - 0.371861) < 0.0001

    def test_encode_document(self):
        doc = "The quick brown fox jumps over the lazy dog newword"
        encoded_doc = self.bm25.encode_document(doc)

        assert len(encoded_doc['indices']) == len(encoded_doc['values'])
        assert set(encoded_doc['indices']) == set([self.get_token_hash(t, self.bm25) for t in doc.split()])

        fox_value = encoded_doc["values"][encoded_doc['indices'].index(self.get_token_hash("fox", self.bm25))]
        assert abs(fox_value - 0.32203) < 0.0001
