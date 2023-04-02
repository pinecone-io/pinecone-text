import os
import numpy as np
from pytest import approx, raises
from pinecone_text.sparse import BM25


class TestBM25:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    PARAMS_PATH = os.path.join(cur_dir, "bm25_params.json")

    def setup_method(self):
        self.corpus = [
            "The quick brown fox jumps over the lazy dog",
            "The lazy dog is brown",
            "The fox is brown",
            "The fox is quick",
            "The fox is brown and quick and brown",
            "The fox is brown and lazy and lazy and lazy",
            "The fox is brown and quick and lazy",
            "The fox is brown and quick and lazy and jumps",
            "The fox is brown and quick and lazy and jumps",
        ]
        self.bm25 = BM25()
        self.bm25.fit(self.corpus)
        self.tokenizer = self.bm25._tokenizer

    def teardown_method(self):
        if os.path.exists(self.PARAMS_PATH):
            os.remove(self.PARAMS_PATH)

    def test_fit_default_params(self):
        assert self.bm25.n_docs == len(self.corpus)
        expected_avgdl = np.mean(
            [len([w for w in self.tokenizer(doc)]) for doc in self.corpus]
        )
        assert self.bm25.avgdl == expected_avgdl

        assert BM25._hash_text("the") not in self.bm25.doc_freq
        assert self.bm25.doc_freq[BM25._hash_text("quick")] == 6
        assert BM25._hash_text("notincorpus") not in self.bm25.doc_freq

    def test_encode_query(self):
        query = "The quick brown fox jumps over the lazy dog newword"
        encoded_query = self.bm25.encode_queries(query)

        assert len(encoded_query["indices"]) == len(encoded_query["values"])
        assert set(encoded_query["indices"]) == set(
            [BM25._hash_text(t) for t in self.tokenizer(query)]
        )

        fox_value = encoded_query["values"][
            encoded_query["indices"].index(BM25._hash_text("fox"))
        ]
        assert fox_value == approx(0.02944, abs=0.0001)

        newword_value = encoded_query["values"][
            encoded_query["indices"].index(BM25._hash_text("newword"))
        ]

        assert newword_value == approx(0.34369, abs=0.0001)

    def test_encode_queries(self):
        queries = [
            "The quick brown fox jumps over the lazy dog newword",
            "The quick brown fox jumps over the lazy dog newword",
        ]
        encoded_queries = self.bm25.encode_queries(queries)

        assert len(encoded_queries) == len(queries)
        assert len(encoded_queries[0]["indices"]) == len(encoded_queries[0]["values"])
        assert len(encoded_queries[1]["indices"]) == len(encoded_queries[1]["values"])
        assert set(encoded_queries[0]["indices"]) == set(
            [BM25._hash_text(t) for t in self.tokenizer(queries[0])]
        )
        assert set(encoded_queries[1]["indices"]) == set(
            [BM25._hash_text(t) for t in self.tokenizer(queries[1])]
        )

        fox_value = encoded_queries[0]["values"][
            encoded_queries[0]["indices"].index(BM25._hash_text("fox"))
        ]
        assert fox_value == approx(0.029442, abs=0.0001)

        newword_value = encoded_queries[0]["values"][
            encoded_queries[0]["indices"].index(BM25._hash_text("newword"))
        ]

        assert newword_value == approx(0.343691, abs=0.0001)

    def test_encode_document(self):
        doc = "The quick brown fox jumps over the lazy dog newword"
        encoded_doc = self.bm25.encode_documents(doc)

        assert len(encoded_doc["indices"]) == len(encoded_doc["values"])
        assert set(encoded_doc["indices"]) == set(
            [BM25._hash_text(t) for t in self.tokenizer(doc)]
        )

        fox_value = encoded_doc["values"][
            encoded_doc["indices"].index(BM25._hash_text("fox"))
        ]

        assert fox_value == approx(0.34782, abs=0.0001)

    def test_encode_documents(self):
        docs = [
            "The quick brown fox jumps over the lazy dog newword",
            "The quick brown fox jumps over the lazy dog newword",
        ]
        encoded_docs = self.bm25.encode_documents(docs)

        assert len(encoded_docs) == len(docs)
        assert len(encoded_docs[0]["indices"]) == len(encoded_docs[0]["values"])
        assert len(encoded_docs[1]["indices"]) == len(encoded_docs[1]["values"])
        assert set(encoded_docs[0]["indices"]) == set(
            [BM25._hash_text(t) for t in self.tokenizer(docs[0])]
        )
        assert set(encoded_docs[1]["indices"]) == set(
            [BM25._hash_text(t) for t in self.tokenizer(docs[1])]
        )

        fox_value = encoded_docs[0]["values"][
            encoded_docs[0]["indices"].index(BM25._hash_text("fox"))
        ]

        assert fox_value == approx(0.347826, abs=0.0001)

    def test_get_set_params_compatibility(self):
        bm25 = BM25()
        bm25.set_params(**self.bm25.get_params())
        assert bm25.get_params() == self.bm25.get_params()

    def test_store_load_params(self):
        self.bm25.dump(self.PARAMS_PATH)
        bm25 = BM25()
        bm25.load(self.PARAMS_PATH)
        assert bm25.get_params() == self.bm25.get_params()

    def test_encode_document_not_fitted(self):
        bm25 = BM25()
        doc = "The quick brown fox jumps over the lazy dog newword"

        with raises(ValueError):
            bm25.encode_documents(doc)

    def test_encode_query_not_fitted(self):
        bm25 = BM25()
        query = "The quick brown fox jumps over the lazy dog newword"

        with raises(ValueError):
            bm25.encode_queries(query)

    def test_get_params_not_fitted(self):
        bm25 = BM25()
        with raises(ValueError):
            bm25.get_params()

    def test_wrong_input_type(self):
        with raises(ValueError):
            self.bm25.encode_documents(1)

        with raises(ValueError):
            self.bm25.encode_queries(1)

    def test_create_default(self):
        bm25 = BM25.default()
        assert bm25.get_params()["n_docs"] == 8841823
        bm25.encode_documents("The quick brown fox jumps over the lazy dog newword")
