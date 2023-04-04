import os
import numpy as np
from pytest import approx, raises
from pinecone_text.sparse import BM25Encoder


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
        self.bm25 = BM25Encoder()
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

        assert BM25Encoder._hash_text("the") not in self.bm25.doc_freq
        assert self.bm25.doc_freq[BM25Encoder._hash_text("quick")] == 6
        assert BM25Encoder._hash_text("notincorpus") not in self.bm25.doc_freq

    def test_encode_query(self):
        query = "The quick brown fox jumps over the lazy dog newword"
        encoded_query = self.bm25.encode_queries(query)

        assert len(encoded_query["indices"]) == len(encoded_query["values"])
        assert set(encoded_query["indices"]) == set(
            [BM25Encoder._hash_text(t) for t in self.tokenizer(query)]
        )

        fox_value = encoded_query["values"][
            encoded_query["indices"].index(BM25Encoder._hash_text("fox"))
        ]
        assert fox_value == approx(0.02944, abs=0.0001)

        newword_value = encoded_query["values"][
            encoded_query["indices"].index(BM25Encoder._hash_text("newword"))
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
            [BM25Encoder._hash_text(t) for t in self.tokenizer(queries[0])]
        )
        assert set(encoded_queries[1]["indices"]) == set(
            [BM25Encoder._hash_text(t) for t in self.tokenizer(queries[1])]
        )

        fox_value = encoded_queries[0]["values"][
            encoded_queries[0]["indices"].index(BM25Encoder._hash_text("fox"))
        ]
        assert fox_value == approx(0.029442, abs=0.0001)

        newword_value = encoded_queries[0]["values"][
            encoded_queries[0]["indices"].index(BM25Encoder._hash_text("newword"))
        ]

        assert newword_value == approx(0.343691, abs=0.0001)

    def test_encode_document(self):
        doc = "The quick brown fox jumps over the lazy dog newword"
        encoded_doc = self.bm25.encode_documents(doc)

        assert len(encoded_doc["indices"]) == len(encoded_doc["values"])
        assert set(encoded_doc["indices"]) == set(
            [BM25Encoder._hash_text(t) for t in self.tokenizer(doc)]
        )

        fox_value = encoded_doc["values"][
            encoded_doc["indices"].index(BM25Encoder._hash_text("fox"))
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
            [BM25Encoder._hash_text(t) for t in self.tokenizer(docs[0])]
        )
        assert set(encoded_docs[1]["indices"]) == set(
            [BM25Encoder._hash_text(t) for t in self.tokenizer(docs[1])]
        )

        fox_value = encoded_docs[0]["values"][
            encoded_docs[0]["indices"].index(BM25Encoder._hash_text("fox"))
        ]

        assert fox_value == approx(0.347826, abs=0.0001)

    def test_tf(self):
        doc = "The quick brown fox jumps jumps jumps over the lazy dog newword newword"
        indices, tf = self.bm25._tf(doc)

        assert len(indices) == len(tf)
        assert len(indices) == 7
        assert tf[indices.index(BM25Encoder._hash_text("fox"))] == 1
        assert tf[indices.index(BM25Encoder._hash_text("newword"))] == 2
        assert tf[indices.index(BM25Encoder._hash_text("jump"))] == 3
        assert BM25Encoder._hash_text("jumps") not in indices
        assert BM25Encoder._hash_text("the") not in indices

    def test_tf_zero_tokens(self):
        indices, tf = self.bm25._tf("")
        assert indices == [] and tf == []

        doc = "The over is #"
        indices, tf = self.bm25._tf(doc)
        assert indices == [] and tf == []

    def test_get_set_params_compatibility(self):
        bm25 = BM25Encoder()
        bm25.set_params(**self.bm25.get_params())
        assert bm25.get_params() == self.bm25.get_params()

    def test_store_load_params(self):
        self.bm25.dump(self.PARAMS_PATH)
        bm25 = BM25Encoder()
        bm25.load(self.PARAMS_PATH)
        assert bm25.get_params() == self.bm25.get_params()

    def test_encode_document_not_fitted(self):
        bm25 = BM25Encoder()
        doc = "The quick brown fox jumps over the lazy dog newword"

        with raises(ValueError):
            bm25.encode_documents(doc)

    def test_encode_query_not_fitted(self):
        bm25 = BM25Encoder()
        query = "The quick brown fox jumps over the lazy dog newword"

        with raises(ValueError):
            bm25.encode_queries(query)

    def test_get_params_not_fitted(self):
        bm25 = BM25Encoder()
        with raises(ValueError):
            bm25.get_params()

    def test_wrong_input_type(self):
        with raises(ValueError):
            self.bm25.encode_documents(1)

        with raises(ValueError):
            self.bm25.encode_queries(1)

    def test_fit_with_empty_documents(self):
        bm25 = BM25Encoder()
        bm25.fit(self.corpus + ["", "the I %"])
        assert bm25.get_params() == self.bm25.get_params()

    def test_fit_with_invalid_documents(self):
        bm25 = BM25Encoder()
        with raises(ValueError):
            bm25.fit(self.corpus + [1, 2])

    def test_create_default(self):
        bm25 = BM25Encoder.default()
        assert bm25.get_params()["n_docs"] == 8841823
        bm25.encode_documents("The quick brown fox jumps over the lazy dog newword")

    def test_compare_to_manual_calculation(self):
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
        expected_doc_freq = {
            BM25Encoder._hash_text("quick"): 6,
            BM25Encoder._hash_text("brown"): 8,
            BM25Encoder._hash_text("fox"): 8,
            BM25Encoder._hash_text("jump"): 3,
            BM25Encoder._hash_text("lazi"): 6,
            BM25Encoder._hash_text("dog"): 2,
        }
        assert self.bm25.doc_freq == expected_doc_freq
        assert self.bm25.avgdl == 4.0

        doc = "The quick brown brown fox jumps over the# lazy lazy lazy! dog newword newword"
        expected_tf = {
            BM25Encoder._hash_text("quick"): 1,
            BM25Encoder._hash_text("brown"): 2,
            BM25Encoder._hash_text("fox"): 1,
            BM25Encoder._hash_text("jump"): 1,
            BM25Encoder._hash_text("lazi"): 3,
            BM25Encoder._hash_text("dog"): 1,
            BM25Encoder._hash_text("newword"): 2,
        }
        expected_indices = list(expected_tf.keys())
        expected_values = list(expected_tf.values())
        assert self.bm25._tf(doc) == (expected_indices, expected_values)

        tf_sum = sum(expected_values)
        tf_normed = [
            v / (1.2 * (1.0 - 0.75 + 0.75 * (tf_sum / 4.0)) + v)
            for v in expected_values
        ]

        encoded_doc = self.bm25.encode_documents(doc)
        assert tf_normed == encoded_doc["values"]
        assert expected_indices == encoded_doc["indices"]

        query = "The quick yellow fox jumps jumps over the lazy lazy dog newword"
        expected_indices = [
            BM25Encoder._hash_text("quick"),
            BM25Encoder._hash_text("yellow"),
            BM25Encoder._hash_text("fox"),
            BM25Encoder._hash_text("jump"),
            BM25Encoder._hash_text("lazi"),
            BM25Encoder._hash_text("dog"),
            BM25Encoder._hash_text("newword"),
        ]
        encoded_query = self.bm25.encode_queries(query)
        expected_df = [expected_doc_freq.get(i, 1) for i in expected_indices]
        idf = [np.log((9.0 + 1.0) / (df + 0.5)) for df in expected_df]
        normed_idf = [v / sum(idf) for v in idf]

        assert encoded_query["values"] == approx(normed_idf, abs=0.0001)
        assert encoded_query["indices"] == expected_indices
