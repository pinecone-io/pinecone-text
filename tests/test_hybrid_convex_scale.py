import pytest
from pytest import approx
from pinecone_text.hybrid import hybrid_convex_scale
from pinecone_text.sparse.splade_encoder import SpladeEncoder
from pinecone_text.dense.sentence_transformer_encoder import SentenceTransformerEncoder


class TestHybridConvexScale:
    def test_hybrid_convex_scale_dummy_data(self):
        dense = [1, 2, -3]
        sparse = {"indices": [0, 1, 2], "values": [1, 2, -3]}
        alpha = 0.5
        scaled_dense, scaled_sparse = hybrid_convex_scale(dense, sparse, alpha)
        assert scaled_dense == approx([0.5, 1.0, -1.5])
        assert scaled_sparse == {"indices": [0, 1, 2], "values": [0.5, 1.0, -1.5]}

        alpha = 0.3
        scaled_dense, scaled_sparse = hybrid_convex_scale(dense, sparse, alpha)
        assert scaled_dense == approx([0.3, 0.6, -0.9])
        assert scaled_sparse == {
            "indices": [0, 1, 2],
            "values": approx([0.7, 1.4, -2.1]),
        }

    def test_hybrid_convex_scale_extreme_alpha(self):
        dense = [1, 2, 3]
        sparse = {"indices": [0, 1, 2], "values": [1, 2, 3]}
        alpha = 0
        scaled_dense, scaled_sparse = hybrid_convex_scale(dense, sparse, alpha)
        assert scaled_dense == [0, 0, 0]
        assert scaled_sparse == {"indices": [0, 1, 2], "values": [1, 2, 3]}

        alpha = 1
        scaled_dense, scaled_sparse = hybrid_convex_scale(dense, sparse, alpha)
        assert scaled_dense == [1, 2, 3]
        assert scaled_sparse == {"indices": [0, 1, 2], "values": [0, 0, 0]}

    def test_hybrid_convex_scale_integration_with_encoders(self):
        query = "I like pinecone"
        dense = SentenceTransformerEncoder(
            "sentence-transformers/all-MiniLM-L6-v2"
        ).encode_queries(query)
        sparse = SpladeEncoder().encode_queries(query)

        alpha = 0.8
        scaled_dense, scaled_sparse = hybrid_convex_scale(dense, sparse, alpha)
        assert scaled_dense == approx([0.8 * v for v in dense])
        assert scaled_sparse == {
            "indices": sparse["indices"],
            "values": approx([0.2 * v for v in sparse["values"]]),
        }

    def test_hybrid_convex_scale_invalid_alpha(self):
        dense = [1, 2, 3]
        sparse = {"indices": [0, 1, 2], "values": [1, 2, 3]}
        alpha = -1
        with pytest.raises(ValueError):
            hybrid_convex_scale(dense, sparse, alpha)
        alpha = 2
        with pytest.raises(ValueError):
            hybrid_convex_scale(dense, sparse, alpha)
