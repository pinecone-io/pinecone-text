from pytest import raises
from pinecone_text.sparse import SPLADE


class TestSplade:
    def test_splade_single_doc_inference(self):
        splade = SPLADE()
        text = "This is a test"
        output = splade.encode_documents(text)
        assert len(output["indices"]) == len(output["values"])

    def test_splade_batch_inference(self):
        splade = SPLADE()
        texts = ["This is a test", "This is a test also"]
        output = splade.encode_documents(texts)
        assert len(output) == 2
        assert len(output[0]["indices"]) == len(output[0]["values"])
        assert len(output[1]["indices"]) == len(output[1]["values"])

        # calculate jacard similarity of output indices
        jacard = len(
            set(output[0]["indices"]).intersection(set(output[1]["indices"]))
        ) / len(set(output[0]["indices"]).union(set(output[1]["indices"])))
        assert 0.5 < jacard < 1.0

    def test_splade_single_query_inference(self):
        splade = SPLADE()
        text = "This is a test"
        output = splade.encode_queries(text)
        assert len(output["indices"]) == len(output["values"])

    def test_splade_batch_query_inference(self):
        splade = SPLADE()
        texts = ["This is a test", "This is a test also"]
        output = splade.encode_queries(texts)
        assert len(output) == 2
        assert len(output[0]["indices"]) == len(output[0]["values"])
        assert len(output[1]["indices"]) == len(output[1]["values"])

        # calculate jacard similarity of output indices
        jacard = len(
            set(output[0]["indices"]).intersection(set(output[1]["indices"]))
        ) / len(set(output[0]["indices"]).union(set(output[1]["indices"])))
        assert 0.5 < jacard < 1.0

    def test_splade_init_invalid_max_seq_length(self):
        with raises(ValueError):
            SPLADE(max_seq_length=0)

        with raises(ValueError):
            SPLADE(max_seq_length=513)
