from unittest.mock import patch

import pytest
import sys
from pytest import raises
from pinecone_text.sparse import SpladeEncoder


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="not supported in python 3.12")
class TestSplade:
    def test_splade_single_doc_inference(self):
        splade = SpladeEncoder()
        text = "This is a test"
        output = splade.encode_documents(text)
        assert len(output["indices"]) == len(output["values"])

    def test_splade_batch_inference(self):
        splade = SpladeEncoder()
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
        splade = SpladeEncoder()
        text = "This is a test"
        output = splade.encode_queries(text)
        assert len(output["indices"]) == len(output["values"])

    def test_splade_batch_query_inference(self):
        splade = SpladeEncoder()
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
            SpladeEncoder(max_seq_length=0)

        with raises(ValueError):
            SpladeEncoder(max_seq_length=513)

    @pytest.mark.parametrize(
        "cuda_available, device_input, expected_device",
        [
            (True, None, "cuda"),
            (False, None, "cpu"),
            (True, "cpu", "cpu"),
            (False, "cuda", "cuda"),
        ],
    )
    def test_init_cuda_available(self, cuda_available, device_input, expected_device):
        import torch

        system_cuda_available = torch.cuda.is_available()

        with patch("torch.cuda.is_available", return_value=cuda_available):
            if not system_cuda_available and expected_device == "cuda":
                with raises(
                    AssertionError, match="Torch not compiled with CUDA enabled"
                ):
                    SpladeEncoder(device=device_input)
            else:
                encoder = SpladeEncoder(device=device_input)
                assert encoder.device == expected_device
                assert encoder.device == expected_device
