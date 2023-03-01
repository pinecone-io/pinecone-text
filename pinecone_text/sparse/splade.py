from typing import List, Union

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pinecone_text.sparse import SparseVector


class Splade:

    """
    SPLADE sparse vector encoder.
    Currently only supports inference with  naver/splade-cocondenser-ensembledistil
    """

    def __init__(self, max_seq_length: int = 256, device: str = "cpu"):
        """
        Args:
            max_seq_length: Maximum sequence length for the model. Must be between 1 and 512.
            device: Device to use for inference.
        """
        if not 0 < max_seq_length <= 512:
            raise ValueError("max_seq_length must be between 1 and 512")

        model = "naver/splade-cocondenser-ensembledistil"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForMaskedLM.from_pretrained(model).to(device)
        self.max_seq_length = max_seq_length
        self.device = device

    def __call__(
        self, texts: Union[str, List[str]]
    ) -> Union[SparseVector, List[SparseVector]]:
        """
        Args:
            texts: single or list of texts to encode.

        Returns a list of Splade sparse vectors, one for each input text.
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        inter = torch.log1p(torch.relu(logits))
        token_max = torch.max(inter, dim=1)

        nz_tokens_i, nz_tokens_j = torch.where(token_max.values > 0)

        output = []
        for i in range(token_max.values.shape[0]):
            nz_tokens = nz_tokens_j[nz_tokens_i == i]
            nz_weights = token_max.values[i, nz_tokens]
            output.append(
                {"indices": nz_tokens.tolist(), "values": nz_weights.tolist()}
            )

        return output[0] if isinstance(texts, str) else output
