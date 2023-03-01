from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pinecone_text.sparse import SparseVector


class SPLADE:

    """SPLADE sparse vector encoder.
    Currently only supports inference with  naver/splade-cocondenser-ensembledistil
    """

    def __init__(self, max_seq_length: int = 256, device: str = "cpu"):
        model = "naver/splade-cocondenser-ensembledistil"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForMaskedLM.from_pretrained(model).to(device)
        self.max_seq_length = max_seq_length
        self.device = device

    def __call__(self, texts: List[str]) -> List[SparseVector]:
        """ "
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

        return output
