import torch
from typing import Optional, Union, List
from sentence_transformers import SentenceTransformer
from pinecone_text.dense.base_dense_ecoder import BaseDenseEncoder


class SentenceTransformerEncoder(BaseDenseEncoder):
    def __init__(
        self,
        document_encoder_name: str,
        query_encoder_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.document_encoder = SentenceTransformer(
            document_encoder_name, device=device
        )
        if query_encoder_name:
            self.query_encoder = SentenceTransformer(query_encoder_name, device=device)
        else:
            self.query_encoder = self.document_encoder

    def encode_documents(
        self, texts: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        return self.document_encoder.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        ).tolist()

    def encode_queries(
        self, texts: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        return self.query_encoder.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        ).tolist()
