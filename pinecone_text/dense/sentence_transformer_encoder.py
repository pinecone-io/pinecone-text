"""
Sentance Transformers are a class of models that take a sentence as input and output a vector representation of the sentence.
These models are useful for tasks such as semantic search, clustering, and classification. The sentence transformer models are
the work of the research team led by Nils Reimers at the University of Stuttgart. For more information, see the [Sentence Transformers paper](https://arxiv.org/abs/1908.10084).

"""

try:
    import torch
except (OSError, ImportError, ModuleNotFoundError) as e:
    _torch_installed = False
else:
    _torch_installed = True

from typing import Optional, Union, List

try:
    from sentence_transformers import SentenceTransformer
except (ImportError, ModuleNotFoundError) as e:
    _transformers_installed = False
else:
    _transformers_installed = True


from pinecone_text.dense.base_dense_ecoder import BaseDenseEncoder


class SentenceTransformerEncoder(BaseDenseEncoder):
    def __init__(
        self,
        document_encoder_name: str,
        query_encoder_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if not _torch_installed:
            raise ImportError(
                """Failed to import torch. Make sure you install dense extra 
                dependencies by running: `pip install pinecone-text[dense]`
        If this doesn't help, it is probably a CUDA error. If you do want to use GPU, 
        please check your CUDA driver.
        If you want to use CPU only, run the following command:
        `pip uninstall -y torch torchvision;pip install -y torch torchvision 
        --index-url https://download.pytorch.org/whl/cpu`"""
            )

        if not _transformers_installed:
            raise ImportError(
                "Failed to import sentence transformers. Make sure you install dense "
                "extra dependencies by running: `pip install pinecone-text[dense]`"
            )
        super().__init__()
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
