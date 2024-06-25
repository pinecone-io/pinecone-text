from typing import List, Union, Optional
from os import PathLike
import os
try:
    import torch
except (OSError, ImportError, ModuleNotFoundError) as e:
    _torch_installed = False
else:
    _torch_installed = True

try:
    from transformers import AutoTokenizer, AutoModelForMaskedLM
except (OSError, ImportError, ModuleNotFoundError) as e:
    _transformers_installed = False
else:
    _transformers_installed = True


from pinecone_text.sparse import SparseVector
from pinecone_text.sparse.base_sparse_encoder import BaseSparseEncoder


class SpladeEncoder(BaseSparseEncoder):

    """
    SPLADE sparse vector encoder.
    Currently only supports inference with  naver/splade-cocondenser-ensembledistil
    """

    def __init__(self, max_seq_length: int = 256, device: Optional[str] = None, model_dir:Optional[PathLike[str]] = None):
        """
        Args:
            max_seq_length: Maximum sequence length for the model. Must be between 1 and 512.
            device: Device to use for inference. Defaults to GPU if available, otherwise CPU.
            model_dir: Directory to download and load model from. Saves time and resources.

        Example:

            ```python
            from pinecone_text.sparse import SPLADE

            splade = SPLADE()

            splade.encode_documents("this is a document") # [{"indices": [102, 18, 12, ...], "values": [0.21, 0.38, 0.15, ...]}, ...]
            ```
        """
        if not _torch_installed:
            raise ImportError(
                """Failed to import torch. Make sure you install pytorch extra dependencies by running: `pip install pinecone-text[splade]`
        If this doesn't help, it is probably a CUDA error. If you do want to use GPU, please check your CUDA driver.
        If you want to use CPU only, run the following command:
        `pip uninstall -y torch torchvision;pip install -y torch torchvision --index-url https://download.pytorch.org/whl/cpu`"""
            )

        if not _transformers_installed:
            raise ImportError(
                "Failed to import transformers. Make sure you install splade "
                "extra dependencies by running: `pip install pinecone-text[splade]`"
            )

        if not 0 < max_seq_length <= 512:
            raise ValueError("max_seq_length must be between 1 and 512")

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        expected_model_name = "naver/splade-cocondenser-ensembledistil"
        if not self._is_correct_model(model_dir, expected_model_name):
            self.tokenizer,self.model=self._download_model(model_dir, expected_model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForMaskedLM.from_pretrained(model_dir).to(self.device)
        self.max_seq_length = max_seq_length
    def _is_correct_model(self, model_dir, expected_model_name):
        # Check for the presence of specific files that indicate the correct model
        config_path = os.path.join(model_dir, 'config.json')
        if not os.path.exists(config_path):
            return False

        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            return config.get("model_type") == expected_model_name

    def _download_model(self, model_dir, model_name):
        # Ensure the directory exists
        os.makedirs(model_dir, exist_ok=True)

        # Download the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_dir)

        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.save_pretrained(model_dir)
        return tokenizer,model
    def encode_documents(
        self, texts: Union[str, List[str]]
    ) -> Union[SparseVector, List[SparseVector]]:
        """
        encode documents to a sparse vector (for upsert to pinecone)

        Args:
            texts: a single or list of documents to encode as a string
        """
        return self._encode(texts)

    def encode_queries(
        self, texts: Union[str, List[str]]
    ) -> Union[SparseVector, List[SparseVector]]:
        """
        encode queries to a sparse vector (for upsert to pinecone)

        Args:
            texts: a single or list of queries to encode as a string
        """
        return self._encode(texts)

    def _encode(
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
