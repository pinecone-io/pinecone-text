import pytest

import nltk
import shutil
import importlib
import sys
from pinecone_text.sparse.bm25_tokenizer import BM25Tokenizer


class TestBM25Tokenizer:
    def test_bm25_tokenizer_default_params(self):
        tokenizer = BM25Tokenizer(
            lower_case=True,
            stem=True,
            remove_punctuation=True,
            remove_stopwords=True,
            language="english",
        )
        assert tokenizer("The quick brown fox jumps over the lazy dog") == [
            "quick",
            "brown",
            "fox",
            "jump",
            "lazi",
            "dog",
        ]
        assert tokenizer("The yellow-FOx is# (brown} a! @%^ $ WALKING!!") == [
            "yellow-fox",
            "brown",
            "walk",
        ]

        non_english_tokens = [
            "čeština",
            "françai",
            "ελληνικά",
            "portuguê",
            "slovenščina",
            "español",
            "türkçe",
            "русский",
            "עברית",
            "العربية",
            "日本語",
            "한국어",
            "中文",
            "हिन्दी",
            "ไทย",
            "বাংলা",
            "தமிழ்",
            "తెలుగు",
        ]

        assert tokenizer(" ".join(non_english_tokens)) == non_english_tokens

    def test_bm25_tokenizer_custom_params(self):
        tokenizer = BM25Tokenizer(
            lower_case=False,
            stem=False,
            remove_punctuation=True,
            remove_stopwords=True,
            language="english",
        )
        assert tokenizer("The quick brown FOX jumps over the lazy dogs") == [
            "quick",
            "brown",
            "FOX",
            "jumps",
            "lazy",
            "dogs",
        ]

        tokenizer = BM25Tokenizer(
            lower_case=True,
            stem=True,
            remove_punctuation=False,
            remove_stopwords=False,
            language="english",
        )

        assert tokenizer("The FOx is# browns a! ## and I") == [
            "the",
            "fox",
            "is",
            "#",
            "brown",
            "a",
            "!",
            "#",
            "#",
            "and",
            "i",
        ]

        tokenizer = BM25Tokenizer(
            lower_case=True,
            stem=True,
            remove_punctuation=True,
            remove_stopwords=True,
            language="french",
        )

        assert tokenizer("Le renard brun (saute) par-dessus le chien paresseux!") == [
            "renard",
            "brun",
            "saut",
            "par-dessus",
            "chien",
            "paress",
        ]

        tokenizer = BM25Tokenizer(
            lower_case=False,
            stem=False,
            remove_punctuation=True,
            remove_stopwords=True,
            language="english",
        )

        assert tokenizer("The Stop words I") == ["Stop", "words"]

    def test_bm25_invalid_params(self):
        with pytest.raises(ValueError):
            BM25Tokenizer(
                lower_case=True,
                stem=True,
                remove_punctuation=True,
                remove_stopwords=True,
                language="invalid",
            )

        with pytest.raises(ValueError):
            BM25Tokenizer(
                lower_case=False,
                stem=True,
                remove_punctuation=True,
                remove_stopwords=True,
                language="english",
            )

    def test_nltk_download(self):
        shutil.rmtree(nltk.find("tokenizers"))
        shutil.rmtree(nltk.find("corpora"))

        importlib.reload(sys.modules["pinecone_text.sparse.bm25_tokenizer"])

        tokenizer = BM25Tokenizer(
            lower_case=True,
            stem=True,
            remove_punctuation=True,
            remove_stopwords=True,
            language="english",
        )

        nltk.find("tokenizers/punkt_tab")
        nltk.find("corpora/stopwords")

        assert tokenizer("The quick brown fox jumps over the lazy dog") == [
            "quick",
            "brown",
            "fox",
            "jump",
            "lazi",
            "dog",
        ]
