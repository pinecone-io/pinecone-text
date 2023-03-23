from pinecone_text.sparse.bm25 import BM25Tokenizer


class TestBM25Tokenizer:
    def test_bm25_tokenizer_default_params(self):
        tokenizer = BM25Tokenizer()
        assert tokenizer("The quick brown fox jumps over the lazy dog") == [
            "quick",
            "brown",
            "fox",
            "jump",
            "lazi",
            "dog",
        ]
        assert tokenizer("The FO)x i$s# br!own a! walking") == ["fox", "brown", "walk"]

    def test_bm25_tokenizer_custom_params(self):
        tokenizer = BM25Tokenizer(lower_case=False, stem=False)
        assert tokenizer("The quick brown FOX j#umps ov!er the lazy dogs") == [
            "quick",
            "brown",
            "FOX",
            "jumps",
            "lazy",
            "dogs",
        ]

        tokenizer = BM25Tokenizer(remove_punctuation=False, remove_single_chars=False)

        assert tokenizer("The FOx i$s# br!owns a! a b") == [
            "fox",
            "i$s#",
            "br!own",
            "a!",
            "b",
        ]

    def test_custom_tokenizer(self):
        tokenizer = BM25Tokenizer(tokenizer=lambda x: ["the", "King", "queen#"])
        assert tokenizer("The quick brown fox jumps over the lazy dog") == [
            "king",
            "queen",
        ]
