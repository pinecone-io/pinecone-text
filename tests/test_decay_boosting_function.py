from datetime import datetime, timedelta
from pinecone_text.sparse.decay_time_score_encoder import DecayTimeScoreEncoder


class TestDecayTimeScoreEncoder:
    def setup_method(self):
        self.encoder = DecayTimeScoreEncoder(half_life=30)

    def test_half_time(self):
        query = self.encoder.encode_query()

        doc_time = self.date_n_days_ago(30)
        document = self.encoder.encode_document(*doc_time)
        score = self.get_score(document, query)
        assert abs(score - 0.5) <= 2 ** -20

    def test_expected_scores_long_horizon(self):
        query = self.encoder.encode_query()

        self.assert_keys_and_values_range(query)

        for days_ago in range(1, 1000):
            doc_time = self.date_n_days_ago(days_ago)
            document = self.encoder.encode_document(*doc_time)
            self.assert_keys_and_values_range(document)
            self.assert_all_document_values_between_0_and_1(document)
            score = self.get_score(document, query)
            assert abs(2 ** (-days_ago / 30.0) - score) <= 2**-20

    @staticmethod
    def get_score(encoded_doc: dict, encoded_query: dict):
        score = 0.0
        for i, v in zip(encoded_query["indices"], encoded_query["values"]):
            if i in encoded_doc["indices"]:
                score += v * encoded_doc["values"][encoded_doc["indices"].index(i)]
        return score

    @staticmethod
    def date_n_days_ago(days_ago: int) -> tuple:
        new_date = datetime.now() - timedelta(days=days_ago)
        return new_date.day, new_date.month, new_date.year

    @staticmethod
    def assert_keys_and_values_range(encoded: dict):
        for value in encoded["values"]:
            assert -3.4028235e38 <= value <= 3.4028235e38
            assert type(value) == float

        for key in encoded["indices"]:
            assert 0 <= key <= 4294967295
            assert type(key) == int

    @staticmethod
    def assert_all_document_values_between_0_and_1(encoded: dict):
        for value in encoded["values"]:
            assert 0 <= value <= 1
            assert type(value) == float
