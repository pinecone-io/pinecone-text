import mmh3
from datetime import datetime
from pinecone_text.sparse import SparseVector


class DecayTimeScoreEncoder:
    MIN_POWER = 20

    def __init__(self, half_life: int):
        assert (
                type(half_life) == int and half_life > 0
        ), "Half time must be an integer > 0"

        self.half_life = half_life
        self.bucket_size = half_life * DecayTimeScoreEncoder.MIN_POWER

    def encode_document(self, day: int, month: int, year: int) -> SparseVector:
        # assert document not from the past
        assert (
            datetime(year, month, day) <= datetime.now()
        ), "Document must not be from the future"

        bucket, bucket_offset = self._get_bucket_and_offset(day, month, year)
        value = 2 ** ((bucket_offset - self.bucket_size) / float(self.half_life))
        key = self._get_key(bucket)
        return {"indices": [key], "values": [value]}

    def encode_query(self) -> SparseVector:
        now = datetime.now()
        bucket, bucket_offset = self._get_bucket_and_offset(
            now.day, now.month, now.year
        )

        sparse_values = {"indices": [], "values": []}
        for lookback in range(
            int(DecayTimeScoreEncoder.MIN_POWER / (self.bucket_size // self.half_life))
            + 1
        ):
            offset = lookback * self.bucket_size + bucket_offset
            key = self._get_key(bucket - lookback)
            value = 2 ** ((self.bucket_size - offset) / float(self.half_life))
            sparse_values["indices"].append(key)
            sparse_values["values"].append(value)

        return sparse_values

    @staticmethod
    def _get_key(bucket: int):
        return mmh3.hash(str(bucket) + "__random_text", signed=False)

    def _get_bucket_and_offset(self, day, month, year):
        dt = datetime(year, month, day)
        epoch = datetime(1970, 1, 1)
        days_since_epoch = (dt - epoch).days
        return days_since_epoch // self.bucket_size, days_since_epoch % self.bucket_size
