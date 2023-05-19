from typing import List, Tuple
from pinecone_text.sparse import SparseVector


def hybrid_convex_scale(
    query_dense: List[float], query_sparse: SparseVector, alpha: float
) -> Tuple[List[float], SparseVector]:
    """Hybrid vector scaling using a convex combination
    Args:
        query_dense: a query dense vector represented as a list of floats
        query_sparse: a query sparse vector represented as a dict of indices and values
        alpha: float between 0 and 1 where 0 == sparse only and 1 == dense only

    Returns:
        a tuple of dense and sparse vectors scaled by alpha as a convex combination:
        ((dense * alpha), (sparse * (1 - alpha)))
    """

    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")

    scaled_sparse = {
        "indices": query_sparse["indices"],
        "values": [v * (1 - alpha) for v in query_sparse["values"]],
    }
    scaled_dense = [v * alpha for v in query_dense]
    return scaled_dense, scaled_sparse
