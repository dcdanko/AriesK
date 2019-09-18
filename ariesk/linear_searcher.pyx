
from .db cimport GridCoverDB
from .utils cimport (
    needle_dist,
    hamming_dist,
)


cdef class LinearSearcher:
    cdef public GridCoverDB db

    def __cinit__(self, grid_cover_db):
        self.db = grid_cover_db

    cpdef search(self, str query, metric='needle'):
        out = []
        for _, target in self.db.get_kmers():
            if metric == 'needle':
                dist = needle_dist(query, target, False)
            elif metric == 'hamming':
                dist = hamming_dist(query, target, False)
            out.append((target, dist))
        return out

    @classmethod
    def from_filepath(cls, filepath):
        return cls(GridCoverDB.load_from_filepath(filepath))
