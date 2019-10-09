# cython: profile=True
# cython: linetrace=True
# cython: language_level=3

cimport numpy as npc

from ariesk.dbs.kmer_db cimport GridCoverDB
from ariesk.utils.kmers cimport (
    needle_dist,
    hamming_dist,
    encode_kmer,
    decode_kmer,
)


cdef class LinearSearcher:
    cdef public GridCoverDB db

    def __cinit__(self, grid_cover_db):
        self.db = grid_cover_db

    cpdef search(self, str query, metric='needle'):
        out = []
        cdef npc.uint8_t [:] encoded_query = encode_kmer(query)
        cdef int i
        cdef npc.uint8_t [:, :] encoded_kmers = self.db.get_encoded_kmers()
        for i in range(encoded_kmers.shape[0]):
            if metric == 'needle':
                dist = needle_dist(encoded_query, encoded_kmers[i, :], False)
            elif metric == 'hamming':
                dist = hamming_dist(encoded_query, encoded_kmers[i, :], False)
            out.append((decode_kmer(encoded_kmers[i, :]), dist))
        return out

    @classmethod
    def from_filepath(cls, filepath):
        return cls(GridCoverDB.load_from_filepath(filepath))
