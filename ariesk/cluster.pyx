
import numpy as np
cimport numpy as npc

from ariesk.bloom_filter cimport BloomFilter
from .utils cimport (
    encode_kmer,
    decode_kmer,
)


cdef class Cluster:

    def __cinit__(self, centroid_id, seqs, sub_k):
        self.centroid_id = centroid_id
        self.seqs = seqs
        self.k = seqs.shape[0]
        self.sub_k = sub_k

    cpdef build_bloom_grid(self, int filter_len, npc.uint64_t[:, :] col_hashes):
        cdef int n_row_hashes = 1
        cdef int grid_height = 4
        cdef npc.uint64_t[:, :] row_hashes = npc.ndarray((n_row_hashes, self.k), dtype=np.uint64)
        cdef int i, j
        for i in range(n_row_hashes):
            for j, val in enumerate(np.random.permutation(self.k)):
                row_hashes[i, j] = val
        self.bloom_grid = BloomGrid(
            self.sub_k, self.k, filter_len, grid_height, row_hashes, col_hashes
        )
        for i in range(self.seqs.shape[0]):
            self.bloom_grid.add(self.seqs[i, :])

    def py_test_membership(self, str seq, int allowed_misses):
        return self.test_membership(encode_kmer(seq), allowed_misses)

    def py_count_membership(self, str seq):
        return self.count_membership(encode_kmer(seq))

    cdef int count_membership(self, npc.uint8_t[:] query_seq):
        cdef int i
        cdef int n_hits = 0
        for i in range(self.seqs.shape[1] - self.sub_k + 1):
            if self.bloom_grid.array_contains(query_seq[i:i + self.sub_k]):
                n_hits += 1
        return n_hits

    cdef bint test_membership(self, npc.uint8_t[:] query_seq, int allowed_misses):
        cdef int n_hits = self.count_membership(query_seq)
        cdef int min_hits = self.seqs.shape[1] - self.sub_k + 1 - (self.sub_k * allowed_misses)
        return n_hits >= min_hits

    cdef int count_membership_hvals(self, npc.uint64_t[:, :] hash_vals):
        cdef int i
        cdef int n_hits = 0
        for i in range(hash_vals.shape[0]):
            if self.bloom_filter.array_contains_hvals(hash_vals[i, :]):
                n_hits += 1
        return n_hits

    cdef bint test_membership_hvals(self, npc.uint64_t[:, :] hash_vals, int allowed_misses):
        cdef int n_hits = self.count_membership_hvals(hash_vals)
        cdef int min_hits = self.seqs.shape[1] - self.sub_k + 1 - (self.sub_k * allowed_misses)
        return n_hits >= min_hits

    @classmethod
    def build_from_seqs(cls, centroid_id, seqs, sub_k, **kwargs):
        """Build from a list of python strings. Mostly for testing."""
        np_seqs = np.array([encode_kmer(seq) for seq in seqs])
        out = cls(centroid_id, np_seqs, sub_k, **kwargs)
        return out
