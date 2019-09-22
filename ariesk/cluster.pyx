
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
        self.sub_k = sub_k

    cpdef build_bloom_filter(self, int filter_len, npc.uint64_t[:, :] hashes):
        self.bloom_filter = BloomFilter(self.sub_k, filter_len, hashes)
        cdef int i, j
        for i in range(self.seqs.shape[0]):
            for j in range(self.seqs.shape[1] - self.sub_k + 1):
                self.bloom_filter.add(self.seqs[i, j:j + self.sub_k])

    def py_test_membership(self, str seq, int allowed_misses):
        return self.test_membership(encode_kmer(seq), allowed_misses)

    def py_count_membership(self, str seq):
        return self.count_membership(encode_kmer(seq))

    cdef int count_membership(self, npc.uint8_t[:] query_seq):
        cdef int i
        cdef int n_hits = 0
        for i in range(self.seqs.shape[1] - self.sub_k + 1):
            if self.bloom_filter.contains(query_seq[i:i + self.sub_k]):
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
            if self.bloom_filter.contains_hvals(hash_vals[i, :]):
                n_hits += 1
        return n_hits

    cdef bint test_membership_hvals(self, npc.uint64_t[:, :] hash_vals, int allowed_misses):
        cdef int n_hits = self.count_membership_hvals(hash_vals)
        cdef int min_hits = self.seqs.shape[1] - self.sub_k + 1 - (self.sub_k * allowed_misses)
        return n_hits >= min_hits

    @classmethod
    def build_from_seqs(cls, centroid_id, seqs, **kwargs):
        """Build from a list of python strings. Mostly for testing."""
        np_seqs = np.array([encode_kmer(seq) for seq in seqs])
        out = cls(centroid_id, np_seqs, **kwargs)
        return out
