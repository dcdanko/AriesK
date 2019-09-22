
import numpy as np
cimport numpy as npc

from ariesk.bloom_filter cimport BloomFilter


cdef class Cluster:
    cdef public int centroid_id
    cdef public npc.uint8_t[:, :] seqs
    cdef public BloomFilter bloom_filter
    cdef public int sub_k

    cpdef build_bloom_filter(self, int filter_len, npc.uint64_t[:, :] hashes)
    cdef bint test_membership(self, npc.uint8_t[:] query_seq, int allowed_misses)
    cdef int count_membership(self, npc.uint8_t[:] query_seq)
    cdef int count_membership_hvals(self, npc.uint64_t[:, :] hash_vals)
    cdef bint test_membership_hvals(self, npc.uint64_t[:, :] hash_vals, int allowed_misses)
