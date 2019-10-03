# cython: language_level=3

import numpy as np
cimport numpy as npc

from ariesk.utils.bloom_filter cimport BloomGrid


cdef class Cluster:
    cdef public int centroid_id
    cdef public npc.uint8_t[:, :] seqs
    cdef public BloomGrid bloom_grid
    cdef public int sub_k, k

    cpdef build_bloom_grid(self, int filter_len, npc.uint64_t[:, :] hashes)
    cdef bint test_membership(self, npc.uint8_t[:] query_seq, int allowed_misses)
    cdef int count_membership(self, npc.uint8_t[:] query_seq)
    cdef int count_membership_hvals(self, npc.uint64_t[:, :] hash_vals)
    cdef bint test_membership_hvals(self, npc.uint64_t[:, :] hash_vals, int allowed_misses)
    cdef bint test_seq(self, int seq_id, npc.uint8_t[:] row_hits)
    cdef npc.uint8_t[:] test_row_membership(self, npc.uint64_t[:, :] hash_vals, int allowed_misses)