
import numpy as np
cimport numpy as npc

from ariesk.bloom_filter cimport BloomFilter


cdef class Cluster:
    cdef public int centroid_id
    cdef public npc.uint8_t[:, :] seqs
    cdef public BloomFilter bloom_filter
    cdef public int sub_k

    cdef build_bloom_filter(self)
    cdef bint test_membership(self, npc.uint8_t[:] query_seq, int allowed_misses)
