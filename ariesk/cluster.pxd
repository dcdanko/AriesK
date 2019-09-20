
import numpy as np
cimport numpy as npc

from ariesk.bloom_filter cimport BloomFilter


cdef class Cluster:
    cdef int centroid_id
    cdef npc.uint8_t[:, :] seqs
    cdef BloomFilter bloom_filter
    cdef int sub_k

    cdef build_bloom_filter(self)
    cdef bint test_membership(self, npc.uint8_t[:] query_seq, int allowed_misses)
