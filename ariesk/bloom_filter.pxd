
import numpy as np
cimport numpy as npc


cdef npc.uint64_t fnva(npc.uint8_t[:] data, npc.uint64_t[:] access_order)


cdef class BloomFilter:
    cdef public npc.uint64_t[:, :] hashes
    cdef public npc.uint8_t[:] bitarray
    cdef public double p
    cdef public int n_hashes, len_filter, len_seq, n_elements

    cdef add(self, npc.uint8_t[:] seq)
    cdef bint contains(self, npc.uint8_t[:] seq)
    cpdef int union(self, BloomFilter other)
    cpdef int intersection(self, BloomFilter other)
