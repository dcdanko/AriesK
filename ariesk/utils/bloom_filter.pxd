# cython: language_level=3

import numpy as np
cimport numpy as npc

cdef npc.uint32_t fast_modulo(npc.uint32_t val, npc.uint64_t N, npc.uint32_t shift)
cdef npc.uint32_t fnva(npc.uint8_t[:] data, npc.uint64_t[:] access_order)


cdef class BloomFilter:
    cdef public npc.uint64_t[:, :] hashes
    cdef public npc.uint8_t[:] bitarray
    cdef public double p
    cdef public int n_hashes, filter_power, len_seq, n_elements
    cdef public npc.uint64_t len_filter

    cdef add(self, npc.uint8_t[:] seq)
    cdef bint contains(self, npc.uint8_t[:] seq)
    cdef bint contains_hvals(self, npc.uint64_t[:] hvals)
    cpdef int union(self, BloomFilter other)
    cpdef int intersection(self, BloomFilter other)


cdef class BloomGrid:
    cdef public npc.uint8_t[:] bitarray
    cdef public npc.uint8_t[:, :] bitgrid
    cdef public int grid_width_power, grid_height, col_k, row_k
    cdef public npc.uint64_t grid_width
    cdef public npc.uint64_t[:, :] col_hashes, row_hashes

    cdef add(self, npc.uint8_t[:] seq)
    cdef npc.uint64_t[:] _get_hashes(self, npc.uint8_t[:] seq)
    cdef bint array_contains(self, npc.uint8_t[:] seq)
    cdef bint array_contains_hvals(self, npc.uint64_t[:] hvals)
    cdef npc.uint8_t[:] grid_contains(self, npc.uint8_t[:] seq)
    cdef npc.uint8_t[:] grid_contains_hvals(self, npc.uint64_t[:] hvals)
    cdef npc.uint8_t[:] count_grid_contains(self, npc.uint8_t[:] seq)
    cdef  npc.uint8_t[:] count_grid_contains_hvals(self, npc.uint64_t[:, :] hvals)
