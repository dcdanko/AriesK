# cython: language_level=3

from ariesk.ram cimport RotatingRamifier
from ariesk.utils.bloom_filter cimport BloomGrid
from ariesk.cluster cimport Cluster
import numpy as np
cimport numpy as npc

cdef simple_list(sql_cursor)


cdef class PreDB:

    cdef public object conn
    cdef public RotatingRamifier ramifier
    cdef public list kmer_insert_buffer
    cdef public int kmer_buffer_filled

    cdef _build_tables(self)
    cdef c_add_kmer(self, npc.uint8_t [:] binary_kmer)
    cdef add_point(self, double[:] rft, npc.uint8_t [::] binary_kmer)
    cdef _clear_buffer(self)
    cdef save_ramifier(self)
    cdef RotatingRamifier load_ramifier(self)
