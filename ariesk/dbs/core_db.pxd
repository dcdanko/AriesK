import numpy as np
cimport numpy as npc

from ariesk.ram cimport RotatingRamifier


cdef class CoreDB:
    cdef public float box_side_len
    cdef public object conn
    cdef public RotatingRamifier ramifier
    cdef public dict centroid_cache
    cdef public list centroid_insert_buffer
    cdef public list kmer_insert_buffer
    cdef public int centroid_buffer_filled
    cdef public int kmer_buffer_filled

    cpdef _build_core_tables(self)
    cdef double[:, :] c_get_centroids(self)
    cdef save_ramifier(self)
    cdef RotatingRamifier load_ramifier(self)
    cpdef get_kmers(self)
    cdef npc.uint8_t[:, :] get_encoded_kmers(self)
    cdef add_point_to_cluster(
        self,
        double[:] centroid,
        npc.uint8_t[::] binary_kmer,
        unicode annotation=?
    )
    cdef _clear_buffer(self)
