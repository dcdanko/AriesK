
from .ram cimport RotatingRamifier
import numpy as np
cimport numpy as npc

cdef simple_list(sql_cursor)


cdef class GridCoverDB:
    cdef public float box_side_len
    cdef public object conn
    cdef public object cursor
    cdef public object centroid_cache
    cdef public RotatingRamifier ramifier
    # A too simple dict based cache
    # for initial testing only as this will grow without bounds
    cdef public dict cluster_cache
    cdef public dict bloom_cache

    cpdef get_kmers(self)
    cdef npc.uint8_t [:, :] get_encoded_kmers(self)
    cdef npc.uint8_t [:, :] get_cluster_members(self, int centroid_id)
    cdef get_bloom_filter(self, int centroid_id)
    cdef add_point_to_cluster(self, double [:] centroid, npc.uint8_t [:] kmer)
    cpdef load_other(self, GridCoverDB other)
    cdef double [:, :] c_get_centroids(self)
    cdef RotatingRamifier load_ramifier(self)
    cdef save_ramifier(self)


