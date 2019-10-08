# cython: language_level=3

from ariesk.ram cimport RotatingRamifier
from ariesk.utils.bloom_filter cimport BloomGrid
from ariesk.cluster cimport Cluster
import numpy as np
cimport numpy as npc

cdef simple_list(sql_cursor)


cdef class GridCoverDB:
    cdef public float box_side_len
    cdef public object conn
    cdef public RotatingRamifier ramifier
    # A too simple dict based cache
    # for initial testing only as this will grow without bounds
    cdef public dict cluster_cache
    cdef public dict centroid_cache
    cdef public list centroid_insert_buffer
    cdef public list kmer_insert_buffer
    cdef public int centroid_buffer_filled
    cdef public int kmer_buffer_filled

    cpdef get_kmers(self)
    cdef _build_tables(self)
    cdef npc.uint8_t[:, :] get_encoded_kmers(self)
    cdef npc.uint8_t[:, :] get_cluster_members(self, int centroid_id)
    cdef Cluster get_cluster(self, int centroid_id, int filter_len, npc.uint64_t[:, :] hashes, int sub_k)
    cdef store_bloom_grid(self, Cluster cluster)
    cpdef build_and_store_bloom_grid(self, int centroid_id, int filter_len, npc.uint64_t[:, :] hashes, int sub_k)
    cpdef BloomGrid retrieve_bloom_grid(self, int centroid_id)
    cdef add_point_to_cluster(self, double[:] centroid, npc.uint8_t [::] kmer)
    cpdef load_other(self, GridCoverDB other)
    cdef double [:, :] c_get_centroids(self)
    cdef RotatingRamifier load_ramifier(self)
    cdef store_inner_clusters(self, Cluster cluster)
    cdef retrieve_inner_clusters(self, Cluster cluster)
    cdef save_ramifier(self)
    cdef _clear_buffer(self)


