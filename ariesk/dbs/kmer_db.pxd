# cython: language_level=3

from ariesk.utils.bloom_filter cimport BloomGrid
from ariesk.cluster cimport Cluster
from ariesk.dbs.core_db cimport CoreDB
import numpy as np
cimport numpy as npc


cdef class GridCoverDB(CoreDB):
    # A too simple dict based cache
    # for initial testing only as this will grow without bounds
    cdef public dict cluster_cache
    cdef public npc.uint64_t[:, :] hash_functions
    cdef public int sub_k
    cdef public int n_hashes
    cdef public int array_size

    cpdef _build_tables(self)
    cpdef _build_indices(self)
    cpdef _drop_indices(self)

    cdef npc.uint64_t[:, :] load_hash_functions(self)
    cdef build_save_hash_functions(self)

    cdef npc.uint8_t[:, :] get_cluster_members(self, int centroid_id)
    cdef Cluster get_cluster(self, int centroid_id)

    cdef store_inner_clusters(self, Cluster cluster)
    cdef retrieve_inner_clusters(self, Cluster cluster)

    cdef store_bloom_grid(self, Cluster cluster)
    cpdef build_and_store_bloom_grid(self, int centroid_id)
    cpdef BloomGrid retrieve_bloom_grid(self, int centroid_id)

    cpdef load_other(self, GridCoverDB other)
