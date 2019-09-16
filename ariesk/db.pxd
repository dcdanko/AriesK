
from .ram cimport RotatingRamifier


cdef simple_list(sql_cursor)


cdef class GridCoverDB:
    cdef public float box_side_len
    cdef public object conn
    cdef public object cursor
    cdef public object centroid_cache
    cdef public RotatingRamifier ramifier

    cpdef get_kmers(self)
    cpdef get_cluster_members(self, int centroid_id)
    cpdef _add_pre_point_to_cluster(self, str centroid_str, str kmer)
    cpdef add_point_to_cluster(self, centroid, str kmer)
    cpdef load_other(self, GridCoverDB other)
    cdef double [:, :] c_get_centroids(self)
    cdef RotatingRamifier load_ramifier(self)
    cdef save_ramifier(self)


