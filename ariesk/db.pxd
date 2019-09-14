

cdef class GridCoverDB:
    cdef public float box_side_len

    cdef str [] get_cluster_members(self, centroid)

    cdef add_point_to_cluster(self, centroid, str kmer)

    cdef double [:, :] get_centroids(self)

    cdef RotatingRamifier get_ramifier(self)
