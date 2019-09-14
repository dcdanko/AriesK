

cdef class GridCoverDB:


    cdef str [] get_cluster_members(self, centroid):
        """Retrieve the members of a cluster. 

        Called often during search, wrapped with cache.
        """
        pass

    cdef add_point_to_cluster(self, centroid, str kmer):
        """Store a new point in the db. centroid is not assumed to exist.

        Called often during build/merge
        """
        pass

    cdef double [:, :] get_centroids(self):
        """Return a memoryview on cetnroids in this db.

        Called just once on database load.
        """
        pass

    cdef RotatingRamifier get_ramifier(self):
        pass

    @classmethod
    def load_from_filepath(cls, filepath):
        """Return a GridCoverDB."""
        pass

    def centroids(self):
        return np.array(self.get_centroids())