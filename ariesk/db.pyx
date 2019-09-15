
import sqlite3
import numpy as np

from .ram cimport RotatingRamifier


cdef class GridCoverDB:

    def __cinit__(self, conn):
        self.conn = conn
        self.cursor = self.conn.cursor()
        self.cursor.execute('CREATE TABLE basics (name text, value text)')
        self.cursor.execute('CREATE TABLE kmers (centroid_id int, seq text)')
        self.cursor.execute('CREATE TABLE centroids (centroid_id int, vals text)')

        self.centroid_cache = {}
        # self.ramifier = self.load_ramifier()

    cpdef get_cluster_members(self, centroid_id):
        """Retrieve the members of a cluster. 

        Called often during search, wrapped with cache.
        """
        vals = self.cursor.execute('SELECT seq FROM kmers WHERE centroid_id=?', (centroid_id,))
        return [el[0] for el in list(vals)]

    cpdef add_point_to_cluster(self, centroid, str kmer):
        """Store a new point in the db. centroid is not assumed to exist.

        Called often during build/merge
        """
        centroid_str = ','.join([str(el) for el in centroid])
        try:
            centroid_id = self.centroid_cache[centroid_str]
        except KeyError:
            centroid_id = len(self.centroid_cache)
            self.centroid_cache[centroid_str] = centroid_id
            self.cursor.execute('INSERT INTO centroids VALUES (?,?)', (centroid_id, centroid_str))

        self.cursor.execute('INSERT INTO kmers VALUES (?,?)', (centroid_id, kmer))

    cdef double [:, :] c_get_centroids(self):
        """Return a memoryview on cetnroids in this db.

        Called just once on database load.
        """
        centroid_strs = self.cursor.execute('SELECT * FROM centroids')
        cdef double [:, :] centroids = np.ndarray((len(centroid_strs), self.d))
        for i, centroid_str in enumerate(centroid_strs):
            for j, val in centroid_str.split(','):
                centroids[i, j] = float(val)
        return centroids

    cpdef close(self):
        """Close the DB and flush data to disk."""
        self.conn.commit()
        self.conn.close()

    cdef RotatingRamifier load_ramifier(self):
        k = int(self.cursor.execute('SELECT value FROM basics WHERE name="k"').fetchone())
        d = int(self.cursor.execute('SELECT value FROM basics WHERE name="d"'))

        center_str = int(self.cursor.execute('SELECT value FROM basics WHERE name="center"'))
        center = np.array([float(el) for el in ','.split(center_str)])
        scale_str = int(self.cursor.execute('SELECT value FROM basics WHERE name="scale"'))
        scale = np.array([float(el) for el in ','.split(scale_str)])
        rotate_str = int(self.cursor.execute('SELECT value FROM basics WHERE name="rotation"'))
        rotation = np.array([float(el) for el in ','.split(rotate_str)])
        rotation = np.reshape(rotation, (k, k))
        return RotatingRamifier(k, d, rotation, center, scale)

    @classmethod
    def load_from_filepath(cls, filepath):
        """Return a GridCoverDB."""
        connection = sqlite3.connect(filepath)
        return GridCoverDB(connection)

    def centroids(self):
        return np.array(self.get_centroids())
