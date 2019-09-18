
import sqlite3
import numpy as np

from multiprocessing import Lock

from .ram cimport RotatingRamifier


cdef simple_list(sql_cursor):
    return [el[0] for el in sql_cursor]


cdef class GridCoverDB:

    def __cinit__(self, conn, ramifier=None, box_side_len=None, multithreaded=False):
        self.conn = conn
        self.cursor = self.conn.cursor()
        self.cursor.execute('CREATE TABLE IF NOT EXISTS basics (name text, value text)')
        self.cursor.execute('CREATE TABLE IF NOT EXISTS kmers (centroid_id int, seq text)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS IX_kmers_centroid ON kmers(centroid_id)')
        self.cursor.execute('CREATE TABLE IF NOT EXISTS centroids (centroid_id int, vals text)')

        self.centroid_cache = {}
        self.cluster_cache = {}
        if ramifier is None:
            self.ramifier = self.load_ramifier()
            self.box_side_len = float(self.cursor.execute(
                    'SELECT value FROM basics WHERE name=?', ('box_side_len',)
            ).fetchone()[0])
        else:
            assert box_side_len is not None
            self.box_side_len = box_side_len
            self.cursor.execute(
                'INSERT INTO basics VALUES (?,?)',
                ('box_side_len', box_side_len)
            )
            self.ramifier = ramifier
            self.save_ramifier()

    cpdef get_kmers(self):
        return list(self.cursor.execute('SELECT * FROM kmers'))

    cpdef get_cluster_members(self, int centroid_id):
        """Retrieve the members of a cluster. 

        Called often during search, wrapped with cache.
        """
        if centroid_id in self.cluster_cache:
            return self.cluster_cache[centroid_id]
        vals = self.cursor.execute('SELECT seq FROM kmers WHERE centroid_id=?', (centroid_id,))
        vals = simple_list(vals)
        self.cluster_cache[centroid_id] = vals
        return vals

    cpdef _add_pre_point_to_cluster(self, str centroid_str, str kmer):
        try:
            centroid_id = self.centroid_cache[centroid_str]
        except KeyError:
            centroid_id = len(self.centroid_cache)
            self.centroid_cache[centroid_str] = centroid_id
            self.cursor.execute('INSERT INTO centroids VALUES (?,?)', (centroid_id, centroid_str))

        self.cursor.execute('INSERT INTO kmers VALUES (?,?)', (centroid_id, kmer))

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
        centroid_strs = list(self.cursor.execute('SELECT * FROM centroids'))
        cdef double [:, :] centroids = np.ndarray((len(centroid_strs), self.ramifier.d))
        for i, centroid_str in centroid_strs:
            for j, val in enumerate(centroid_str.split(',')):
                centroids[i, j] = float(val)
        return centroids

    def close(self):
        """Close the DB and flush data to disk."""
        self.conn.commit()
        self.conn.close()

    def commit(self):
        """Flush data to disk."""
        self.conn.commit()

    cpdef load_other(self, GridCoverDB other):
        """Add contents of other db to this db."""
        my_centroid_strs = {
            centroid_str: cid
            for cid, centroid_str in self.cursor.execute('SELECT * FROM centroids')
        }
        centroid_id_remap = {}
        for other_id, other_centroid_str in other.cursor.execute('SELECT * FROM centroids'):
            if other_centroid_str in my_centroid_strs:
                centroid_id_remap[other_id] = my_centroid_strs[other_centroid_str]
            else:
                new_id = len(my_centroid_strs)
                centroid_id_remap[other_id] = new_id
                self.cursor.execute(
                    'INSERT INTO centroids VALUES (?,?)',
                    (new_id, other_centroid_str)
                )

        for other_id, kmer in other.cursor.execute('SELECT * FROM kmers'):
            new_id = centroid_id_remap[other_id]
            self.cursor.execute('INSERT INTO kmers VALUES (?,?)', (new_id, kmer))
        self.commit()

    cdef save_ramifier(self):
        stringify = lambda M: ','.join([str(el) for el in M])
        self.cursor.executemany(
            'INSERT INTO basics VALUES (?,?)',
            [
                ('k', str(self.ramifier.k)),
                ('d', str(self.ramifier.d)),
                ('center', stringify(self.ramifier.center)),
                ('scale', stringify(self.ramifier.scale)),
                ('rotation', stringify(np.array(self.ramifier.rotation).flatten())),
            ]
        )

    cdef RotatingRamifier load_ramifier(self):

        def get_basic(key):
            val = self.cursor.execute('SELECT value FROM basics WHERE name=?', (key,))
            return val.fetchone()[0]

        def numpify(key):
            val = get_basic(key)
            return np.array([float(el) for el in val.split(',')])

        k = int(get_basic('k'))
        d = int(get_basic('d'))
        center = numpify('center')
        scale = numpify('scale')
        rotation = numpify('rotation')
        rotation = np.reshape(rotation, (4 * k, 4 * k))

        return RotatingRamifier(k, d, rotation, center, scale)

    @classmethod
    def load_from_filepath(cls, filepath):
        """Return a GridCoverDB."""
        connection = sqlite3.connect(filepath)
        return GridCoverDB(connection)

    def centroids(self):
        return np.array(self.c_get_centroids())
