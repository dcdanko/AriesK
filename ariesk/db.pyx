
import sqlite3
import numpy as np
cimport numpy as npc

from multiprocessing import Lock
from pybloom import BloomFilter

from libcpp.string cimport string
from libcpp.vector cimport vector

from .ram cimport RotatingRamifier


cdef simple_list(sql_cursor):
    return [el[0] for el in sql_cursor]


cdef class GridCoverDB:

    def __cinit__(self, conn, ramifier=None, box_side_len=None, multithreaded=False):
        self.conn = conn
        self.cursor = self.conn.cursor()
        self.cursor.execute('CREATE TABLE IF NOT EXISTS basics (name text, value text)')
        self.cursor.execute('CREATE TABLE IF NOT EXISTS kmers (centroid_id int, seq BLOB)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS IX_kmers_centroid ON kmers(centroid_id)')
        self.cursor.execute('CREATE TABLE IF NOT EXISTS centroids (centroid_id int, vals BLOB)')

        self.centroid_cache = {}
        self.cluster_cache = {}
        self.bloom_cache = {}
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
        cdef dict base_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        cdef list out = []
        for cid, binary_kmer in self.cursor.execute('SELECT * FROM kmers'):
            binary_kmer = np.frombuffer(binary_kmer, dtype=np.uint8)
            kmer = ''.join([base_map[base] for base in binary_kmer])
            out.append((cid, kmer))
        return out

    def py_get_cluster_members(self, int centroid_id):
        return np.array(self.get_cluster_members(centroid_id))

    cdef npc.uint8_t [:, :] get_cluster_members(self, int centroid_id):
        """Retrieve the members of a cluster. 

        Called often during search, wrapped with cache.
        """
        if centroid_id in self.cluster_cache:
            return self.cluster_cache[centroid_id]
        cdef int i, j
        cdef list kmers = simple_list(self.cursor.execute('SELECT seq FROM kmers WHERE centroid_id=?', (centroid_id,)))
        cdef npc.uint8_t [:, :] binary_kmers = np.ndarray((len(kmers), self.ramifier.k), dtype=np.uint8)
        cdef const npc.uint8_t [:] binary_kmer
        for i, kmer in enumerate(kmers):
            binary_kmer = np.frombuffer(kmer, dtype=np.uint8)
            for j in range(self.ramifier.k):
                binary_kmers[i, j] = binary_kmer[j]
        self.cluster_cache[centroid_id] = binary_kmers
        return binary_kmers

    cdef get_bloom_filter(self, int centroid_id):
        if centroid_id in self.bloom_cache[centroid_id]:
            return self.bloom_cache[centroid_id]

        cdef npc.uint8_t [:, :] vals = self.get_cluster_members(centroid_id)
        bloom_filter = BloomFilter(1000, error_rate=0.05)
        for i in range(vals.shape[0]):
            for j in range(self.ramifier.k - 5 + 1):
                bloom_filter.add(vals[i, j:j + 5])
        self.bloom_cache[centroid_id] = bloom_filter
        return bloom_filter


    def py_add_point_to_cluster(self, npc.ndarray centroid, str kmer):
        cdef dict base_map = {'A': 0., 'C': 1., 'G': 2, 'T': 3}
        cdef npc.uint8_t [:] binary_kmer = np.array([base_map[base] for base in kmer], dtype=np.uint8)
        self.add_point_to_cluster(centroid, binary_kmer)

    cdef add_point_to_cluster(self, double [:] centroid, npc.uint8_t [:] binary_kmer):
        """Store a new point in the db. centroid is not assumed to exist.

        Called often during build/merge
        """
        cdef int centroid_id
        if tuple(centroid) in self.centroid_cache:
            centroid_id = self.centroid_cache[tuple(centroid)]
        else:
            centroid_id = len(self.centroid_cache)
            self.centroid_cache[tuple(centroid)] = centroid_id
            self.cursor.execute('INSERT INTO centroids VALUES (?,?)', (centroid_id, np.array(centroid, dtype=float).tobytes()))

        self.cursor.execute('INSERT INTO kmers VALUES (?,?)', (centroid_id, np.array(binary_kmer, dtype=np.uint8).tobytes()))


    cdef double [:, :] c_get_centroids(self):
        """Return a memoryview on cetnroids in this db.

        Called just once on database load.
        """
        centroid_strs = list(self.cursor.execute('SELECT * FROM centroids'))
        cdef int i, j
        cdef double [:, :] centroids = np.ndarray((len(centroid_strs), self.ramifier.d))
        cdef const double [:] centroid
        for i, centroid_str in centroid_strs:
            centroid = np.frombuffer(centroid_str, dtype=float, count=self.ramifier.d)
            for j in range(self.ramifier.d):
                centroids[i, j] = centroid[j]
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
