# cython: profile=True
# cython: linetrace=True
# cython: language_level=3

import sqlite3
import numpy as np
cimport numpy as npc
from ariesk.utils.kmers cimport encode_kmer, decode_kmer
from ariesk.ram cimport RotatingRamifier

BUFFER_SIZE = 10 * 1000


cdef simple_list(sql_cursor):
    return [el[0] for el in sql_cursor]


cdef class CoreDB:

    def __cinit__(self, conn, ramifier=None, box_side_len=None):
        self.conn = conn
        self.centroid_insert_buffer = [None] * BUFFER_SIZE
        self.centroid_buffer_filled = 0
        self.kmer_insert_buffer = [None] * BUFFER_SIZE
        self.kmer_buffer_filled = 0

        self.centroid_cache = {}
        cdef double[:, :] centroids
        cdef bytes centroid_key
        cdef int i

        self._build_core_tables()
        if ramifier is None:
            self.ramifier = self.load_ramifier()
            self.box_side_len = float(self.conn.execute(
                'SELECT value FROM basics WHERE name=?', ('box_side_len',)
            ).fetchone()[0])
            centroids = self.c_get_centroids()
            for i in range(centroids.shape[0]):
                centroid_key = np.array(centroids[i, :], dtype=float).tobytes()
                self.centroid_cache[centroid_key] = i
        else:
            if box_side_len is None:
                box_side_len = 1
            self.box_side_len = box_side_len
            self.conn.execute(
                'INSERT INTO basics VALUES (?,?)',
                ('box_side_len', box_side_len)
            )
            self.ramifier = ramifier
            self.save_ramifier()

    cpdef _build_core_tables(self):
        self.conn.execute('CREATE TABLE IF NOT EXISTS basics (name text, value text)')
        self.conn.execute('CREATE TABLE IF NOT EXISTS centroids (centroid_id int, vals BLOB)')
        self.conn.execute('CREATE TABLE IF NOT EXISTS seqs (centroid_id int, seq BLOB, annotation text)')

    cpdef get_kmers(self):
        cdef list out = []
        cdef const npc.uint8_t [:] binary_kmer
        for cid, binary_kmer, annotation in self.conn.execute('SELECT * FROM seqs'):
            binary_kmer = np.frombuffer(binary_kmer, dtype=np.uint8)
            kmer = decode_kmer(binary_kmer)
            out.append((cid, kmer))
        return out

    cdef npc.uint8_t [:, :] get_encoded_kmers(self):
        cdef int i, j
        cdef list kmers = simple_list(self.conn.execute('SELECT seq FROM seqs'))
        cdef npc.uint8_t [:, :] binary_kmers = np.ndarray((len(kmers), self.ramifier.k), dtype=np.uint8)
        cdef const npc.uint8_t [:] binary_kmer
        for i, kmer in enumerate(kmers):
            binary_kmer = np.frombuffer(kmer, dtype=np.uint8)
            for j in range(self.ramifier.k):
                binary_kmers[i, j] = binary_kmer[j]
        return binary_kmers

    cdef double [:, :] c_get_centroids(self):
        """Return a memoryview on cetnroids in this db.

        Called just once on database load.
        """
        centroid_strs = list(self.conn.execute('SELECT * FROM centroids'))
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
        self.commit()
        self.conn.commit()
        self.conn.close()

    def commit(self):
        raise NotImplementedError()

    def py_add_point_to_cluster(self, npc.ndarray centroid, str kmer, unicode annotation=None):
        cdef npc.uint8_t [:] binary_kmer = encode_kmer(kmer)
        self.add_point_to_cluster(centroid, binary_kmer, annotation=annotation)

    cdef add_point_to_cluster(self, double[:] centroid, npc.uint8_t [::] binary_kmer, str annotation=None):
        """Store a new point in the db. centroid is not assumed to exist.

        Called often during build/merge
        """
        cdef int centroid_id = self.add_centroid(centroid)
        self.kmer_insert_buffer[self.kmer_buffer_filled] = (
            centroid_id, np.array(binary_kmer, dtype=np.uint8).tobytes(), annotation
        )
        self.kmer_buffer_filled += 1
        if self.kmer_buffer_filled >= BUFFER_SIZE:
            self._clear_buffer()

    cdef _clear_buffer(self):
        if self.centroid_buffer_filled > 0:
            if self.centroid_buffer_filled == BUFFER_SIZE:
                self.conn.executemany(
                    'INSERT INTO centroids VALUES (?,?)',
                    self.centroid_insert_buffer
                )
            else:
                self.conn.executemany(
                    'INSERT INTO centroids VALUES (?,?)',
                    self.centroid_insert_buffer[:self.centroid_buffer_filled]
                )
        if self.kmer_buffer_filled > 0:
            if self.kmer_buffer_filled == BUFFER_SIZE:
                self.conn.executemany(
                    'INSERT INTO seqs VALUES (?,?,?)',
                    self.kmer_insert_buffer
                )
            else:
                self.conn.executemany(
                    'INSERT INTO seqs VALUES (?,?,?)',
                    self.kmer_insert_buffer[:self.kmer_buffer_filled]
                )
        self.centroid_buffer_filled = 0
        self.kmer_buffer_filled = 0

    cdef int add_centroid(self, double[:] centroid):
        cdef int centroid_id
        cdef bytes centroid_key = np.array(centroid, dtype=float).tobytes()
        if centroid_key in self.centroid_cache:
            centroid_id = self.centroid_cache[centroid_key]
        else:
            self.centroid_cache[centroid_key] = len(self.centroid_cache)
            centroid_id = self.centroid_cache[centroid_key]
            self.centroid_insert_buffer[self.centroid_buffer_filled] = (
                centroid_id, centroid_key
            )
            self.centroid_buffer_filled += 1
        if self.centroid_buffer_filled == BUFFER_SIZE:
            self._clear_buffer()
        return centroid_id

    cdef save_ramifier(self):
        stringify = lambda M: ','.join([str(el) for el in M])
        self.conn.executemany(
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
            val = self.conn.execute('SELECT value FROM basics WHERE name=?', (key,))
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
        connection = sqlite3.connect(filepath, cached_statements=10 * 1000)
        return CoreDB(connection)

    def centroids(self):
        return np.array(self.c_get_centroids())
