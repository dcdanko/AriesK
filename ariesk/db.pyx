# cython: profile=True
# cython: linetrace=True
# cython: language_level=3

import sqlite3
import numpy as np
cimport numpy as npc

from multiprocessing import Lock

from libcpp.string cimport string
from libcpp.vector cimport vector

from ariesk.utils.bloom_filter cimport BloomGrid
from ariesk.utils.kmers cimport encode_kmer, decode_kmer
from ariesk.ram cimport RotatingRamifier
from ariesk.cluster cimport Cluster

BUFFER_SIZE = 10 * 1000

cdef simple_list(sql_cursor):
    return [el[0] for el in sql_cursor]


cdef npc.uint64_t fnva(double[:] data):
    cdef npc.uint64_t hval = 0xcbf29ce484222325
    cdef int i
    cdef max_int = 2 ** 4
    for i in range(data.shape[0]):
        hval = hval ^ (<npc.int64_t> data[i])
        hval = hval * 0x100000001b3 % (max_int)
    return hval


cdef class GridCoverDB:

    def __cinit__(self, conn, ramifier=None, box_side_len=None, multithreaded=False):
        self.conn = conn
        self._build_tables()

        self.centroid_insert_buffer = [None] * BUFFER_SIZE
        self.centroid_buffer_filled = 0
        self.kmer_insert_buffer = [None] * BUFFER_SIZE
        self.kmer_buffer_filled = 0

        self.centroid_cache = {}
        self.cluster_cache = {}
        if ramifier is None:
            self.ramifier = self.load_ramifier()
            self.box_side_len = float(self.conn.execute(
                'SELECT value FROM basics WHERE name=?', ('box_side_len',)
            ).fetchone()[0])
        else:
            assert box_side_len is not None
            self.box_side_len = box_side_len
            self.conn.execute(
                'INSERT INTO basics VALUES (?,?)',
                ('box_side_len', box_side_len)
            )
            self.ramifier = ramifier
            self.save_ramifier()

    cdef _build_tables(self):
        self.conn.execute('CREATE TABLE IF NOT EXISTS basics (name text, value text)')
        self.conn.execute('CREATE TABLE IF NOT EXISTS kmers (centroid_id int, seq BLOB)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS IX_kmers_centroid ON kmers(centroid_id)')
        self.conn.execute('CREATE TABLE IF NOT EXISTS centroids (centroid_id int, vals BLOB)')
        self.conn.execute(
            '''CREATE TABLE IF NOT EXISTS blooms (
                centroid_id int, col_k int, row_k int, grid_width int, grid_height int,
                bitarray BLOB, bitgrid BLOB, row_hashes BLOB, col_hashes BLOB
            )'''
        )
        self.conn.execute('CREATE INDEX IF NOT EXISTS IX_blooms_centroid ON blooms(centroid_id)')

    cpdef get_kmers(self):
        cdef list out = []
        cdef const npc.uint8_t [:] binary_kmer
        for cid, binary_kmer in self.conn.execute('SELECT * FROM kmers'):
            binary_kmer = np.frombuffer(binary_kmer, dtype=np.uint8)
            kmer = decode_kmer(binary_kmer)
            out.append((cid, kmer))
        return out

    cdef npc.uint8_t [:, :] get_encoded_kmers(self):
        cdef int i, j
        cdef list kmers = simple_list(self.conn.execute('SELECT seq FROM kmers'))
        cdef npc.uint8_t [:, :] binary_kmers = np.ndarray((len(kmers), self.ramifier.k), dtype=np.uint8)
        cdef const npc.uint8_t [:] binary_kmer
        for i, kmer in enumerate(kmers):
            binary_kmer = np.frombuffer(kmer, dtype=np.uint8)
            for j in range(self.ramifier.k):
                binary_kmers[i, j] = binary_kmer[j]
        return binary_kmers

    def py_get_cluster_members(self, int centroid_id):
        return np.array(self.get_cluster_members(centroid_id))

    cdef npc.uint8_t [:, :] get_cluster_members(self, int centroid_id):
        """Retrieve the members of a cluster. 

        Called often during search, wrapped with cache.
        """
        cdef int i, j
        cdef list kmers = simple_list(self.conn.execute('SELECT seq FROM kmers WHERE centroid_id=?', (centroid_id,)))
        cdef npc.uint8_t [:, :] binary_kmers = np.ndarray((len(kmers), self.ramifier.k), dtype=np.uint8)
        cdef const npc.uint8_t [:] binary_kmer
        for i, kmer in enumerate(kmers):
            binary_kmer = np.frombuffer(kmer, dtype=np.uint8)
            for j in range(self.ramifier.k):
                binary_kmers[i, j] = binary_kmer[j]
        return binary_kmers

    cdef Cluster get_cluster(self, int centroid_id, int filter_len, npc.uint64_t[:, :] hashes, int sub_k):
        if centroid_id in self.cluster_cache:
            return self.cluster_cache[centroid_id]
        cdef npc.uint8_t[:, :] seqs = self.get_cluster_members(centroid_id)
        cdef Cluster cluster = Cluster(centroid_id, seqs, sub_k)
        cluster.build_bloom_grid(filter_len, hashes)
        cluster.build_subclusters(self.ramifier.k // 5)
        self.cluster_cache[centroid_id] = cluster
        return cluster

    cdef store_bloom_grid(self, Cluster cluster):
        self.conn.execute(
            'INSERT INTO blooms VALUES (?,?,?,?,?,?,?,?,?)',
            (
                cluster.centroid_id,
                cluster.bloom_grid.col_k,
                cluster.bloom_grid.row_k,
                cluster.bloom_grid.grid_width,
                cluster.bloom_grid.grid_height,
                np.array(cluster.bloom_grid.bitarray, dtype=np.uint8).tobytes(),
                np.array(cluster.bloom_grid.bitgrid, dtype=np.uint8).tobytes(),
                np.array(cluster.bloom_grid.row_hashes, dtype=np.uint64).tobytes(),
                np.array(cluster.bloom_grid.col_hashes, dtype=np.uint64).tobytes(),
            )
        )

    cpdef build_and_store_bloom_grid(self, int centroid_id, int filter_len, npc.uint64_t[:, :] hashes, int sub_k):
        cdef Cluster cluster = self.get_cluster(centroid_id, filter_len, hashes, sub_k)
        self.store_bloom_grid(cluster)

    cpdef BloomGrid retrieve_bloom_grid(self, int centroid_id):
        cdef int grid_width, grid_height, col_k, row_k
        packed = list(self.conn.execute(
            'SELECT * FROM blooms WHERE centroid_id=?', (centroid_id,)
        ))[0]
        col_k = packed[1]
        row_k = packed[2]
        grid_width = packed[3]
        grid_height = packed[4]
        cdef const npc.uint8_t[:] raw_bitarray   = packed[5]
        cdef const npc.uint8_t[:] raw_bitgrid    = packed[6]
        cdef const npc.uint8_t[:] raw_row_hashes = packed[7]
        cdef const npc.uint8_t[:] raw_col_hashes = packed[8]
        cdef const npc.uint8_t[:] bitarray = np.frombuffer(raw_bitarray, dtype=np.uint8)
        cdef const npc.uint8_t[:, :] bitgrid = np.reshape(
            np.frombuffer(raw_bitgrid, dtype=np.uint8),
            (grid_height, grid_width)
        )
        cdef const npc.uint64_t[:] flat_row_hashes = np.frombuffer(raw_row_hashes, dtype=np.uint64)
        cdef int n_row_hashes = flat_row_hashes.shape[0] // row_k
        cdef const npc.uint64_t[:, :] row_hashes = np.reshape(
            flat_row_hashes, (n_row_hashes, row_k)
        )
        cdef const npc.uint64_t[:] flat_col_hashes = np.frombuffer(raw_col_hashes, dtype=np.uint64)
        cdef int n_col_hashes = flat_col_hashes.shape[0] // col_k
        cdef const npc.uint64_t[:, :] col_hashes = np.reshape(
            flat_col_hashes, (n_col_hashes, col_k)
        )
        cdef BloomGrid bg = BloomGrid(
            col_k, row_k, grid_width, grid_height, np.copy(row_hashes), np.copy(col_hashes)
        )
        cdef int i, j
        for i in range(bg.bitarray.shape[0]):
            bg.bitarray[i] = bitarray[i]
        for i in range(bg.bitgrid.shape[0]):
            for j in range(bg.bitgrid.shape[1]):
                bg.bitgrid[i, j] = bitgrid[i, j]
        return bg

    def py_add_point_to_cluster(self, npc.ndarray centroid, str kmer):
        cdef npc.uint8_t [:] binary_kmer = encode_kmer(kmer)
        self.add_point_to_cluster(centroid, binary_kmer)

    cdef add_point_to_cluster(self, double[:] centroid, npc.uint8_t [::] binary_kmer):
        """Store a new point in the db. centroid is not assumed to exist.

        Called often during build/merge
        """
        cdef int centroid_id
        cdef tuple centroid_key = tuple(centroid)
        if centroid_key in self.centroid_cache:
            centroid_id = self.centroid_cache[centroid_key]
        else:
            self.centroid_cache[centroid_key] = len(self.centroid_cache)
            centroid_id = self.centroid_cache[centroid_key]
            self.centroid_insert_buffer[self.centroid_buffer_filled] = (
                centroid_id, np.array(centroid, dtype=float).tobytes()
            )
            self.centroid_buffer_filled += 1
        self.kmer_insert_buffer[self.kmer_buffer_filled] = (
            centroid_id, np.array(binary_kmer, dtype=np.uint8).tobytes()
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
                    'INSERT INTO kmers VALUES (?,?)',
                    self.kmer_insert_buffer
                )
            else:
                self.conn.executemany(
                    'INSERT INTO kmers VALUES (?,?)',
                    self.kmer_insert_buffer[:self.kmer_buffer_filled]
                )
        self.centroid_buffer_filled = 0
        self.kmer_buffer_filled = 0

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
        self._clear_buffer()
        self.conn.commit()
        self.conn.close()

    def commit(self):
        """Flush data to disk."""
        self._clear_buffer()
        self.conn.commit()

    cpdef load_other(self, GridCoverDB other):
        """Add contents of other db to this db."""
        my_centroid_strs = {
            centroid_str: cid
            for cid, centroid_str in self.conn.execute('SELECT * FROM centroids')
        }
        centroid_id_remap = {}
        for other_id, other_centroid_str in other.conn.execute('SELECT * FROM centroids'):
            if other_centroid_str in my_centroid_strs:
                centroid_id_remap[other_id] = my_centroid_strs[other_centroid_str]
            else:
                new_id = len(my_centroid_strs)
                centroid_id_remap[other_id] = new_id
                self.conn.execute(
                    'INSERT INTO centroids VALUES (?,?)',
                    (new_id, other_centroid_str)
                )

        for other_id, kmer in other.conn.execute('SELECT * FROM kmers'):
            new_id = centroid_id_remap[other_id]
            self.conn.execute('INSERT INTO kmers VALUES (?,?)', (new_id, kmer))
        self.commit()

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
        return GridCoverDB(connection)

    def centroids(self):
        return np.array(self.c_get_centroids())
