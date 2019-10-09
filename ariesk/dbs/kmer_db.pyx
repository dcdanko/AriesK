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
from ariesk.dbs.core_db cimport CoreDB

BUFFER_SIZE = 10 * 1000

cdef simple_list(sql_cursor):
    return [el[0] for el in sql_cursor]


cdef class GridCoverDB(CoreDB):

    def __cinit__(self, conn, ramifier=None, box_side_len=None):
        super().__init__(conn, ramifier=ramifier, box_side_len=box_side_len)
        self._build_tables()
        self._build_indices()
        self.cluster_cache = {}

        self.sub_k = 7
        self.n_hashes = 8
        self.array_size = 2 ** 10  # must be a power of 2
        try:
            self.hash_functions = self.load_hash_functions()
        except IndexError:
            self.build_save_hash_functions()

    cpdef _build_tables(self):
        self.conn.execute(
            '''CREATE TABLE IF NOT EXISTS blooms (
                centroid_id int NOT NULL UNIQUE,
                col_k int, row_k int, grid_width int, grid_height int,
                bitarray BLOB, bitgrid BLOB, row_hashes BLOB, col_hashes BLOB
            )'''
        )
        self.conn.execute(
            '''CREATE TABLE IF NOT EXISTS inner_clusters (
            centroid_id int NOT NULL UNIQUE,
            inner_centers text, inner_cluster_members BLOB
            )'''
        )

    cpdef _build_indices(self):
        self.conn.execute('CREATE INDEX IF NOT EXISTS IX_seqs_centroid ON seqs(centroid_id)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS IX_blooms_centroid ON blooms(centroid_id)')

    cpdef _drop_indices(self):
        self.conn.execute('DROP INDEX IF EXISTS IX_seqs_centroid')
        self.conn.execute('DROP INDEX IF EXISTS IX_blooms_centroid')

    def py_get_cluster_members(self, int centroid_id):
        return np.array(self.get_cluster_members(centroid_id))

    cdef npc.uint8_t[:, :] get_cluster_members(self, int centroid_id):
        """Retrieve the members of a cluster. 

        Called often during search, wrapped with cache.
        """
        cdef int i, j
        cdef list kmers = simple_list(self.conn.execute('SELECT seq FROM seqs WHERE centroid_id=?', (centroid_id,)))
        cdef npc.uint8_t [:, :] binary_kmers = np.ndarray((len(kmers), self.ramifier.k), dtype=np.uint8)
        cdef const npc.uint8_t [:] binary_kmer
        for i, kmer in enumerate(kmers):
            binary_kmer = np.frombuffer(kmer, dtype=np.uint8)
            for j in range(self.ramifier.k):
                binary_kmers[i, j] = binary_kmer[j]
        return binary_kmers

    cdef Cluster get_cluster(self, int centroid_id):
        if centroid_id in self.cluster_cache:
            return self.cluster_cache[centroid_id]
        cdef npc.uint8_t[:, :] seqs = self.get_cluster_members(centroid_id)
        cdef Cluster cluster = Cluster(centroid_id, seqs, self.sub_k)
        try:
            cluster.bloom_grid = self.retrieve_bloom_grid(centroid_id)
        except IndexError:
            cluster.build_bloom_grid(self.array_size, self.hash_functions)
            self.store_bloom_grid(cluster)
        try:
            self.retrieve_inner_clusters(cluster)
        except IndexError:
            cluster.build_subclusters(self.ramifier.k // 5)
            self.store_inner_clusters(cluster)
        self.cluster_cache[centroid_id] = cluster
        return cluster

    cdef store_inner_clusters(self, Cluster cluster):
        self.conn.execute(
            'INSERT INTO inner_clusters VALUES (?,?,?)',
            (
                cluster.centroid_id,
                ','.join([str(el) for el in cluster.inner_centers]),
                np.array(cluster.inner_clusters, dtype=np.uint64).tobytes(),
            )
        )

    cdef retrieve_inner_clusters(self, Cluster cluster):
        packed = list(self.conn.execute(
            'SELECT * FROM inner_clusters WHERE centroid_id=?', (cluster.centroid_id,)
        ))[0]
        cdef list centers = [int(el) for el in packed[1].split(',')]
        cdef const npc.uint8_t[:] raw_ic_members = packed[2]
        cdef const npc.uint64_t[:, :] ic_members = np.reshape(
            np.frombuffer(raw_ic_members, dtype=np.uint64),
            (cluster.n_seqs, cluster.n_seqs)
        )
        cluster.inner_centers = centers
        cluster.inner_clusters = np.copy(ic_members)

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

    cpdef build_and_store_bloom_grid(self, int centroid_id):
        cdef Cluster cluster = self.get_cluster(centroid_id)

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

        for other_id, kmer, annotation in other.conn.execute('SELECT * FROM seqs'):
            new_id = centroid_id_remap[other_id]
            self.conn.execute('INSERT INTO seqs VALUES (?,?,?)', (new_id, kmer, annotation))
        self.commit()

    cdef npc.uint64_t[:, :] load_hash_functions(self):
        val = list(self.conn.execute(
            'SELECT value FROM basics WHERE name=?',
            ('hash_functions',)
        ))[0][0]
        cdef npc.uint64_t[:, :] hash_functions = np.array([
            [int(el) for el in row.split(',')]
            for row in val.split(',;') if len(row) > 0
        ], dtype=np.uint64)
        return hash_functions

    cdef build_save_hash_functions(self):
        cdef int i, j
        self.hash_functions = np.ndarray((self.n_hashes, self.sub_k), dtype=np.uint64)
        for i in range(self.n_hashes):
            for j, val in enumerate(np.random.permutation(self.sub_k)):
                self.hash_functions[i, j] = val
        cdef str hf_str = ''
        for i in range(self.hash_functions.shape[0]):
            for j in range(self.hash_functions.shape[1]):
                hf_str += str(self.hash_functions[i, j])
                hf_str += ','
            hf_str += ';'
        self.conn.execute('INSERT INTO basics VALUES (?,?)', ('hash_functions', hf_str))
        self.conn.commit()

    @classmethod
    def load_from_filepath(cls, filepath):
        """Return a GridCoverDB."""
        connection = sqlite3.connect(filepath, cached_statements=10 * 1000)
        return GridCoverDB(connection)
