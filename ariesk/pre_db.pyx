# cython: profile=True
# cython: linetrace=True
# cython: language_level=3

import sqlite3
import numpy as np
cimport numpy as npc
cimport cython
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdio cimport *
from posix.stdio cimport * # FILE, fopen, fclose
from libc.stdlib cimport malloc, free

from ariesk.utils.kmers cimport encode_kmer, encode_kmer_from_buffer
from ariesk.ram cimport RotatingRamifier
from ariesk.cluster cimport Cluster

BUFFER_SIZE = 10 * 1000

cdef simple_list(sql_cursor):
    return [el[0] for el in sql_cursor]


cdef class PreDB:

    def __cinit__(self, conn, ramifier=None):
        self.conn = conn
        self._build_tables()

        self.kmer_insert_buffer = [None] * BUFFER_SIZE
        self.kmer_buffer_filled = 0

        if ramifier is None:
            self.ramifier = self.load_ramifier()
        else:
            self.ramifier = ramifier
            self.save_ramifier()

    cdef _build_tables(self):
        self.conn.execute('CREATE TABLE IF NOT EXISTS basics (name text, value text)')
        self.conn.execute('CREATE TABLE IF NOT EXISTS kmers (rft BLOB, seq BLOB)')

    def py_add_kmer(self, str kmer):
        cdef npc.uint8_t [:] binary_kmer = encode_kmer(kmer)
        self.c_add_kmer(binary_kmer)

    cdef c_add_kmer(self, npc.uint8_t [:] binary_kmer):
        cdef double[:] rft = self.ramifier.c_ramify(binary_kmer)
        self.add_point(rft, binary_kmer)

    def py_add_point(self, npc.ndarray rft, str kmer):
        cdef npc.uint8_t [:] binary_kmer = encode_kmer(kmer)
        self.add_point(rft, binary_kmer)

    cdef add_point(self, double[:] rft, npc.uint8_t [::] binary_kmer):
        self.kmer_insert_buffer[self.kmer_buffer_filled] = (
            np.array(rft, dtype=float).tobytes(),
            np.array(binary_kmer, dtype=np.uint8).tobytes()
        )
        self.kmer_buffer_filled += 1
        if self.kmer_buffer_filled >= BUFFER_SIZE:
            self._clear_buffer()

    cdef _clear_buffer(self):
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
        self.kmer_buffer_filled = 0

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fast_add_kmers_from_fasta(self, str filename, num_to_add=0):
        cdef FILE * cfile = fopen(filename.encode("UTF-8"), "rb")
        if cfile == NULL:
            raise FileNotFoundError(2, "No such file or directory: '%s'" % filename)

        cdef int n_added = 0
        cdef char * line = NULL
        cdef size_t l = 0
        cdef ssize_t read
        cdef size_t n_kmers_in_line, i
        cdef npc.uint8_t[:] kmer
        while (num_to_add <= 0) or (n_added < num_to_add):
            getline(&line, &l, cfile)  # header
            read = getdelim(&line, &l, b'>', cfile)  # read
            if read == -1: break
            while (num_to_add <= 0) or (n_added < num_to_add):
                if line[0] == b'\n':
                    line += 1
                kmer = encode_kmer_from_buffer(line, self.ramifier.k)
                if (num_to_add > 0) and (n_added >= num_to_add):
                    break
                if kmer[self.ramifier.k - 1] > 3:
                    break
                self.c_add_kmer(kmer)
                n_added += 1
                line += 1
            line = NULL  # I don't understand why this line is necessary but
                         # without it the program throws a strange error: 
                         # `pointer being realloacted was not allocated`
        fclose(cfile)
        return n_added

    @classmethod
    def load_from_filepath(cls, filepath, ramifier=None):
        """Return a GridCoverDB."""
        connection = sqlite3.connect(filepath, cached_statements=10 * 1000)
        return PreDB(connection, ramifier=ramifier)
