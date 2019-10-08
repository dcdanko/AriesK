# cython: profile=True
# cython: linetrace=True
# cython: language_level=3

import numpy as np
import sqlite3
cimport numpy as npc
cimport cython
from libc.stdio cimport *
from posix.stdio cimport * # FILE, fopen, fclose
from libc.stdlib cimport malloc, free

from ariesk.utils.kmers cimport encode_kmer, encode_kmer_from_buffer
from ariesk.ram cimport RotatingRamifier
from ariesk.db cimport GridCoverDB
from ariesk.pre_db import PreDB


cdef class GridCoverBuilder:
    cdef public RotatingRamifier ramifier
    cdef public GridCoverDB db
    cdef public int num_kmers_added
    cdef public int max_size

    def __cinit__(self, db):
        self.db = db
        self.ramifier = db.ramifier
        self.num_kmers_added = 0

    def _bulk_pre_add_kmers(self, lines, sep=','):
        kmers = [lines.strip().split(sep)[0] for line in lines]
        out = [(self._pre_add_kmer(kmer), kmer) for kmer in kmers]
        return out

    cdef str _pre_add_kmer(self, kmer):
        cdef double [:] centroid_rft = np.floor(self.ramifier.c_ramify(kmer) / self.db.box_side_len)
        cdef str centroid_str = ','.join([str(el) for el in centroid_rft])
        return centroid_str

    cpdef add_kmer(self, str kmer):
        cdef npc.uint8_t [:] binary_kmer = encode_kmer(kmer)
        self.c_add_kmer(binary_kmer)

    cdef c_add_kmer(self, npc.uint8_t [:] binary_kmer):
        cdef double[:] centroid_rft = np.floor(
            self.ramifier.c_ramify(binary_kmer) / self.db.box_side_len,
            casting='safe'
        )
        self.db.add_point_to_cluster(centroid_rft, binary_kmer)
        self.num_kmers_added += 1

    def commit(self):
        self.db.commit()

    def close(self):
        self.db.close()

    def bulk_add_kmers(self, kmers):
        for kmer in kmers:
            self.add_kmer(kmer)

    def add_kmers_from_file(self, str filename, sep=',', start=0, num_to_add=0, preload=False):
        with open(filename) as f:
            n_added = 0
            if start > 0:
                f.readlines(start)
            for line in f:
                if (num_to_add <= 0) or (n_added < num_to_add):
                    kmer = line.split(sep)[0]
                    self.add_kmer(kmer)
                    n_added += 1
            return n_added

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fast_add_kmers_from_file(self, str filename, num_to_add=0):
        cdef FILE * cfile = fopen(filename.encode("UTF-8"), "rb")
        if cfile == NULL:
            raise FileNotFoundError(2, "No such file or directory: '%s'" % filename)

        cdef n_added = 0
        cdef char * line = NULL
        cdef size_t l = 0
        cdef ssize_t read
        cdef size_t n_kmers_in_line, i
        cdef npc.uint8_t[:] kmer
        while (num_to_add <= 0) or (n_added < num_to_add):
            read = getline(&line, &l, cfile)
            if read == -1: break
            if line[0] != b'>':
                n_kmers_in_line = l - self.ramifier.k + 1
                i = 0
                while (i < n_kmers_in_line) and ((num_to_add <= 0) or (n_added < num_to_add)):
                    kmer = encode_kmer_from_buffer(line, self.ramifier.k)
                    if (num_to_add > 0) and (n_added >= num_to_add):
                        break
                    if kmer[self.ramifier.k - 1] > 3:
                        break
                    self.c_add_kmer(kmer)
                    n_added += 1
                    line += 1
                    i += 1
            line = NULL  # I don't understand why this line is necessary but
                         # without it the program throws a strange error: 
                         # `pointer being realloacted was not allocated`
        fclose(cfile)
        return n_added

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
    def build_from_predb(cls, filepath, predb, box_side_len, logger=None, log_interval=10000):
        db = GridCoverDB(
            sqlite3.connect(filepath),
            ramifier=predb.ramifier,
            box_side_len=box_side_len
        )
        out = cls(db)
        out.add_kmers_from_predb(predb, logger=logger, log_interval=log_interval)
        return out

    def add_kmers_from_predb(self, predb, logger=None, log_interval=10000):
        cdef const npc.uint8_t[:] rft_blob
        cdef const npc.uint8_t[:] seq_blob
        cdef double[:] centroid_rft
        cdef const double[:] rft
        cdef const npc.uint8_t[:] kmer
        cdef npc.uint8_t[:] kmer2 = np.ndarray((self.ramifier.k,), np.uint8)
        for rft_blob, seq_blob in predb.conn.execute('SELECT * FROM kmers'):
            rft = np.frombuffer(rft_blob, dtype=float, count=self.ramifier.d)
            kmer = np.frombuffer(seq_blob, dtype=np.uint8)
            for i in range(kmer.shape[0]):
                kmer2[i] = kmer[i]
            centroid_rft = np.floor(np.array(rft) / self.db.box_side_len)
            self.db.add_point_to_cluster(centroid_rft, kmer2)
            self.num_kmers_added += 1
            if logger and (self.num_kmers_added % log_interval == 0):
                logger(self.num_kmers_added)

    @classmethod
    def from_filepath(cls, filepath, ramifier, box_side_len):
        db = GridCoverDB(sqlite3.connect(filepath), ramifier=ramifier, box_side_len=box_side_len)
        return cls(db)
