# cython: profile=False
# cython: linetrace=False
# cython: language_level=3

import sqlite3
import numpy as np
cimport numpy as npc
from libc.stdio cimport *
from posix.stdio cimport * # FILE, fopen, fclose
from libc.stdlib cimport malloc, free
from math import ceil

from ariesk.utils.kmers cimport encode_kmer, decode_kmer, encode_seq_from_buffer
from ariesk.dbs.core_db cimport CoreDB
from ariesk.seed_align cimport get_target_kmers


SEQ_BLOCK_LEN = 10 * 1000
BUFFER_SIZE = 10 * 1000


cdef class PreContigDB(CoreDB):
    cdef public int seq_block_len
    cdef public int coord_buffer_filled
    cdef public list coord_buffer
    cdef public int contig_counter

    def __cinit__(self, conn, ramifier=None, box_side_len=None):
        super().__init__(conn, ramifier=ramifier, box_side_len=box_side_len)
        self.seq_block_len = SEQ_BLOCK_LEN
        self.coord_buffer_filled = 0
        self.coord_buffer = [None] * BUFFER_SIZE
        self._build_tables()
        self._build_indices()
        self.contig_counter = 0
        try:
            val = self.conn.execute('SELECT value FROM basics WHERE name=?', ('seq_block_len',))
            self.seq_block_len = int(list(val)[0][0])
        except IndexError:
            self.seq_block_len = SEQ_BLOCK_LEN
            self.conn.executemany(
                'INSERT INTO basics VALUES (?,?)',
                [
                    ('seq_block_len', str(self.seq_block_len)),
                ]
            )

    def get_all_contigs(self):
        cdef list out = []
        cdef const npc.uint8_t[:] contig
        for contig_id, seq, genome_name, contig_name, contig_start in self.conn.execute('SELECT * FROM contigs'):
            seq = np.frombuffer(seq, dtype=np.uint8)
            kmer = decode_kmer(seq)
            out.append((contig_id, kmer, genome_name, contig_name, contig_start))
        return out

    cpdef _build_tables(self):
        self.conn.execute(
            '''CREATE TABLE IF NOT EXISTS contigs (
            contig_id int, seq BLOB, genome_name text, contig_name text, contig_start int
            )'''
        )
        self.conn.execute(
            '''CREATE TABLE IF NOT EXISTS rfts (
                rft BLOB,
                rft_contig int,
                FOREIGN KEY (rft_contig) REFERENCES contigs(contig_id)
            )'''
        )

    cpdef _build_indices(self):
        self.conn.execute('CREATE INDEX IF NOT EXISTS IX_contigs_id ON contigs(contig_id)')

    cpdef _drop_indices(self):
        self.conn.execute('DROP INDEX IF EXISTS IX_contigs_id')

    cdef int add_contig_seq(self,
        str genome_name, str contig_name, int contig_start, npc.uint8_t[:] contig_section
    ):
        self.contig_counter += 1
        self.conn.execute(
            'INSERT INTO contigs VALUES (?,?,?,?,?)',
            (
                self.contig_counter,
                np.array(contig_section, dtype=np.uint8).tobytes(),
                genome_name,
                contig_name,
                contig_start,
            )
        )
        return self.contig_counter

    cdef add_rft_to_contig(self, double[:] rft, int contig_id):
        self.coord_buffer[self.coord_buffer_filled] = (
            np.array(rft).tobytes(),
            contig_id
        )
        self.coord_buffer_filled += 1
        if self.coord_buffer_filled == BUFFER_SIZE:
            self._clear_rft_buffer()

    def py_add_contig(self, str genome_name, str contig_name, str contig, int gap=1):
        self.add_contig(genome_name, contig_name, encode_kmer(contig), gap=gap)

    cdef add_contig(self, str genome_name, str contig_name, npc.uint8_t[:] contig, int gap=1):
        cdef npc.uint8_t[:] kmer
        cdef double[:] rft
        cdef int i
        cdef int contig_id = -1
        for i in range(0, contig.shape[0] - self.ramifier.k + 1, gap):
            if i % self.seq_block_len == 0:
                contig_id = self.add_contig_seq(
                    genome_name, contig_name, i,
                    contig[i:min(i + self.seq_block_len, contig.shape[0])]
                )
            kmer = contig[i:i + self.ramifier.k]
            rft = self.ramifier.c_ramify(kmer)
            self.add_rft_to_contig(rft, contig_id)

    cdef _clear_rft_buffer(self):
        if self.coord_buffer_filled > 0:
            if self.coord_buffer_filled == BUFFER_SIZE:
                self.conn.executemany(
                    'INSERT INTO rfts VALUES (?,?)',
                    self.coord_buffer
                )
            else:
                self.conn.executemany(
                    'INSERT INTO rfts VALUES (?,?)',
                    self.coord_buffer[:self.coord_buffer_filled]
                )
            self.coord_buffer_filled = 0

    def commit(self):
        self._clear_rft_buffer()
        self._clear_buffer()
        self.conn.commit()

    @classmethod
    def load_from_filepath(cls, filepath):
        """Return a GridCoverDB."""
        connection = sqlite3.connect(filepath, cached_statements=10 * 1000)
        return PreContigDB(connection)

    def fast_add_kmers_from_fasta(self, str filename):
        cdef FILE * cfile = fopen(filename.encode("UTF-8"), "rb")
        if cfile == NULL:
            raise FileNotFoundError(2, "No such file or directory: '%s'" % filename)

        cdef int n_added = 0
        cdef char * line = NULL
        cdef char * header = NULL
        cdef size_t l = 0
        cdef ssize_t read
        cdef size_t n_kmers_in_line, i
        cdef npc.uint8_t[:] seq
        while True:
            getline(&header, &l, cfile)
            read = getdelim(&line, &l, b'>', cfile)
            if read == -1: break
            seq = encode_seq_from_buffer(line, l)
            self.add_contig(filename, str(header).strip()[1:], seq)
            n_added += 1
            header = NULL
            line = NULL  # I don't understand why this line is necessary but
                         # without it the program throws a strange error: 
                         # `pointer being realloacted was not allocated`
        fclose(cfile)
        return n_added
