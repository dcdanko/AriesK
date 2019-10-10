# cython: profile=True
# cython: linetrace=True
# cython: language_level=3

import sqlite3
import numpy as np
cimport numpy as npc
from libc.stdio cimport *
from posix.stdio cimport * # FILE, fopen, fclose
from libc.stdlib cimport malloc, free

from ariesk.utils.kmers cimport encode_kmer, decode_kmer, encode_seq_from_buffer
from ariesk.dbs.core_db cimport CoreDB

SEQ_BLOCK_LEN = 10 * 1000
CONTIG_GAP = 2
GENOME_GAP = 20
BUFFER_SIZE = 10 * 1000


cdef class ContigDB(CoreDB):

    def __cinit__(self, conn, ramifier=None, box_side_len=None):
        super().__init__(conn, ramifier=ramifier, box_side_len=box_side_len)
        self.seq_block_len = SEQ_BLOCK_LEN
        self.current_seq_coord = 0
        self.genomes_added = set()
        self.coord_buffer_filled = 0
        self.coord_buffer = [None] * BUFFER_SIZE
        self.contig_cache = {}
        self.centroid_id_cache = {}
        self._build_tables()
        self._build_indices()
        try:
            val = self.conn.execute('SELECT value FROM basics WHERE name=?', ('seq_block_len',))
            self.seq_block_len = int(list(val)[0][0])
            val = self.conn.execute('SELECT value FROM basics WHERE name=?', ('current_seq_coord',))
            self.current_seq_coord = int(list(val)[0][0])
        except IndexError:
            self.seq_block_len = SEQ_BLOCK_LEN
            self.current_seq_coord = 0
            self.conn.executemany(
                'INSERT INTO basics VALUES (?,?)',
                [
                    ('seq_block_len', str(self.seq_block_len)),
                    ('current_seq_coord', str(self.current_seq_coord)),
                ]
            )

    cpdef _build_tables(self):
        self.conn.execute(
            '''CREATE TABLE IF NOT EXISTS contigs (
            seq_coord int, seq BLOB, genome_name text, contig_name text
            )'''
        )
        self.conn.execute(
            '''CREATE TABLE IF NOT EXISTS seq_coords (
                centroid_id int, seq_coord int
            )'''
        )

    cpdef _build_indices(self):
        self.conn.execute('CREATE INDEX IF NOT EXISTS IX_seq_coords_centroid ON seq_coords(centroid_id)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS IX_contigs_coord ON contigs(seq_coord)')

    cpdef _drop_indices(self):
        self.conn.execute('DROP INDEX IF EXISTS IX_seq_coords_centroid')
        self.conn.execute('DROP INDEX IF EXISTS IX_contigs_coord')

    def get_all_contigs(self):
        cdef list out = []
        cdef const npc.uint8_t[:] contig
        for cid, contig, _, _ in self.conn.execute('SELECT * FROM contigs'):
            contig = np.frombuffer(contig, dtype=np.uint8)
            kmer = decode_kmer(contig)
            out.append((cid, kmer))
        return out

    cpdef list get_coords(self, int centroid_id):
        if centroid_id in self.centroid_id_cache:
            return self.centroid_id_cache[centroid_id]
        out = [el[0] for el in self.conn.execute(
            'SELECT seq_coord FROM seq_coords WHERE centroid_id=?', (centroid_id,)
        )]
        self.centroid_id_cache[centroid_id] = out
        return out

    cpdef tuple get_contig(self, int seq_coord):
        if seq_coord in self.contig_cache:
            return self.contig_cache[seq_coord]
        packed = [el for el in self.conn.execute(
            'SELECT seq, genome_name, contig_name FROM contigs WHERE seq_coord=?', (seq_coord,)
        )][0]
        cdef const npc.uint8_t [:] contig = packed[0]
        cdef str genome_name = packed[1]
        cdef str contig_name = packed[2]
        contig = np.frombuffer(contig, dtype=np.uint8)
        cdef tuple out = (genome_name, contig_name, np.copy(contig))
        self.contig_cache[seq_coord] = out
        return out

    cdef add_contig_seq(self, str genome_name, str contig_name, int seq_coord, npc.uint8_t[:] contig_section):
        self.conn.execute(
            'INSERT INTO contigs VALUES (?,?,?,?)',
            (seq_coord, np.array(contig_section, dtype=np.uint8).tobytes(), genome_name, contig_name)
        )

    cdef add_coord_to_centroid(self, int centroid_id, int seq_coord):
        self.coord_buffer[self.coord_buffer_filled] = (centroid_id, seq_coord)
        self.coord_buffer_filled += 1
        if self.coord_buffer_filled == BUFFER_SIZE:
            self._clear_coord_buffer()

    def py_add_contig(self, str genome_name, str contig_name, str contig, int gap=1):
        self.add_contig(genome_name, contig_name, encode_kmer(contig), gap=gap)

    cdef add_contig(self, str genome_name, str contig_name, npc.uint8_t[:] contig, int gap=1):
        cdef int offset = self.current_seq_coord + CONTIG_GAP
        if genome_name not in self.genomes_added:
            offset += GENOME_GAP
            self.genomes_added.add(genome_name)
        cdef npc.uint8_t[:] kmer
        cdef double[:] centroid
        cdef tuple centroid_key
        cdef int i, seq_coord, centroid_id
        for i in range(0, contig.shape[0] - self.ramifier.k + 1, gap):
            kmer = contig[i:i + self.ramifier.k]
            try:
                centroid = np.floor(self.ramifier.c_ramify(kmer) / self.box_side_len, casting='safe')
            except IndexError:
                print()
                print(genome_name)
                print(contig_name)
                print(np.array(kmer))
                print(decode_kmer(kmer))
                raise
            centroid_id = self.add_centroid(centroid)
            seq_coord = offset + (i // self.seq_block_len)
            self.add_coord_to_centroid(centroid_id, seq_coord)
            if i % self.seq_block_len == 0:
                self.add_contig_seq(
                    genome_name, contig_name, seq_coord,
                    contig[i:min(i + self.seq_block_len, contig.shape[0])]
                )

    cdef _clear_coord_buffer(self):
        if self.coord_buffer_filled > 0:
            if self.coord_buffer_filled == BUFFER_SIZE:
                self.conn.executemany(
                    'INSERT INTO seq_coords VALUES (?,?)',
                    self.coord_buffer
                )
            else:
                self.conn.executemany(
                    'INSERT INTO seq_coords VALUES (?,?)',
                    self.coord_buffer[:self.coord_buffer_filled]
                )
            self.coord_buffer_filled = 0

    def commit(self):
        self._clear_coord_buffer()
        self._clear_buffer()
        self.conn.commit()

    @classmethod
    def load_from_filepath(cls, filepath):
        """Return a GridCoverDB."""
        connection = sqlite3.connect(filepath, cached_statements=10 * 1000)
        return ContigDB(connection)

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
            self.add_contig(filename, str(header), seq)
            n_added += 1
            header = NULL
            line = NULL  # I don't understand why this line is necessary but
                         # without it the program throws a strange error: 
                         # `pointer being realloacted was not allocated`
        fclose(cfile)
        return n_added
