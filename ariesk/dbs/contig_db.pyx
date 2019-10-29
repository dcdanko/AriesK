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
from libc.math cimport floor

from ariesk.utils.kmers cimport encode_kmer, decode_kmer, encode_seq_from_buffer
from ariesk.dbs.core_db cimport CoreDB
from ariesk.seed_align cimport get_target_kmers


SEQ_BLOCK_LEN = 10 * 1000
CONTIG_GAP = 2
GENOME_GAP = 20
BUFFER_SIZE = 10 * 1000


cdef class ContigDB(CoreDB):

    def __cinit__(self, conn, ramifier=None, box_side_len=None, logger=None):
        super().__init__(conn, ramifier=ramifier, box_side_len=box_side_len)
        self.seq_block_len = SEQ_BLOCK_LEN
        self.current_seq_coord = 0
        self.genomes_added = set()
        self.contig_cache = {}
        self.contig_kmer_cache = {}
        self.centroid_id_cache = {}
        self._build_tables()
        self._build_indices()
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
        if self.logging:
            logger('Loaded Contig Database.')

    cpdef _build_tables(self):
        if self.logging:
            self.logger('Building SQL tables...')
        self.conn.execute(
            '''CREATE TABLE IF NOT EXISTS contigs (
            genome_name text,
            contig_name text,
            centroid_id int,
            start_coord int,
            end_coord int,
            seq BLOB
            )'''
        )

    cpdef _build_indices(self):
        if self.logging:
            self.logger('Building SQL indices...')
        self.conn.execute('CREATE INDEX IF NOT EXISTS IX_contigs_centroid ON contigs(centroid_id)')

    cpdef _drop_indices(self):
        self.conn.execute('DROP INDEX IF EXISTS IX_contigs_centroid')

    def get_all_contigs(self):
        cdef list out = []
        cdef const npc.uint8_t[:] contig
        for vals in self.conn.execute('SELECT * FROM contigs'):
            genome_name, contig_name, centroid_id, start_coord, end_coord, contig = vals
            contig = np.frombuffer(contig, dtype=np.uint8)
            kmer = decode_kmer(contig)
            out.append((genome_name, contig_name, centroid_id, start_coord, end_coord, contig))
        return out

    cpdef tuple get_contigs(self, int centroid_id):
        if centroid_id in self.contig_cache:
            return self.contig_cache[centroid_id]
        cmd = '''
            SELECT genome_name, contig_name, start_coord, end_coord, seq
            FROM contigs
            WHERE centroid_id=?
        '''
        cursor = self.conn.execute(cmd, (centroid_id,))
        cdef list out = []
        for genome_name, contig_name, start_coord, end_coord, seq_blob in cursor:
            contig = np.frombuffer(seq_blob, dtype=np.uint8)
            out.append((genome_name, contig_name, start_coord, end_coord, np.copy(contig)))
        self.contig_cache[centroid_id] = out
        return out

    cdef add_contig_seq(self,
                        str genome_name, str contig_name, int centroid_id,
                        int start_coord, int end_coord, npc.uint8_t[:] contig_section):
        self.conn.execute(
            'INSERT INTO contigs VALUES (?,?,?,?,?,?)',
            (
                genome_name, contig_name, centroid_id,
                start_coord, end_coord,
                np.array(contig_section, dtype=np.uint8).tobytes(),
            )
        )

    def py_add_contig(self, str genome_name, str contig_name, str contig, int gap=1):
        self.add_contig(genome_name, contig_name, encode_kmer(contig), gap=gap)

    cdef add_contig(self, str genome_name, str contig_name, npc.uint8_t[:] contig, int gap=1):
        cdef int i
        cdef int section_end = 0
        cdef int section_start = 0
        cdef int current_centroid_id = -1
        for i in range(0, contig.shape[0] - self.ramifier.k + 1, gap):
            kmer = contig[i:i + self.ramifier.k]
            centroid = np.floor(self.ramifier.c_ramify(kmer) / self.box_side_len, casting='safe')
            centroid_id = self.add_centroid(centroid)
            if current_centroid_id < 0:
                current_centroid_id = centroid_id
            if centroid_id != current_centroid_id:
                self.add_contig_seq(
                    genome_name, contig_name, current_centroid_id,
                    section_start, section_end, contig[section_start:section_end]

                )
                section_start = i
                current_centroid_id = centroid_id
            section_end = i + self.ramifier.k
        if section_end > section_start:
            self.add_contig_seq(
                genome_name, contig_name, current_centroid_id,
                section_start, section_end, contig[section_start:section_end]

            )

    def commit(self):
        self._clear_buffer()
        self.conn.commit()

    def load_other(self, other, rebuild_indices=True):
        cdef dict other_centroid_remap = {}
        for cid, centroid_blob in other.conn.execute('SELECT * FROM centroids'):
            if centroid_blob in self.centroid_cache:
                other_centroid_remap[cid] = self.centroid_cache[centroid_blob]
            else:
                other_centroid_remap[cid] = len(self.centroid_cache)
        self._drop_indices()
        self.conn.executemany(
            'INSERT INTO centroids VALUES (?,?)',
            (
                (other_centroid_remap[cid], centroid_blob)
                for cid, centroid_blob in other.conn.execute('SELECT * FROM centroids')
            )
        )
        self.conn.executemany(
            'INSERT INTO contigs VALUES (?,?,?,?,?,?)',
            (
                (genome_name, contig_name, other_centroid_remap[cid], start_coord, end_coord, contig)
                for genome_name, contig_name, cid, start_coord, end_coord, contig
                in other.conn.execute('SELECT * FROM contigs')
            )
        )
        self.commit()
        if rebuild_indices:
            self._build_indices()

    @classmethod
    def load_from_filepath(cls, filepath, logger=None):
        """Return a GridCoverDB."""
        if logger is not None:
            logger('Connecting to SQL database...')
        connection = sqlite3.connect(filepath, cached_statements=10 * 1000)
        if logger is not None:
            logger('Loading resources from database...')
        return ContigDB(connection, logger=logger)

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
            self.add_contig(filename.strip(), str(header).strip()[1:], seq)
            n_added += 1
            header = NULL
            line = NULL  # I don't understand why this line is necessary but
                         # without it the program throws a strange error: 
                         # `pointer being realloacted was not allocated`
        fclose(cfile)
        return n_added
