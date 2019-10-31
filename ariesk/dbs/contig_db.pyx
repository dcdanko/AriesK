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
        self.seq_cache = {}
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
            contig_name text,
            centroid_id int,
            start_coord int,
            end_coord int
            )'''
        )
        self.conn.execute(
            '''CREATE TABLE IF NOT EXISTS nucl_seqs (
            contig_name text,
            seq BLOB
            )'''
        )

    cpdef _build_indices(self):
        if self.logging:
            self.logger('Building SQL indices...')
        self.conn.execute('CREATE INDEX IF NOT EXISTS IX_contigs_centroid ON contigs(centroid_id)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS IX_nucl_seqs_genome ON nucl_seqs(contig_name)')

    cpdef _drop_indices(self):
        self.conn.execute('DROP INDEX IF EXISTS IX_contigs_centroid')
        self.conn.execute('DROP INDEX IF EXISTS IX_nucl_seqs_genome')

    def get_all_contigs(self):
        cdef list out = []
        for vals in self.conn.execute('SELECT * FROM contigs'):
            contig_name, centroid_id, start_coord, end_coord = vals
            out.append((contig_name, centroid_id, start_coord, end_coord))
        return out

    cpdef list get_contigs(self, int centroid_id):
        if centroid_id in self.contig_cache:
            return self.contig_cache[centroid_id]
        cmd = '''
            SELECT contig_name, start_coord, end_coord
            FROM contigs
            WHERE centroid_id=?
        '''
        cursor = self.conn.execute(cmd, (centroid_id,))
        cdef list out = []
        for contig_name, start_coord, end_coord in cursor:
            out.append((contig_name, start_coord, end_coord))
        self.contig_cache[centroid_id] = out
        return out

    cdef npc.uint8_t[:] get_seq(self, str contig_name, int start_coord, int end_coord):
        cdef npc.uint8_t[:] contig
        cdef const npc.uint8_t[:] seq_blob
        if contig_name in self.seq_cache:
            contig = self.seq_cache[contig_name]
        else:
            cmd = '''
                SELECT seq
                FROM nucl_seqs
                WHERE contig_name=?
            '''
            seq_blob = self.conn.execute(cmd, (contig_name,)).fetchone()[0]
            contig = np.copy(np.frombuffer(seq_blob, dtype=np.uint8))
            self.seq_cache[contig_name] = contig
        return contig[max(start_coord, 0):min(end_coord, contig.shape[0])]

    cdef add_contig_seq(self,
                        str contig_name, int centroid_id,
                        int start_coord, int end_coord):
        self.conn.execute(
            'INSERT INTO contigs VALUES (?,?,?,?)',
            (
                contig_name, centroid_id,
                start_coord, end_coord,
            )
        )

    def py_add_contig(self, str contig_name, str contig, int gap=1):
        self.add_contig(contig_name, encode_kmer(contig), gap=gap)

    cdef add_contig(self, str contig_name, npc.uint8_t[:] contig, int gap=1):
        self.conn.execute(
            'INSERT INTO nucl_seqs VALUES (?,?)',
            (
                contig_name,
                np.array(contig, dtype=np.uint8).tobytes(),
            )
        )
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
                    contig_name, current_centroid_id,
                    section_start, section_end

                )
                section_start = i
                current_centroid_id = centroid_id
            section_end = i + self.ramifier.k
        if section_end > section_start:
            self.add_contig_seq(
                contig_name, current_centroid_id,
                section_start, section_end
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
            'INSERT INTO contigs VALUES (?,?,?,?)',
            (
                (contig_name, other_centroid_remap[cid], start_coord, end_coord)
                for contig_name, cid, start_coord, end_coord
                in other.conn.execute('SELECT * FROM contigs')
            )
        )
        self.conn.executemany(
            'INSERT INTO nucl_seqs VALUES (?,?)',
            (
                (contig_name, contig)
                for contig_name, contig
                in other.conn.execute('SELECT * FROM nucl_seqs')
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
            self.add_contig(filename.strip() + '___' + str(header).strip()[2:-3].strip(), seq)
            n_added += 1
            header = NULL
            line = NULL  # I don't understand why this line is necessary but
                         # without it the program throws a strange error: 
                         # `pointer being realloacted was not allocated`
        fclose(cfile)
        return n_added
