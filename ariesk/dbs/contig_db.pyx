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

    def __cinit__(self, conn, ramifier=None, box_side_len=None):
        super().__init__(conn, ramifier=ramifier, box_side_len=box_side_len)
        self.seq_block_len = SEQ_BLOCK_LEN
        self.current_seq_coord = 0
        self.genomes_added = set()
        self.coord_buffer_filled = 0
        self.coord_buffer = [None] * BUFFER_SIZE
        self.contig_cache = {}
        self.contig_kmer_cache = {}
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
            seq_coord int, seq BLOB, genome_name text, contig_name text, contig_coord int
            )'''
        )
        self.conn.execute(
            '''CREATE TABLE IF NOT EXISTS seq_coords (
                centroid_id int, seq_coord int,
                FOREIGN KEY (centroid_id) REFERENCES centroids(centroid_id),
                FOREIGN KEY (seq_coord)   REFERENCES contigs(seq_coord)
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
        for cid, contig, genome_name, contig_name, contig_coord in self.conn.execute('SELECT * FROM contigs'):
            contig = np.frombuffer(contig, dtype=np.uint8)
            kmer = decode_kmer(contig)
            out.append((cid, kmer, genome_name, contig_name, contig_coord))
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
            'SELECT seq, genome_name, contig_name, contig_coord FROM contigs WHERE seq_coord=?', (seq_coord,)
        )][0]
        cdef const npc.uint8_t [:] contig = packed[0]
        cdef str genome_name = packed[1]
        cdef str contig_name = packed[2]
        cdef int contig_coord = packed[3]
        contig = np.frombuffer(contig, dtype=np.uint8)
        cdef tuple out = (genome_name, contig_name, contig_coord, np.copy(contig))
        self.contig_cache[seq_coord] = out
        return out

    cdef npc.uint32_t[:, :] get_contig_kmers(self, int seq_coord, int k):
        if seq_coord in self.contig_kmer_cache:
            return self.contig_kmer_cache[seq_coord]
        _, _, _, contig = self.get_contig(seq_coord)
        cdef npc.uint32_t[:, :] t_kmers = get_target_kmers(contig, k)
        self.contig_kmer_cache[seq_coord] = t_kmers
        return t_kmers

    cdef add_contig_seq(self, str genome_name, str contig_name, int seq_coord, int contig_coord, npc.uint8_t[:] contig_section):
        self.conn.execute(
            'INSERT INTO contigs VALUES (?,?,?,?,?)',
            (seq_coord, np.array(contig_section, dtype=np.uint8).tobytes(), genome_name, contig_name, contig_coord)
        )

    cdef add_coord_to_centroid(self, int centroid_id, int seq_coord):
        self.coord_buffer[self.coord_buffer_filled] = (centroid_id, seq_coord)
        self.coord_buffer_filled += 1
        if self.coord_buffer_filled == BUFFER_SIZE:
            self._clear_coord_buffer()

    def py_add_contig(self, str genome_name, str contig_name, str contig, int gap=1):
        self.add_contig(genome_name, contig_name, encode_kmer(contig), gap=gap)

    cdef add_contig(self, str genome_name, str contig_name, npc.uint8_t[:] contig, int gap=1):
        if genome_name not in self.genomes_added:
            self.current_seq_coord += GENOME_GAP
            self.genomes_added.add(genome_name)
        self.current_seq_coord += CONTIG_GAP
        cdef npc.uint8_t[:] kmer
        cdef double[:] centroid
        cdef tuple centroid_key
        cdef int i, seq_coord, centroid_id
        for i in range(0, contig.shape[0] - self.ramifier.k + 1, gap):
            kmer = contig[i:i + self.ramifier.k]
            centroid = np.floor(self.ramifier.c_ramify(kmer) / self.box_side_len, casting='safe')
            centroid_id = self.add_centroid(centroid)
            seq_coord = self.current_seq_coord + (i // self.seq_block_len)
            self.add_coord_to_centroid(centroid_id, seq_coord)
            if i % self.seq_block_len == 0:
                self.add_contig_seq(
                    genome_name, contig_name, seq_coord, i,
                    contig[i:min(i + self.seq_block_len, contig.shape[0])]
                )
        self.current_seq_coord += int(ceil(contig.shape[0] / self.seq_block_len))

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
        self.conn.execute(
            f'UPDATE basics SET value="{self.current_seq_coord}" WHERE name = "current_seq_coord";'
        )

    def commit(self):
        self._clear_coord_buffer()
        self._clear_buffer()
        self.conn.commit()

    @classmethod
    def from_predb(cls, filepath, predb, box_side_len):
        db = cls(
            sqlite3.connect(filepath),
            ramifier=predb.ramifier,
            box_side_len=box_side_len
        )
        db._drop_indices()
        db.add_from_predb(predb)
        return db

    def add_from_predb(self, predb):
        cdef dict raw_rfts = {}
        # cdef npc.uint8_t[:] raw_rft
        cdef const double[:] rft
        cdef double[:] centroid = np.ndarray((self.ramifier.d,))
        cdef int centroid_id
        cdef int contig_id
        for raw_rft, contig_id in predb.conn.execute('SELECT * FROM rfts'):
            rft = np.frombuffer(raw_rft, dtype=float, count=self.ramifier.d)
            for i in range(self.ramifier.d):
                centroid[i] = floor(rft[i] / self.box_side_len)
            centroid_id = self.add_centroid(centroid)
            try:
                raw_rfts[contig_id].append(centroid_id)
            except KeyError:
                raw_rfts[contig_id] = [centroid_id]
        for contig_id, seq, genome_name, contig_name, contig_start in predb.conn.execute('SELECT * FROM contigs'):
            self.add_contig_from_predb(
                raw_rfts, contig_id, seq, genome_name, contig_name, contig_start
            )

    def add_contig_from_predb(self, raw_rfts, contig_id, seq, genome_name, contig_name, contig_start):
        self.current_seq_coord += CONTIG_GAP
        if genome_name not in self.genomes_added:
            self.current_seq_coord += GENOME_GAP
            self.genomes_added.add(genome_name)
        self.conn.execute(
            'INSERT INTO contigs VALUES (?,?,?,?,?)',
            (self.current_seq_coord, seq, genome_name, contig_name, contig_start)
        )
        cdef int centroid_id
        for centroid_id in raw_rfts[contig_id]:
            self.add_coord_to_centroid(centroid_id, self.current_seq_coord)

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
            'INSERT INTO seq_coords VALUES (?,?)',
            (
                (other_centroid_remap[cid], self.current_seq_coord + coord)
                for cid, coord in other.conn.execute('SELECT * FROM seq_coords')
            )
        )
        self.conn.executemany(
            'INSERT INTO contigs VALUES (?,?,?,?,?)',
            (
                (self.current_seq_coord + coord, seq, gname, cname, ccoord)
                for coord, seq, gname, cname, ccoord in other.conn.execute('SELECT * FROM contigs')
            )
        )
        self.current_seq_coord += other.current_seq_coord
        self.commit()
        if rebuild_indices:
            self._build_indices()

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
            self.add_contig(filename, str(header).strip()[1:], seq)
            n_added += 1
            header = NULL
            line = NULL  # I don't understand why this line is necessary but
                         # without it the program throws a strange error: 
                         # `pointer being realloacted was not allocated`
        fclose(cfile)
        return n_added
