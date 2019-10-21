# cython: profile=True
# cython: linetrace=True
# cython: language_level=3

cimport numpy as npc
import numpy as np
from libc.stdio cimport *
from posix.stdio cimport * # FILE, fopen, fclose
from libc.stdlib cimport malloc, free
from scipy.spatial import cKDTree
# from scipy.spatial cimport cKDTree as cKDTree_t
# from ariesk.ckdtree cimport cKDTree
#from skbio.alignment import StripedSmithWaterman
# from skbio.alignment cimport StripedSmithWaterman as StripedSmithWaterman_t
# from ariesk.lib_ssw cimport *
from ariesk.ssw cimport StripedSmithWaterman
from ariesk.dbs.contig_db cimport ContigDB
from ariesk.utils.kmers cimport (
    needle_dist,
    hamming_dist,
    encode_kmer,
    decode_kmer,
    encode_seq_from_buffer,
)
from ariesk.seed_align cimport seed_and_extend, get_query_kmers

cdef npc.uint8_t K_LEN = 7
cdef npc.uint8_t K_GAP = 3


cdef class ContigSearcher:
    cdef public ContigDB db
    cdef public double[:, :] centroid_rfts
    cdef public object tree
    cdef public object logger
    cdef public bint logging
    cdef public float radius

    def __cinit__(self, contig_db, logger=None):
        if self.logger is not None:
            self.logging = True
            self.logger('Loading database...')
        self.db = contig_db
        self.centroid_rfts = self.db.c_get_centroids()
        if self.logging:
            self.logger(f'Retrieved centroids from database.')
        for i in range(self.centroid_rfts.shape[0]):
            for j in range(self.db.ramifier.d):
                self.centroid_rfts[i, j] *= self.db.box_side_len
                self.centroid_rfts[i, j] += (self.db.box_side_len / 2)
        self.tree = cKDTree(self.centroid_rfts)
        if self.logging:
            self.logger(f'Built search tree.')
        self.radius = (self.db.ramifier.d ** (0.5)) * self.db.box_side_len
        self.logger = logger
        self.logging = False
        if self.logging:
            self.logger(f'Searcher loaded. Radius {self.radius}, num. centers {self.centroid_rfts.shape[0]}')

    def py_search(self, str query, double coarse_radius, double kmer_fraction, double identity=0.5):
        return [
            el
            for el in self.search(
                encode_kmer(query), coarse_radius, kmer_fraction, identity
            )
        ]

    cdef list search(self, npc.uint8_t[:] query, double coarse_radius, double kmer_fraction, double identity):
        if self.logging:
            self.logger(f'Starting query. Coarse radius {coarse_radius}, k-mer fraction {kmer_fraction}')
        cdef npc.uint32_t n_kmers = (query.shape[0] - self.db.ramifier.k + 1) // (self.db.ramifier.k // 2)
        cdef dict counts = self.coarse_search(n_kmers, query, coarse_radius)
        if self.logging:
            self.logger(f'Coarse search complete. {len(counts)} candidates.')
        cdef npc.uint64_t[:, :] q_kmers = get_query_kmers(query, K_LEN, K_GAP)
        out = self.fine_search(query, q_kmers, K_LEN, n_kmers, counts, kmer_fraction, identity)
        if self.logging:
            self.logger(f'Fine search complete. {len(out)} passed.')
        return out

    cdef list fine_search(self,
                          npc.uint8_t[:] query, npc.uint64_t[:, :] q_kmers,
                          npc.uint8_t k, npc.uint32_t n_kmers, dict counts,
                          double kmer_fraction, double identity):
        cdef list out = []
        cdef double aln_score
        cdef npc.uint64_t[:, :] intervals
        cdef npc.uint32_t[:, :] t_kmers
        for seq_coord, count in counts.items():
            if count >= (n_kmers * kmer_fraction):
                genome_name, contig_name, contig_coord, contig = self.db.get_contig(seq_coord)
                t_kmers = self.db.get_contig_kmers(seq_coord, k)
                intervals = seed_and_extend(query, contig, q_kmers, t_kmers, k)
                if intervals.shape[0] > 0:
                    out.append((genome_name, contig_name, contig_coord, intervals))
        return out

    cdef dict coarse_search(self, int n_kmers, npc.uint8_t[:] query, double coarse_radius):
        cdef double[:] rft
        cdef double[:, :] rfts = np.ndarray(
            (n_kmers, self.db.ramifier.d)
        )
        cdef int i, j, k_start, k_end
        for i in range(n_kmers):
            k_start = i * (self.db.ramifier.k // 2)
            k_end = k_start + self.db.ramifier.k
            rft = self.db.ramifier.c_ramify(query[k_start:k_end])
            for j in range(self.db.ramifier.d):
                rfts[i, j] = rft[j]
        cdef object query_tree = cKDTree(rfts)
        cdef list centroid_hits
        cdef int hit
        cdef dict counts = {}
        for centroid_hits in query_tree.query_ball_tree(self.tree, coarse_radius + self.radius):
            for hit in centroid_hits:
                for seq_coord in self.db.get_coords(hit):
                    counts[seq_coord] = 1 + counts.get(seq_coord, 0)
        return counts

    def search_contigs_from_fasta(self, str filename, double coarse_radius, double kmer_fraction, double identity):
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
        cdef dict out = {}
        while True:
            getline(&header, &l, cfile)
            read = getdelim(&line, &l, b'>', cfile)
            if read == -1: break
            seq = encode_seq_from_buffer(line, l)
            out[decode_kmer(seq)] = self.search(seq, coarse_radius, kmer_fraction, identity)
            n_added += 1
            header = NULL
            line = NULL  # I don't understand why this line is necessary but
                         # without it the program throws a strange error: 
                         # `pointer being realloacted was not allocated`
        fclose(cfile)
        return out

    @classmethod
    def from_filepath(cls, filepath, logger=None):
        return cls(ContigDB.load_from_filepath(filepath), logger=logger)
