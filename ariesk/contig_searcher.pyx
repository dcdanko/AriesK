# cython: profile=True
# cython: linetrace=True
# cython: language_level=3

cimport numpy as npc
import numpy as np

from scipy.spatial import cKDTree
# from scipy.spatial cimport cKDTree as cKDTree_t
from skbio.alignment import StripedSmithWaterman
# from skbio.alignment cimport StripedSmithWaterman as StripedSmithWaterman_t

from ariesk.dbs.contig_db cimport ContigDB
from ariesk.utils.kmers cimport (
    needle_dist,
    hamming_dist,
    encode_kmer,
    decode_kmer,
)


cdef class ContigSearcher:
    cdef public ContigDB db
    cdef public double[:, :] centroid_rfts
    cdef public object tree
    cdef public object logger
    cdef public bint logging
    cdef public float radius

    def __cinit__(self, contig_db, logger=None):
        self.db = contig_db
        self.centroid_rfts = self.db.c_get_centroids()
        self.tree = cKDTree(self.centroid_rfts)
        self.radius = (self.db.ramifier.d ** (0.5)) * self.db.box_side_len
        self.logger = logger
        self.logging = False
        if self.logger is not None:
            self.logging = True

    def py_search(self, str query, double coarse_radius, double kmer_fraction):
        return [
            (el[0], decode_kmer(el[1]))
            for el in self.search(
                encode_kmer(query), coarse_radius, kmer_fraction
            )
        ]

    cdef list search(self, npc.uint8_t[:] query, double coarse_radius, double kmer_fraction):
        if self.logging:
            self.logger(f'Starting query. Coarse radius {coarse_radius}, k-mer fraction {kmer_fraction}')
        cdef int n_kmers = (query.shape[0] - self.db.ramifier.k + 1) // (self.db.ramifier.k // 2)
        cdef dict counts = self.coarse_search(n_kmers, query, coarse_radius)
        if self.logging:
            self.logger(f'Coarse search complete. {len(counts)} candidates.')
        cdef object water = StripedSmithWaterman(decode_kmer(query), score_only=True)
        cdef list out = []
        for seq_coord, count in counts.items():
            if count > (n_kmers * kmer_fraction):
                contig = self.db.get_contig(seq_coord)
                aln = water(decode_kmer(contig))
                out.append((aln.optimal_alignment_score, contig))
        if self.logging:
            self.logger(f'Fine search complete. {len(out)} passed.')
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
        for centroid_hits in self.tree.query_ball_tree(query_tree, coarse_radius + self.radius):
            for hit in centroid_hits:
                for seq_coord in self.db.get_coords(hit):
                    counts[seq_coord] = 1 + counts.get(seq_coord, 0)
        return counts


    @classmethod
    def from_filepath(cls, filepath, logger=None):
        return cls(ContigDB.load_from_filepath(filepath), logger=logger)
