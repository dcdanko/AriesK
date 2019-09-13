
import numpy as np
cimport numpy as npc
from scipy.spatial import cKDTree as KDTree

from .ram cimport RotatingRamifier
from .utils cimport reverse_convert_kmer, KmerAddable


cdef double needle_dist(k1, k2):
    cdef double [:, :] score = np.zeros((len(k1) + 1, len(k2) + 1))
    cdef double match_score = 0
    cdef double mismatch_penalty = 1.5
    cdef double gap_penalty = 1.6
    for i in range(len(k1) + 1):
        score[i][0] = gap_penalty * i
    for j in range(len(k2) + 1):
        score[0][j] = gap_penalty * j

    def _match_score(b1, b2):
        return match_score if b1 == b2 else mismatch_penalty

    for i in range(1, len(k1) + 1):
        for j in range(1, len(k2) + 1):
            match = score[i - 1][j - 1] + _match_score(k1[i - 1], k2[j - 1])
            delete = score[i - 1][j] + gap_penalty
            insert = score[i][j - 1] + gap_penalty
            score[i][j] = min(match, delete, insert)
    final_score = score[len(k1)][len(k2)]

    return final_score / len(k1)


cdef class GridCoverSearcher:
    cdef public long [:] kmers
    cdef public float box_side_len, radius
    cdef public double [:, :] centroid_rfts
    cdef public object clusters
    cdef public RotatingRamifier ramifier
    cdef public KDTree tree

    def __cinit__(self, box_side_len, ramifier, kmers, clusters):
        self.box_side_len = box_side_len
        self.radius = (ramifier.d ** (1/2)) * box_side_len
        self.ramifier = ramifier
        self.kmers = kmers
        self.clusters = clusters

        self.centroid_rfts = npc.ndarray((len(clusters), self.ramifier.d))
        for i, centroid in enumerate(clusters.keys()):
            kmer = reverse_convert_kmer(kmers[centroid])
            rft = ramifier.c_ramify(kmer)
            self.centroid_rfts[i] = rft
        self.tree = KDTree(self.centroid_rfts)

    cdef int [:] _coarse_search(str kmer, double search_radius, double eps=0.5):
        double coarse_search_radius = search_radius + (eps * self.radius)
        double [:] rft = self.ramifier.c_ramify(kemr)
        int [:] centroid_hits = self.tree.query_ball_point(rft, coarse_search_radius, eps=0.1)
        return centroid_hits

    cdef _fine_search(str query_kmer, int cluster):
        out = []
        for member_index in self.clusters[cluster]:
            kmer = reverse_convert_kmer(self.kmers[member_index])
            needle = needle_dist(query_kmer, kmer)
            if needle < 0.2:
                out.append(kmer)
        return out

    cpdef search(str kmer, double search_radius, double eps=0.5):
        out = []
        for center in self._coarse_search(kmer, search_radius, eps=eps):
            out += self._fine_search(kmer, center)
        return out

    @classmethod
    def from_dict(cls, saved):
        ramifier = RotatingRamifier.from_dict(saved['ramifier'])
        kmers = np.array(saved_dict['kmer'], dtype=float)
        clusters = {clust['centroid']: clust['members'] for clust in saved['clusters']}
        searcher = cls(saved['radius'], ramifier, kmers)

        return searcher
