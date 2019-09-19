
import numpy as np
cimport numpy as npc

from scipy.spatial import cKDTree

from .ram cimport RotatingRamifier
from .db cimport GridCoverDB
from .utils cimport (
    encode_kmer,
    decode_kmer,
    needle_dist,
    hamming_dist,
)


cdef class GridCoverSearcher:
    cdef public GridCoverDB db
    cdef public float radius
    cdef public double [:, :] centroid_rfts
    cdef public RotatingRamifier ramifier

    cdef public object tree

    def __cinit__(self, grid_cover_db):
        self.db = grid_cover_db
        self.radius = (self.db.ramifier.d ** (0.5)) * self.db.box_side_len
        self.ramifier = self.db.ramifier
        self.centroid_rfts = self.db.c_get_centroids()
        for i in range(len(self.centroid_rfts)):
            for j in range(self.db.ramifier.d):
                self.centroid_rfts[i, j] *= self.db.box_side_len
                self.centroid_rfts[i, j] += (self.db.box_side_len / 2)
        self.tree = cKDTree(self.centroid_rfts)

    cpdef _coarse_search(self, str kmer, double search_radius, double eps=1.01):
        cdef double coarse_search_radius = search_radius + (eps * self.radius)
        cdef double [:] rft = self.ramifier.c_ramify(kmer)
        centroid_hits = self.tree.query_ball_point(rft, coarse_search_radius)
        return centroid_hits

    cpdef _fine_search(self,
                       str query_kmer, int center,
                       double inner_radius=0.2, inner_metric='needle'):
        query_submers = {query_kmer[i:i + 5] for i in range(len(query_kmer) - 5 + 1)}
        bloom_filter = self.db.get_bloom_filter(center)
        count = sum([1 if el in bloom_filter else 0 for el in query_submers])
        min_count = len(query_kmer) + 1 - 5 * (1 + int(inner_radius * len(query_kmer)))
        if count < min_count:
            return []

        out = []
        for kmer in self.db.get_cluster_members(center):

            if inner_metric == 'none':
                out.append(kmer)
            elif inner_metric == 'needle':
                inner = needle_dist(query_kmer, kmer, True)
                if inner < inner_radius:
                    out.append(kmer)
            elif inner_metric == 'hamming':
                inner = hamming_dist(query_kmer, kmer, True)
                if inner < inner_radius:
                    out.append(kmer)
        return out

    cpdef search(self,
                 str kmer, double search_radius,
                 double inner_radius=0.2, double eps=1.01, inner_metric='needle'):
        out = []
        for center in self._coarse_search(kmer, search_radius, eps=eps):
            out += self._fine_search(
                kmer, center,
                inner_radius=inner_radius, inner_metric=inner_metric
            )
        return out

    def file_search(self,
                    str filepath, str out_filepath, double search_radius,
                    double inner_radius=0.2, double eps=1.01, inner_metric='needle'):
        with open(filepath) as f, open(out_filepath, 'w') as o:
            for line in f:
                kmer = line.strip().split(',')[0].split('\t')[0]
                results = self.search(
                    kmer, search_radius,
                    inner_radius=inner_radius, inner_metric=inner_metric, eps=eps
                )
                for result in results:
                    o.write(f'{kmer} {result}\n')

    @classmethod
    def from_filepath(cls, filepath):
        return cls(GridCoverDB.load_from_filepath(filepath))
