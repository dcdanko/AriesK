
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
    cdef public double[:, :] centroid_rfts
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

    def py_coarse_search(self, str kmer, double search_radius, double eps=1.01):
        return self._coarse_search(encode_kmer(kmer), search_radius, eps=eps)

    cdef list _coarse_search(self, npc.uint8_t[:] binary_kmer, double search_radius, double eps=1.01):
        cdef double coarse_search_radius = search_radius + (eps * self.radius)
        cdef double[:] rft = self.ramifier.c_ramify(binary_kmer)
        cdef list centroid_hits = self.tree.query_ball_point(rft, coarse_search_radius)
        return centroid_hits

    cdef npc.uint8_t[:, :] _fine_search(self, npc.uint8_t[:] binary_query_kmer, int center,
            double inner_radius=0.2, inner_metric='needle'):
        cdef npc.uint8_t[:, :] kmers = self.db.get_cluster_members(center)
        cdef npc.uint8_t[:, :] out = np.ndarray((kmers.shape[0], self.ramifier.k), dtype=np.uint8)
        cdef int i, j
        cdef int added = 0
        cdef bint add

        for i in range(kmers.shape[0]):
            add = False
            if inner_metric == 'none':
                add = True
            elif inner_metric == 'needle':
                inner = needle_dist(binary_query_kmer, kmers[i, :], True)
                add = inner < inner_radius
            elif inner_metric == 'hamming':
                inner = hamming_dist(binary_query_kmer, kmers[i, :], True)
                add = inner < inner_radius
            if add:
                for j in range(self.ramifier.k):
                    out[added, j] = kmers[i, j]
                added += 1
        out = out[0:added, :]
        return out

    def py_search(self, str kmer, double search_radius,
        double inner_radius=0.2, double eps=1.01, inner_metric='needle'):
        cdef npc.uint8_t[:, :] hits = np.array(self.search(
            encode_kmer(kmer),
            search_radius,
            inner_radius=inner_radius,
            eps=eps,
            inner_metric=inner_metric
        ), dtype=np.uint8)
        cdef list out = []
        cdef int i

        for i in range(hits.shape[0]):
            out.append(decode_kmer(hits[i, :]))
        return out

    cdef npc.uint8_t[:, :] search(self, npc.uint8_t[:] binary_kmer, double search_radius,
        double inner_radius=0.2, double eps=1.01, inner_metric='needle'):
        cdef npc.uint8_t[:, :] out = np.ndarray((0, self.ramifier.k), dtype=np.uint8)
        cdef list centers = self._coarse_search(binary_kmer, search_radius, eps=eps)
        cdef int i
        cdef npc.uint8_t[:, :] searched
        for center in centers:
            searched = self._fine_search(
                binary_kmer, center, inner_radius=inner_radius, inner_metric=inner_metric
            )
            if searched.shape[0] > 0:
                out = np.append(out, searched, axis=0)
        return out

    def file_search(self,
                    str filepath, str out_filepath, double search_radius,
                    double inner_radius=0.2, double eps=1.01, inner_metric='needle'):
        cdef str kmer
        cdef npc.uint8_t[:, :] results
        cdef int i
        with open(filepath) as f, open(out_filepath, 'w') as o:
            for line in f:
                kmer = line.strip().split(',')[0].split('\t')[0]
                results = self.search(
                    encode_kmer(kmer), search_radius,
                    inner_radius=inner_radius, inner_metric=inner_metric, eps=eps
                )
                for i in range(results.shape[0]):
                    result = decode_kmer(results[i, :])
                    o.write(f'{kmer} {result}\n')

    @classmethod
    def from_filepath(cls, filepath):
        return cls(GridCoverDB.load_from_filepath(filepath))
