
import numpy as np
cimport numpy as npc

from scipy.spatial import cKDTree

from .ram cimport RotatingRamifier
from .db cimport GridCoverDB
from .utils cimport (
    reverse_convert_kmer,
    KmerAddable,
    needle_dist,
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
        self.tree = cKDTree(self.centroid_rfts)

    cpdef _coarse_search(self, str kmer, double search_radius, double eps=1.01):
        cdef double coarse_search_radius = search_radius + (eps * self.radius)
        cdef double [:] rft = self.ramifier.c_ramify(kmer)
        centroid_hits = self.tree.query_ball_point(rft, coarse_search_radius)
        return centroid_hits

    cpdef _fine_search(self, str query_kmer, int center, double inner_radius=0.2, fast_search=False):
        out = []
        for kmer in self.db.get_cluster_members(center):
            if fast_search:
                out.append(kmer)
            else:
                inner = needle_dist(query_kmer, kmer)
                if inner < inner_radius:
                    out.append(kmer)
        return out

    cpdef search(self, str kmer, double search_radius, double inner_radius=0.2, double eps=1.01, fast_search=False):
        out = []
        for center in self._coarse_search(kmer, search_radius, eps=eps):
            out += self._fine_search(kmer, center, inner_radius=inner_radius, fast_search=fast_search)
        return out

    @classmethod
    def from_filepath(cls, filepath):
        return cls(GridCoverDB.load_from_filepath(filepath))
