
import numpy as np
cimport numpy as npc

from scipy.spatial import cKDTree

from libc.math cimport ceil
from .ram cimport RotatingRamifier
from .db cimport GridCoverDB
from .cluster cimport Cluster
from .utils cimport (
    encode_kmer,
    decode_kmer,
    needle_fast,
    hamming_dist,
)


cdef class GridCoverSearcher:
    cdef public GridCoverDB db
    cdef public float radius
    cdef public double[:, :] centroid_rfts
    cdef public RotatingRamifier ramifier
    cdef public bint logging
    cdef public object logger

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
        self.logging = False

    def add_logger(self, logger):
        self.logging = True
        self.logger = logger

    def py_coarse_search(self, str kmer, double search_radius, double eps=1.01):
        return self._coarse_search(encode_kmer(kmer), search_radius, eps=eps)

    cdef list _coarse_search(self, npc.uint8_t[:] binary_kmer, double search_radius, double eps=1.01):
        cdef double coarse_search_radius = search_radius + (eps * self.radius)
        cdef double[:] rft = self.ramifier.c_ramify(binary_kmer)
        cdef list centroid_hits = self.tree.query_ball_point(rft, coarse_search_radius)
        return centroid_hits

    cdef npc.uint8_t[:, :] _fine_search(self, npc.uint8_t[:] query_kmer, Cluster cluster,
                                        double inner_radius=0.2, inner_metric='needle'):
        """Search a single cluster and return all members within inner_radius."""
        cdef npc.uint8_t[:, :] out = np.ndarray(
            (cluster.seqs.shape[0], self.ramifier.k),
            dtype=np.uint8
        )
        cdef double[:, :] score = np.zeros((self.ramifier.k + 1, self.ramifier.k + 1))
        cdef int i, j
        cdef int added = 0
        for i in range(cluster.seqs.shape[0]):
            if inner_metric == 'needle':
                inner = needle_fast(query_kmer, cluster.seqs[i, :], True, score)
            elif inner_metric == 'hamming':
                inner = hamming_dist(query_kmer, cluster.seqs[i, :], True)
            if inner_metric == 'none' or inner < inner_radius:
                for j in range(self.ramifier.k):
                    out[added, j] = cluster.seqs[i, j]
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
        if self.logging:
            self.logger(f'Starting search.')
        cdef npc.uint8_t[:, :] out = np.ndarray((0, self.ramifier.k), dtype=np.uint8)
        cdef float start_time_1, elapsed_time_1, start_time_2, elapsed_time_2, start_time_3, elapsed_time_3
        cdef list centers = self._coarse_search(binary_kmer, search_radius, eps=eps)
        if self.logging:
            self.logger(f'Coarse search complete. {len(centers)} clusters.')

        cdef int i = 0
        cdef npc.uint8_t[:, :] searched
        cdef Cluster cluster
        cdef npc.uint8_t[:] filtered_centers = np.zeros((len(centers,)), dtype=np.uint8)
        cdef int max_misses = <int> ceil(inner_radius * binary_kmer.shape[0])
        cdef int n_points_original = 0
        for center in centers:
            cluster = self.db.get_cluster(center)
            if self.logging:
                n_points_original += cluster.seqs.shape[0]
            if inner_metric == 'none':
                filtered_centers[i] = 1
            elif cluster.test_membership(binary_kmer, max_misses):
                filtered_centers[i] = 1
            i += 1
        if self.logging:
            self.logger(f'Cluster filtering complete. {sum(filtered_centers)} clusters remaining.')
            self.logger(f'Allowing up to {max_misses}.')
 
        i = -1
        for center in centers:
            i += 1
            if filtered_centers[i] == 1:
                cluster = self.db.get_cluster(center)
                searched = self._fine_search(
                    binary_kmer, cluster,
                    inner_radius=inner_radius, inner_metric=inner_metric
                )
                if searched.shape[0] > 0:
                    out = np.append(out, searched, axis=0)
        if self.logging:
            self.logger(f'Fine search complete. {out.shape[0]} candidates passed.')
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
