# cython: profile=True
# cython: linetrace=True
# cython: language_level=3

import numpy as np
cimport numpy as npc
from scipy.spatial import cKDTree

from libc.math cimport ceil
from ariesk.ram cimport RotatingRamifier
from ariesk.dbs.kmer_db cimport GridCoverDB
from ariesk.cluster cimport Cluster
from ariesk.utils.bloom_filter cimport fnva, fast_modulo
from ariesk.utils.kmers cimport (
    encode_kmer,
    decode_kmer,
    needle_fast,
    bounded_needle_fast,
    hamming_dist,
)


cdef class GridCoverSearcher:
    cdef public GridCoverDB db
    cdef public float radius
    cdef public double[:, :] centroid_rfts
    cdef public RotatingRamifier ramifier
    cdef public bint logging
    cdef public object logger
    cdef public int sub_k
    cdef public int n_hashes
    cdef public int array_size
    cdef public npc.uint64_t[:, :] hash_functions

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
        self.sub_k = self.db.sub_k
        self.n_hashes = self.db.n_hashes
        self.array_size = self.db.array_size
        self.hash_functions = self.db.hash_functions

    def add_logger(self, logger):
        self.logging = True
        self.logger = logger

    def py_coarse_search(self, str kmer, double search_radius, double eps=1.01):
        return self._coarse_search(encode_kmer(kmer), search_radius, eps=eps)

    def py_search(self, str kmer, double search_radius, max_filter_misses=None,
        double inner_radius=0.2, double eps=1.01, inner_metric='needle'):
        if max_filter_misses is None:
            max_filter_misses = int(ceil(inner_radius * len(kmer)))
        cdef npc.uint8_t[:, :] hits = np.array(self.search(
            encode_kmer(kmer),
            search_radius,
            max_filter_misses,
            inner_radius=inner_radius,
            eps=eps,
            inner_metric=inner_metric
        ), dtype=np.uint8)
        cdef list out = []
        cdef int i

        for i in range(hits.shape[0]):
            out.append(decode_kmer(hits[i, :]))
        return out

    def file_search(self,
                    str filepath, str out_filepath, double search_radius, max_filter_misses=None,
                    double inner_radius=0.2, double eps=1.01, inner_metric='needle'):
        if max_filter_misses is None:
            max_filter_misses = int(ceil(inner_radius * self.ramifier.k))
        cdef str kmer
        cdef npc.uint8_t[:, :] results
        cdef int i
        with open(filepath) as f, open(out_filepath, 'w') as o:
            for line in f:
                kmer = line.strip().split(',')[0].split('\t')[0]
                results = self.search(
                    encode_kmer(kmer), search_radius,
                    max_filter_misses,
                    inner_radius=inner_radius, inner_metric=inner_metric, eps=eps
                )
                for i in range(results.shape[0]):
                    result = decode_kmer(results[i, :])
                    o.write(f'{kmer} {result}\n')

    @classmethod
    def from_filepath(cls, filepath):
        return cls(GridCoverDB.load_from_filepath(filepath))

    cdef list _coarse_search(
        self,
        npc.uint8_t[:] binary_kmer,
        double search_radius,
        double eps=1.01
    ):
        """Return a list of the cluster indices which are within <radius> of the query."""
        cdef double coarse_search_radius = search_radius + (eps * self.radius)
        cdef double[:] rft = self.ramifier.c_ramify(binary_kmer)
        cdef list centroid_hits = self.tree.query_ball_point(rft, coarse_search_radius)
        return centroid_hits

    cdef npc.uint8_t[:] _filter_search(
        self,
        npc.uint8_t[:] binary_kmer,
        list centers,
        npc.uint32_t[:, :] hash_vals,
        int max_misses,
        double inner_radius=0.2,
        double eps=1.01,
        inner_metric='needle'
    ):
        """Return a binary numpy array. Each element is 1 if a cluster passed filtering, otherwise 0."""
        cdef int i = 0
        cdef npc.uint8_t[:, :] searched
        cdef Cluster cluster
        cdef npc.uint8_t[:] filtered_centers = np.zeros((len(centers,)), dtype=np.uint8)
        cdef int n_points_original = 0
        for center in centers:
            cluster = self.db.get_cluster(center)
            if self.logging:
                n_points_original += cluster.seqs.shape[0]
            if inner_metric == 'none':
                filtered_centers[i] = 1
            elif cluster.seqs.shape[0] <= 0 or cluster.test_membership_hvals(hash_vals, max_misses):
                filtered_centers[i] = 1
            i += 1
        if self.logging:
            self.logger(f'Cluster filtering complete. {sum(filtered_centers)} clusters remaining.')
            self.logger(f'Allowing up to {max_misses}.')
        return filtered_centers

    cdef npc.uint8_t[:, :] _fine_search(
        self,
        npc.uint8_t[:] query_kmer,
        Cluster cluster,
        npc.uint8_t[:] row_hits,
        double inner_radius=0.2,
        inner_metric='needle'
    ):
        """Search a single cluster and return all members within inner_radius."""
        cdef npc.uint8_t[:, :] out = np.ndarray(
            (cluster.seqs.shape[0], self.ramifier.k),
            dtype=np.uint8
        )
        cdef double[:, :] score = 1000 * np.ones((self.ramifier.k + 1, self.ramifier.k + 1))
        cdef int i, j
        cdef int added = 0
        cdef npc.uint8_t bound = <npc.uint8_t> ceil(inner_radius * query_kmer.shape[0])
        cdef double[:] dists 
        if inner_metric == 'needle':
            dists = cluster.search_cluster(query_kmer, bound, score)
        cdef double inner = 100 * self.ramifier.k  # big value that will be larger than inner rad
        for i in range(cluster.seqs.shape[0]):
            if inner_metric == 'needle':  # and cluster.test_seq(i, row_hits):
                inner = dists[i]
            elif inner_metric == 'hamming':
                inner = hamming_dist(query_kmer, cluster.seqs[i, :], True)
            if inner_metric == 'none' or inner <= inner_radius:
                for j in range(self.ramifier.k):
                    out[added, j] = cluster.seqs[i, j]
                added += 1
        out = out[0:added, :]
        return out

    cdef npc.uint32_t[:, :] compute_hashes_for_seq(self, npc.uint8_t[:] seq):
        cdef npc.uint32_t[:, :] hash_vals = np.ndarray(
            (seq.shape[0] - self.sub_k + 1, self.n_hashes), dtype=np.uint32
        )
        cdef int i, j
        for i in range(seq.shape[0] - self.sub_k + 1):
            for j in range(self.n_hashes):
                hash_vals[i, j] = fnva(seq[i:i + self.sub_k], self.hash_functions[j, :])
                hash_vals[i, j] = fast_modulo(hash_vals[i, j], self.array_size)
        return hash_vals

    cdef npc.uint8_t[:, :] search(
        self,
        npc.uint8_t[:] binary_kmer,
        double search_radius,
        int max_filter_misses,
        double inner_radius=0.2,
        double eps=1.01,
        inner_metric='needle'
    ):
        """Perform 3-stage search on a query. Return an array of k-mers with dimensions (n_hits, k)."""
        # Coarse Search
        if self.logging:
            self.logger(f'Starting search.')
        cdef list centers = self._coarse_search(binary_kmer, search_radius, eps=eps)
        if self.logging:
            self.logger(f'Coarse search complete. {len(centers)} clusters.')

        # Filtering
        cdef int i = 0
        cdef npc.uint32_t[:, :] hash_vals = self.compute_hashes_for_seq(binary_kmer)
        cdef npc.uint8_t[:] filtered_centers = self._filter_search(
            binary_kmer,
            centers,
            hash_vals,
            max_filter_misses,
            inner_radius=inner_radius,
            eps=eps,
            inner_metric=inner_metric,
        )

        # Fine search
        cdef npc.uint8_t[:, :] out = np.ndarray((0, self.ramifier.k), dtype=np.uint8)
        i = -1
        cdef npc.uint8_t[:] row_hits
        for center in centers:
            i += 1
            if filtered_centers[i] == 1:
                cluster = self.db.get_cluster(center)
                row_hits = cluster.test_row_membership(hash_vals, max_filter_misses)
                if inner_metric == 'none' or max(row_hits) > 0:
                    searched = self._fine_search(
                        binary_kmer,
                        cluster,
                        row_hits,
                        inner_radius=inner_radius,
                        inner_metric=inner_metric
                    )
                    out = np.append(out, searched, axis=0)
        if self.logging:
            self.logger(f'Fine search complete. {out.shape[0]} candidates passed.')
        return out
