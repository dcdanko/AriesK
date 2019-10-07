# cython: profile=True
# cython: linetrace=True
# cython: language_level=3

import numpy as np
cimport numpy as npc

from ariesk.utils.bloom_filter cimport BloomGrid, fnva
from ariesk.utils.kmers cimport (
    encode_kmer,
    decode_kmer,
    bounded_needle_fast,
)

COARSEN = 10


cdef class Cluster:

    def __cinit__(self, centroid_id, seqs, sub_k):
        self.centroid_id = centroid_id
        self.seqs = seqs
        self.n_seqs = self.seqs.shape[0]
        self.k = seqs.shape[1]
        self.sub_k = sub_k

    cpdef build_bloom_grid(self, int filter_len, npc.uint64_t[:, :] col_hashes):
        cdef int n_row_hashes = 1
        cdef int grid_height = 4
        cdef npc.uint64_t[:, :] row_hashes = npc.ndarray((n_row_hashes, self.k), dtype=np.uint64)
        cdef int i, j
        for i in range(n_row_hashes):
            for j, val in enumerate(np.random.permutation(self.k)):
                row_hashes[i, j] = val
        self.bloom_grid = BloomGrid(
            self.sub_k, self.k, filter_len, grid_height, row_hashes, col_hashes
        )
        for i in range(self.seqs.shape[0]):
            self.bloom_grid.add(self.seqs[i, :])

    cpdef build_subclusters(self, int radius):
        if self.n_seqs <= COARSEN:
            self.build_linear_subclusters()
        else:
            self.build_spherical_subclusters(radius)

    cdef build_linear_subclusters(self):
        self.inner_clusters = np.zeros((self.n_seqs, self.n_seqs), dtype=np.uint64)
        self.inner_centers = []
        self.inner_radius = 0
        self.inner_cluster_type = 'linear'
        cdef int seq_i
        for seq_i in range(self.n_seqs):
            self.inner_centers.append(seq_i)

    cdef build_spherical_subclusters(self, int radius):
        cdef list centers = []
        cdef npc.uint64_t[:, :] clusters = np.zeros((self.n_seqs, self.n_seqs), dtype=np.uint64)
        cdef double[:, :] score = 1000 * np.ones((self.k + 1, self.k + 1))
        cdef int seq_i, center_i
        cdef double dist
        for seq_i in range(self.n_seqs):
            added = False
            for center_i in centers:
                dist = bounded_needle_fast(
                    self.seqs[seq_i], self.seqs[center_i], radius, False, score
                )
                if dist <= radius:
                    clusters[center_i, seq_i] = 1
                    added = True
                    break
            if not added:
                centers.append(seq_i)
        self.inner_radius = radius
        self.inner_centers = centers
        self.inner_clusters = clusters
        self.inner_cluster_type = 'spherical'

    def py_search_cluster(self, str seq, int bound):
        cdef double[:, :] score = 1000 * np.ones((self.k + 1, self.k + 1))
        return np.array(self.search_cluster(encode_kmer(seq), bound, score))

    cdef double[:] search_cluster(self, npc.uint8_t[:] seq, int bound, double[:, :] score):
        cdef int n_seqs = len(self.seqs)
        cdef double[:] dists = 1000 * np.ones((self.n_seqs,))
        for center_i in self.inner_centers:
            dist = bounded_needle_fast(
                seq, self.seqs[center_i], bound - self.inner_radius, False, score
            )
            if dist <= (bound - self.inner_radius):
                dists[center_i] = dist
                for i in range(self.n_seqs):
                    if self.inner_clusters[center_i, i] > 0:
                        dists[i] = dist
            elif dist <= bound:
                dists[center_i] = dist
                for i in range(self.n_seqs):
                    if self.inner_clusters[center_i, i] > 0:
                        dist = bounded_needle_fast(
                            seq, self.seqs[i], bound, False, score
                        )
                        if dist >= bound:
                            dists[i] = dist
            elif dist <= (bound + self.inner_radius):
                for i in range(self.n_seqs):
                    if self.inner_clusters[center_i, i] > 0:
                        dist = bounded_needle_fast(
                            seq, self.seqs[i], bound, False, score
                        )
                        if dist >= bound:
                            dists[i] = dist
        return dists



    def py_test_membership(self, str seq, int allowed_misses):
        return self.test_membership(encode_kmer(seq), allowed_misses)

    def py_count_membership(self, str seq):
        return self.count_membership(encode_kmer(seq))

    cdef int count_membership(self, npc.uint8_t[:] query_seq):
        cdef int i
        cdef int n_hits = 0
        for i in range(self.seqs.shape[1] - self.sub_k + 1):
            if self.bloom_grid.array_contains(query_seq[i:i + self.sub_k]):
                n_hits += 1
        return n_hits

    cdef bint test_membership(self, npc.uint8_t[:] query_seq, int allowed_misses):
        cdef int n_hits = self.count_membership(query_seq)
        cdef int min_hits = self.seqs.shape[1] - self.sub_k + 1 - (self.sub_k * allowed_misses)
        return n_hits >= min_hits

    cdef int count_membership_hvals(self, npc.uint32_t[:, :] hash_vals):
        cdef int i
        cdef int n_hits = 0
        for i in range(hash_vals.shape[0]):
            if self.bloom_grid.array_contains_hvals(hash_vals[i, :]):
                n_hits += 1
        return n_hits

    cdef bint test_membership_hvals(self, npc.uint32_t[:, :] hash_vals, int allowed_misses):
        cdef int n_hits = self.count_membership_hvals(hash_vals)
        cdef int min_hits = self.seqs.shape[1] - self.sub_k + 1 - (self.sub_k * allowed_misses)
        return n_hits >= min_hits

    cdef npc.uint8_t[:] test_row_membership(self, npc.uint32_t[:, :] hash_vals, int allowed_misses):
        cdef int min_hits = self.seqs.shape[1] - self.sub_k + 1 - (self.sub_k * allowed_misses)
        cdef npc.uint8_t[:] row_counts = self.bloom_grid.count_grid_contains_hvals(hash_vals)
        for i in range(self.bloom_grid.grid_height):
            if row_counts[i] >= min_hits:
                row_counts[i] = 1
            else:
                row_counts[i] = 0
        return row_counts

    cdef bint test_seq(self, int seq_id, npc.uint8_t[:] row_hits):
        cdef npc.uint64_t hval
        cdef int n_hvals = 0
        for i in range(self.bloom_grid.row_hashes.shape[0]):
            hval = fnva(self.seqs[seq_id, :], self.bloom_grid.row_hashes[i, :])
            hval = hval % self.bloom_grid.grid_height
            n_hvals += row_hits[hval]
        return n_hvals == self.bloom_grid.row_hashes.shape[0]

    @classmethod
    def build_from_seqs(cls, centroid_id, seqs, sub_k, **kwargs):
        """Build from a list of python strings. Mostly for testing."""
        np_seqs = np.array([encode_kmer(seq) for seq in seqs])
        out = cls(centroid_id, np_seqs, sub_k, **kwargs)
        return out
