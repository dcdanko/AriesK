
import numpy as np
cimport numpy as npc
from scipy.spatial import cKDTree as KDTree

from .ramft import build_rs_matrix


cdef long convert_kmer(str kmer):
    cdef long out = 0
    for i, base in enumerate(kmer):
        val = 0
        if base == 'C':
            val = 1
        elif base == 'G':
            val = 2
        elif base == 'T':
            val = 3
        out += val + (4 ** i) 
    return out


cdef class RftKdTree:
    cdef public float radius
    cdef public long k, num_kmers_added, max_size, rft_dims
    cdef public long [:] kmers
    cdef public double [:, :] rfts
    cdef public double [:, :] rs_matrix
    cdef public object clusters

    def __cinit__(self, radius, k, max_size):
        self.radius = radius
        self.k = k
        self.rft_dims = min(12, k - 1)
        self.max_size = max_size
        self.num_kmers_added = 0

        self.rs_matrix = build_rs_matrix(self.k)

        self.kmers = npc.ndarray((self.max_size,), dtype=long)
        self.rfts = npc.ndarray((self.max_size, self.rft_dims))

        self.clusters = {}

    def  _ramify(self, int index, str kmer):
        cdef long [:, :] binary_kmer = np.array([
            [1 if base == seqb else 0 for seqb in kmer]
            for base in 'ACGT'
        ]).T
        cdef double [:, :] rft = abs(np.dot(self.rs_matrix, binary_kmer))
        cdef double [:] power_series = np.sum(rft, axis=1)
        self.rfts[self.num_kmers_added] = power_series[1:(1 + self.rft_dims)]

    cpdef add_kmer(self, str kmer):
        assert self.num_kmers_added < self.max_size
        cdef long kmer_code = convert_kmer(kmer)
        self.kmers[self.num_kmers_added] = kmer_code
        self._ramify(self.num_kmers_added, kmer)
        self.num_kmers_added += 1

    def bulk_add_kmers(self, kmers):
        for kmer in kmers:
            self.add_kmer(kmer)

    def cluster_greedy(self, logger=None):
        all_tree = KDTree(self.rfts)

        index_map = {i: i for i in range(self.num_kmers_added)}
        clusters, clustered_points = {}, set()

        for centroid_index in range(self.num_kmers_added):
            if centroid_index > 0 and centroid_index % 1000 == 0:
                if logger is not None:
                    logger(f'Point {centroid_index}, currently {len(clusters)} clusters')
                
                # Rebuild all_tree to only include points which are not yet clustered
                # this works because we cannot cluster points twice and it makes
                # the search space smaller (at the expense of rebuilding the tree and
                # added code complexity for offset)
                unclustered_points = npc.ndarray(
                    (self.max_size - len(clustered_points), self.rft_dims)
                )
                index_map = {}
                current_point_index = 0
                for unclustered_point_index in range(self.num_kmers_added):
                    if unclustered_point_index in clustered_points:
                        continue
                    unclustered_point = self.rfts[unclustered_point_index]
                    index_map[current_point_index] = unclustered_point_index
                    unclustered_points[current_point_index] = unclustered_point
                    current_point_index += 1
                if current_point_index > 0:
                    all_tree = KDTree(unclustered_points)

            if centroid_index in clustered_points:
                continue

            centroid = self.rfts[centroid_index]
            this_cluster_members = all_tree.query_ball_point(centroid, self.radius, eps=0.1)
            this_cluster_members = {
                index_map[member] for member in this_cluster_members
                if index_map[member] not in clustered_points
            }
            clusters[centroid_index] = set([centroid_index]) | this_cluster_members
            clustered_points |= this_cluster_members

        self.clusters = clusters

    def stats(self):
        return {
            'num_kmers': sum([len(clust) for clust in self.clusters.values()]),
            'num_singletons': sum([
                1 if len(clust) == 1 else 0 for clust in self.clusters.values()
            ]),
            'num_clusters': len(self.clusters),
        }

    def to_dict(self):
        out = {}
        for centroid_index, member_indices in self.clusters.items():
            centroid = self.kmers[centroid_index]
            out[centroid] = [self.kmers[member_index] for member_index in member_indices]
        return out
