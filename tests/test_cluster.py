
import random
import numpy as np
from unittest import TestCase

from ariesk.utils.bloom_filter import BloomFilter, BloomGrid
from ariesk.cluster import Cluster

KMER_31 = 'ATCGATCGATCGATCGATCGATCGATCGATCG'


def random_kmer(k):
    return ''.join([random.choice('ATCG') for _ in range(k)])


class TestCluster(TestCase):

    def test_cluster_membership(self):
        sub_k, k = 7, 31
        seqs = [random_kmer(k) for _ in range(100)]
        clust = Cluster.build_from_seqs(0, seqs, sub_k)
        clust.bloom_grid = BloomGrid.build_from_probs(k, sub_k, 4, 1, 500, 0.01)
        for i in range(len(seqs)):
            clust.bloom_grid.py_add(seqs[i])
        self.assertEqual(clust.py_count_membership(seqs[0]), k - sub_k + 1)
        self.assertLess(clust.py_count_membership(random_kmer(k)), k - sub_k + 1)
        self.assertTrue(clust.py_test_membership(seqs[0], 0))
        self.assertFalse(clust.py_test_membership(random_kmer(k), 0))

    def test_cluster_linear_search(self):
        sub_k, k = 7, 31
        seqs = [random_kmer(k) for _ in range(3)]
        clust = Cluster.build_from_seqs(0, seqs, sub_k)
        clust.build_subclusters(1)  # will build linear
        dists = clust.py_search_cluster(seqs[0], 1)
        self.assertEqual(clust.inner_cluster_type, 'linear')
        self.assertEqual(min(dists), 0)
        self.assertEqual(dists.shape[0], 3)

    def test_cluster_spherical_search(self):
        sub_k, k = 7, 31
        seqs = [random_kmer(k) for _ in range(100)]
        sphere = [seqs[0][:-1] + 'A', seqs[0][:-1] + 'C', seqs[0][:-1] + 'G', seqs[0][:-1] + 'T']
        clust = Cluster.build_from_seqs(0, seqs + sphere, sub_k)
        clust.build_subclusters(1)  # will build spherical
        dists = clust.py_search_cluster(seqs[0], 1)
        self.assertEqual(clust.inner_cluster_type, 'spherical')
        self.assertEqual(min(dists), 0)
        self.assertLessEqual(sum(sorted(dists)[:4]), 3)
        self.assertEqual(dists.shape[0], 104)
