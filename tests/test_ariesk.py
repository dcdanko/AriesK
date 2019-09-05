
import random

from unittest import TestCase
from ariesk.rft_kdtree import RftKdTree


def random_kmer(k):
    return ''.join([random.choice('ATCG') for _ in range(k)])


class TestAriesK(TestCase):

    def test_add_kmer(self):
        tree = RftKdTree(0.1, 32, 1)
        tree.add_kmer('ATCGATCGATCGATCGATCGATCGATCGATCG')
        self.assertTrue(tree.kmers.shape[0] == 1)

    def test_cannot_add_kmer(self):
        tree = RftKdTree(0.1, 32, 1)
        tree.add_kmer('ATCGATCGATCGATCGATCGATCGATCGATCG')
        self.assertRaises(
            AssertionError,
            lambda: tree.add_kmer('ATCGATCGATCGATCGATCGATCGATCGATCG')
        )

    def test_ramify(self):
        tree = RftKdTree(0.1, 32, 1)
        tree._ramify(0, 'ATCGATCGATCGATCGATCGATCGATCGATCG')
        self.assertTrue(tree.rfts.shape[0] == 1)

    def test_cluster_greedy_no_batch(self):
        N, K = 1000, 10
        tree = RftKdTree(0.4, K, N)
        for _ in range(N):
            tree.add_kmer(random_kmer(K))
        tree.cluster_greedy()
        self.assertTrue(len(tree.clusters) < N)
        self.assertTrue(0 < len(tree.clusters))

    def test_cluster_greedy_with_batch_twice(self):
        N, K = 2002, 10
        tree = RftKdTree(0.4, K, N)
        for _ in range(N):
            tree.add_kmer(random_kmer(K))
        tree.cluster_greedy()
        self.assertTrue(len(tree.clusters) < N)
        self.assertTrue(0 < len(tree.clusters))

    def test_cluster_greedy_with_batch(self):
        N, K = 1999, 10
        tree = RftKdTree(0.4, K, N)
        for _ in range(N):
            tree.add_kmer(random_kmer(K))
        tree.cluster_greedy()
        self.assertTrue(len(tree.clusters) < N)
        self.assertTrue(0 < len(tree.clusters))
