
import random

from os.path import join, dirname
from unittest import TestCase
from ariesk.rft_kdtree import RftKdTree
from ariesk.dists import DistanceFactory
from ariesk.ram import (
    Ramifier,
    StatisticalRam,
    RotatingRamifier,
)

KMER_TABLE = join(dirname(__file__), 'small_31mer_table.csv')


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
        ramifier = Ramifier(32)
        rft = ramifier.ramify('ATCGATCGATCGATCGATCGATCGATCGATCG')
        self.assertTrue(len(rft.shape) == 1)
        self.assertTrue(rft.shape[0] == (4 * 32))

    def test_centers(self):
        stat_ram = StatisticalRam(31, 100)
        stat_ram.add_kmers_from_file(KMER_TABLE)
        centers = stat_ram.get_centers()
        self.assertTrue(centers.shape == (4 * 31,))

    def test_scales(self):
        stat_ram = StatisticalRam(31, 100)
        stat_ram.add_kmers_from_file(KMER_TABLE)
        scales = stat_ram.get_scales()
        self.assertTrue(scales.shape == (4 * 31,))

    def test_rotation(self):
        stat_ram = StatisticalRam(31, 100)
        stat_ram.add_kmers_from_file(KMER_TABLE)
        rotation = stat_ram.get_rotation()
        self.assertTrue(rotation.shape == (4 * 31, 4 * 31))

    def test_rotating_ramifier(self):
        stat_ram = StatisticalRam(31, 100)
        stat_ram.add_kmers_from_file(KMER_TABLE)
        centers = stat_ram.get_centers()
        scales = stat_ram.get_scales()
        rotation = stat_ram.get_rotation()
        rotater = RotatingRamifier(31, 8, rotation, centers, scales)
        rft = rotater.ramify('ATCGATCGATCGATCGATCGATCGATCGATC')
        self.assertTrue(rft.shape == (8,))

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

    def test_dists(self):
        dist_factory = DistanceFactory(31)
        all_dists = dist_factory.all_dists(
            'ATCGATCGATCGATCGATCGATCGATCGATCG',
            'TTCGATCGATCGATCGATCGATCGATCGATCG'
        )
        self.assertEqual(all_dists['hamming'], 1)

