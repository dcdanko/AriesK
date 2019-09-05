

from unittest import TestCase
from ariesk.rft_kdtree import RftKdTree


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
