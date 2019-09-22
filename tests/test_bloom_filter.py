
import random
import numpy as np
from unittest import TestCase

from ariesk.bloom_filter import BloomFilter, BloomGrid
from ariesk.cluster import Cluster

KMER_31 = 'ATCGATCGATCGATCGATCGATCGATCGATCG'


def random_kmer(k):
    return ''.join([random.choice('ATCG') for _ in range(k)])


class TestUtils(TestCase):

    def test_add_to_bloom_grid(self):
        k, sub_k = 31, 6
        bg = BloomGrid.build_from_probs(k, sub_k, 10, 2, 500, 0.01)
        seqs = []
        for _ in range(10):
            seqs.append(random_kmer(k))
            bg.py_add(seqs[-1])
        for seq in seqs:
            for i in range(k - sub_k + 1):
                sub_seq = seq[i:i + sub_k]
                self.assertTrue(bg.py_array_contains(sub_seq))
                self.assertGreaterEqual(sum(bg.py_grid_contains(sub_seq)), 1)
        for seq in seqs:
            self.assertGreaterEqual(sum(bg.py_count_grid_contains(seq)), 1)

    def test_add_to_bloom(self):
        bf = BloomFilter.build_from_probs(5, 500, 0.01)
        for _ in range(100):
            bf.py_add(random_kmer(5))
        self.assertEqual(bf.n_elements, 100)

    def test_in_bloom(self):
        bf = BloomFilter.build_from_probs(5, 500, 0.01)
        kmer = random_kmer(5)
        bf.py_add(kmer)
        self.assertTrue(bf.py_contains(kmer))

    def test_not_in_bloom(self):
        bf = BloomFilter.build_from_probs(5, 500, 0.01)
        kmer = random_kmer(5)
        bf.py_add(kmer)
        self.assertTrue(bf.py_contains(kmer))
        for _ in range(10):
            self.assertFalse(bf.py_contains(random_kmer(5)))

    def test_mostly_not_in_bloom_large(self):
        bf = BloomFilter.build_from_probs(31, 200, 0.01)
        for _ in range(100):
            bf.py_add(random_kmer(31))

        in_bloom = 0
        for _ in range(100):
            if bf.py_contains(random_kmer(31)):
                in_bloom += 1
        self.assertLessEqual(in_bloom, 5)

    def test_mostly_not_in_bloom_small(self):
        k = 6
        bf = BloomFilter.build_from_probs(k, 200, 0.01)
        for _ in range(100):
            bf.py_add(random_kmer(k))
        in_bloom = 0
        for _ in range(100):
            if bf.py_contains(random_kmer(k)):
                in_bloom += 1
        self.assertLessEqual(in_bloom, 5)

    def test_intersection(self):
        bf1 = BloomFilter.build_from_probs(40, 500, 0.00001)
        bf2 = BloomFilter.build_from_probs(40, 500, 0.00001)
        for _ in range(100):
            kmer = random_kmer(40)
            bf1.py_add(kmer)
            bf2.py_add(kmer)
        for _ in range(100):
            bf1.py_add(random_kmer(40))
            bf2.py_add(random_kmer(40))
        intersection = bf1.intersection(bf2)
        self.assertGreaterEqual(intersection, -10)
        self.assertLessEqual(intersection, 110)

    def test_union(self):
        bf1 = BloomFilter.build_from_probs(40, 500, 0.01)
        print(bf1.len_filter)
        print(bf1.n_hashes)
        bf2 = BloomFilter.build_from_probs(40, 500, 0.01)
        for _ in range(100):
            kmer = random_kmer(40)
            bf1.py_add(kmer)
            bf2.py_add(kmer)
        for _ in range(100):
            bf1.py_add(random_kmer(40))
            bf2.py_add(random_kmer(40))
        union = bf1.union(bf2)
        print(union)
        self.assertGreaterEqual(union, 290)
        self.assertLessEqual(union, 410)

    def test_cluster_membership(self):
        sub_k, k = 6, 31
        seqs = [random_kmer(k) for _ in range(100)]
        clust = Cluster.build_from_seqs(0, seqs, sub_k)
        clust.bloom_grid = BloomGrid.build_from_probs(k, sub_k, 4, 1, 500, 0.01)
        for i in range(len(seqs)):
            clust.bloom_grid.py_add(seqs[i])
        self.assertEqual(clust.py_count_membership(seqs[0]), 26)
        self.assertLess(clust.py_count_membership(random_kmer(k)), 26)
        self.assertTrue(clust.py_test_membership(seqs[0], 0))
        self.assertFalse(clust.py_test_membership(random_kmer(k), 0))
