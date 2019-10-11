
import random

from os.path import join, dirname
from unittest import TestCase

from ariesk.seed_align import py_seed_extend

QUERY = 'ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG'
seq2 = 'ATATATATATATATATATATATATATATATATATATATATATATATAT'
seq3 = 'CCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCA'


def random_kmer(k):
    return ''.join([random.choice('ATCG') for _ in range(k)])


class TestSeedExtend(TestCase):

    def test_seed_extend(self):
        target = seq3 + seq2 + QUERY + seq2 + seq3
        matching_intervals = py_seed_extend(QUERY, target)
        self.assertEqual(matching_intervals.shape[0], 1)
        self.assertEqual(
            matching_intervals[0, 1] - matching_intervals[0, 0],
            matching_intervals[0, 3] - matching_intervals[0, 2]
        )
        self.assertLessEqual(matching_intervals[0, 1] - matching_intervals[0, 0], 60)
        self.assertGreaterEqual(matching_intervals[0, 1] - matching_intervals[0, 0], 40)

    def test_seed_extend_continues(self):
        target = seq3 + seq2 + QUERY + QUERY + seq2 + seq3
        matching_intervals = py_seed_extend(QUERY + QUERY, target)
        self.assertEqual(matching_intervals.shape[0], 1)
        self.assertEqual(
            matching_intervals[0, 1] - matching_intervals[0, 0],
            matching_intervals[0, 3] - matching_intervals[0, 2]
        )
        self.assertLessEqual(matching_intervals[0, 1] - matching_intervals[0, 0], 110)
        self.assertGreaterEqual(matching_intervals[0, 1] - matching_intervals[0, 0], 90)

    def test_seed_extend_big_gap(self):
        target = seq3 + seq2 + QUERY + seq2 + seq3 + seq2 + seq3 + seq2 + QUERY + seq2 + seq3
        matching_intervals = py_seed_extend(QUERY + QUERY, target)
        self.assertEqual(matching_intervals.shape[0], 2)
        self.assertLessEqual(matching_intervals[0, 1] - matching_intervals[0, 0], 60)
        self.assertGreaterEqual(matching_intervals[0, 1] - matching_intervals[0, 0], 40)
        self.assertLessEqual(matching_intervals[1, 1] - matching_intervals[1, 0], 60)
        self.assertGreaterEqual(matching_intervals[1, 1] - matching_intervals[1, 0], 40)

    def test_seed_extend_medium_gap(self):
        target = seq3 + seq2 + QUERY + seq2[:24] + QUERY + seq2 + seq3
        matching_intervals = py_seed_extend(QUERY + QUERY, target)
        self.assertEqual(matching_intervals.shape[0], 1)
        self.assertLessEqual(matching_intervals[0, 1] - matching_intervals[0, 0], 110)
        self.assertGreaterEqual(matching_intervals[0, 1] - matching_intervals[0, 0], 90)

    def test_seed_extend_small_gap(self):
        target = seq3 + seq2 + QUERY + seq2 + seq3
        gap = 'AAAAA'
        matching_intervals = py_seed_extend(QUERY[:24] + gap + QUERY[24:], target)
        self.assertEqual(matching_intervals.shape[0], 1)
        self.assertLessEqual(matching_intervals[0, 1] - matching_intervals[0, 0], 60)
        self.assertGreaterEqual(matching_intervals[0, 1] - matching_intervals[0, 0], 40)
