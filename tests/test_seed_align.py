
import random

from os.path import join, dirname
from unittest import TestCase

from ariesk.seed_align import py_seed_extend


KMER_31 = 'ATCGATCGATCGATCGATCGATCGATCGATC'
MIS     = 'TTCGATCGATCGATCGATCGATCGATCGATC'
GAP     = 'TATCGATCGATCGATCGATCGATCGATCGAT'


def random_kmer(k):
    return ''.join([random.choice('ATCG') for _ in range(k)])


class TestSeedExtend(TestCase):

    def test_seed_extend(self):
        seq1 = 'ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG'
        seq2 = 'ATATATATATATATATATATATATATATATATATATATATATATATAT'
        seq3 = 'CCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCA'
        target = seq3 + seq2 + seq1 + seq2 + seq3
        matching_intervals = py_seed_extend(seq1, target)
        self.assertEqual(
            matching_intervals[0, 1] - matching_intervals[0, 0],
            matching_intervals[0, 3] - matching_intervals[0, 2]
        )

    def test_seed_extend_big_gap(self):
        seq1 = 'ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG'
        seq2 = 'ATATATATATATATATATATATATATATATATATATATATATATATAT'
        seq3 = 'CCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCA'
        target = seq3 + seq2 + seq1 + seq2 + seq3 + seq2 + seq1 + seq2 + seq3
        matching_intervals = py_seed_extend(seq1 + seq1, target)
        print(matching_intervals)
        assert False

    def test_seed_extend_gap(self):
        seq1 = 'ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG'
        seq2 = 'ATATATATATATATATATATATATATATATATATATATATATATATAT'
        seq3 = 'CCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCA'
        target = seq3 + seq2 + seq1 + seq2 + seq3
        gap = 'AAAAA'
        matching_intervals = py_seed_extend(seq1[:24] + gap + seq1[24:], target)
        print(matching_intervals)
        self.assertEqual(
            matching_intervals[0, 1] - matching_intervals[0, 0] - len(gap),
            matching_intervals[0, 3] - matching_intervals[0, 2]
        )
