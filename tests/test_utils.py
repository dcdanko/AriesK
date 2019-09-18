
import random


from os.path import join, dirname
from unittest import TestCase
from ariesk.dists import DistanceFactory

from ariesk.utils import py_convert_kmer, py_reverse_convert_kmer
from ariesk.linear_searcher import LinearSearcher

KMER_TABLE = join(dirname(__file__), 'small_31mer_table.csv')
KMER_ROTATION = join(dirname(__file__), '../data/rotation_minikraken.json')
GRID_COVER = join(dirname(__file__), 'small_grid_cover.sqlite')
KMER_31 = 'ATCGATCGATCGATCGATCGATCGATCGATCG'


def random_kmer(k):
    return ''.join([random.choice('ATCG') for _ in range(k)])


class TestUtils(TestCase):

    def test_encode_decode_one_kmer(self):
        kmer = 'ATCG'
        code = py_convert_kmer(kmer)
        decoded = py_reverse_convert_kmer(code)
        self.assertEqual(kmer, decoded)

    def test_encode_decode_one_large_kmer(self):
        kmer = 'ATCG' * 128
        code = py_convert_kmer(kmer)
        decoded = py_reverse_convert_kmer(code)
        self.assertEqual(kmer, decoded)

    def test_encode_decode_many_kmers(self):
        for _ in range(10):
            for k in range(20, 40):
                kmer = random_kmer(k)
                code = py_convert_kmer(kmer)
                decoded = py_reverse_convert_kmer(code)
                self.assertEqual(kmer, decoded)

    def test_dists(self):
        dist_factory = DistanceFactory(31)
        all_dists = dist_factory.all_dists(
            'ATCGATCGATCGATCGATCGATCGATCGATCG',
            'TTCGATCGATCGATCGATCGATCGATCGATCG'
        )
        self.assertEqual(all_dists['hamming'], 1)

    def test_linear_search(self):
        searcher = LinearSearcher.from_filepath(GRID_COVER)
        results = searcher.search(KMER_31, metric='needle')
        self.assertEqual(len(results), 100)

    def test_linear_search_hamming(self):
        searcher = LinearSearcher.from_filepath(GRID_COVER)
        results = searcher.search(KMER_31, metric='hamming')
        self.assertEqual(len(results), 100)
