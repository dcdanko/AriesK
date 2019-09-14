
import random

from json import loads

from os.path import join, dirname
from unittest import TestCase
from ariesk.dists import DistanceFactory
from ariesk.ram import (
    Ramifier,
    StatisticalRam,
    RotatingRamifier,
)
from ariesk.grid_cover import GridCoverBuilder
from ariesk.searcher import GridCoverSearcher

from ariesk.utils import py_convert_kmer, py_reverse_convert_kmer

KMER_TABLE = join(dirname(__file__), 'small_31mer_table.csv')
KMER_ROTATION = join(dirname(__file__), '../data/rotation_minikraken.json')
GRID_COVER = join(dirname(__file__), 'small_grid_cover.json')

def random_kmer(k):
    return ''.join([random.choice('ATCG') for _ in range(k)])


class TestAriesK(TestCase):

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


    def test_dists(self):
        dist_factory = DistanceFactory(31)
        all_dists = dist_factory.all_dists(
            'ATCGATCGATCGATCGATCGATCGATCGATCG',
            'TTCGATCGATCGATCGATCGATCGATCGATCG'
        )
        self.assertEqual(all_dists['hamming'], 1)


    def test_build_grid_cover(self):
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        grid = GridCoverBuilder(0.5, 100, ramifier)
        grid.add_kmers_from_file(KMER_TABLE)
        grid.cluster()
        n_centers = len(grid.clusters.keys())
        n_points = sum([len(cluster) for cluster in grid.clusters.values()])
        self.assertGreater(n_centers, 0)
        self.assertLess(n_centers, 100)
        self.assertEqual(n_points, 100)

    def test_one_search_grid_cover_broad(self):
        grid = GridCoverSearcher.from_dict(loads(open(GRID_COVER).read()))
        query = 'AATACGTCCGGAGTATCGACGCACACATGGT'
        results = grid.search(query, 10)
        self.assertIn(query, results)

    def test_one_search_grid_cover_tight(self):
        grid = GridCoverSearcher.from_dict(loads(open(GRID_COVER).read()))
        query = 'ATGATCCTTCCGCCAAAGTACGTCCGGAGCA'
        results = grid.search(query, 0.0001)
        self.assertIn(query, results)
        self.assertGreaterEqual(len(results), 1)

    def test_all_search_grid_cover_broad(self):
        grid = GridCoverSearcher.from_dict(loads(open(GRID_COVER).read()))
        n_hits, n_kmers = 0, 0
        for kmer in grid.kmers:
            n_kmers += 1
            query = py_reverse_convert_kmer(kmer)
            results = grid.search(query, 10)
            if query in results:
                n_hits += 1
        self.assertEqual(n_hits, n_kmers)

    def test_all_search_grid_cover_tight(self):
        grid = GridCoverSearcher.from_dict(loads(open(GRID_COVER).read()))
        n_hits, n_kmers = 0, 0
        for kmer in grid.kmers:
            n_kmers += 1
            query = py_reverse_convert_kmer(kmer)
            results = grid.search(query, 0.0001)
            if query in results:
                n_hits += 1
        self.assertEqual(n_hits, n_kmers)

