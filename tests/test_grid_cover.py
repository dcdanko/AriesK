
import sqlite3

from json import loads
from os.path import join, dirname
from unittest import TestCase

from ariesk.ram import RotatingRamifier
from ariesk.grid_cover import GridCoverBuilder
from ariesk.searcher import GridCoverSearcher
from ariesk.utils import py_reverse_convert_kmer
from ariesk.db import GridCoverDB

KMER_TABLE = join(dirname(__file__), 'small_31mer_table.csv')
KMER_ROTATION = join(dirname(__file__), '../data/rotation_minikraken.json')
GRID_COVER = join(dirname(__file__), 'small_grid_cover.json')


class TestGridCover(TestCase):

    def test_build_grid_cover(self):
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        db = GridCoverDB(sqlite3.connect(':memory:'), ramifier=ramifier, box_side_len=0.5)
        grid = GridCoverBuilder(db)
        grid.add_kmers_from_file(KMER_TABLE)
        grid.commit()
        n_centers = grid.db.centroids().shape[0]
        n_points = len(grid.db.get_kmers())
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
