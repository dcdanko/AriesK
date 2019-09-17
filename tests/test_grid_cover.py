
import sqlite3
import os

from json import loads
from os.path import join, dirname
from unittest import TestCase

from ariesk.ram import RotatingRamifier
from ariesk.grid_cover import GridCoverBuilder
from ariesk.searcher import GridCoverSearcher
from ariesk.utils import py_reverse_convert_kmer
from ariesk.db import GridCoverDB
from ariesk.parallel_build import coordinate_parallel_build

KMER_TABLE = join(dirname(__file__), 'small_31mer_table.csv')
KMER_ROTATION = join(dirname(__file__), '../data/rotation_minikraken.json')
GRID_COVER = join(dirname(__file__), 'small_grid_cover.sqlite')


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

    def test_build_parallel(self):
        out_name = 'temp.test_parallel_build.sqlite'
        coordinate_parallel_build(out_name, KMER_TABLE, KMER_ROTATION, 2, 0, 100, 0.5, 8, chunk_size=25)
        db = GridCoverDB.load_from_filepath(out_name)
        n_centers = db.centroids().shape[0]
        n_points = len(db.get_kmers())
        self.assertGreater(n_centers, 0)
        self.assertLess(n_centers, 100)
        self.assertEqual(n_points, 100)
        os.remove(out_name)

    def test_one_coarse_search_grid_cover_broad(self):
        grid = GridCoverSearcher.from_filepath(GRID_COVER)
        query = 'AATACGTCCGGAGTATCGACGCACACATGGT'
        results = grid._coarse_search(query, 10)
        self.assertGreater(len(results), 0)

    def test_one_search_grid_cover_broad(self):
        grid = GridCoverSearcher.from_filepath(GRID_COVER)
        query = 'AATACGTCCGGAGTATCGACGCACACATGGT'
        results = grid.search(query, 10)
        self.assertIn(query, results)

    def test_all_coarse_search_grid_cover_broad(self):
        grid = GridCoverSearcher.from_filepath(GRID_COVER)
        n_hits, n_kmers = 0, 0
        for centroid_id, kmer in grid.db.get_kmers():
            n_kmers += 1
            results = grid._coarse_search(kmer, 10)
            if centroid_id in results:
                n_hits += 1
        self.assertEqual(n_hits, n_kmers)

    def test_one_search_grid_cover_tight(self):
        grid = GridCoverSearcher.from_filepath(GRID_COVER)
        query = 'ATGATCCTTCCGCCAAAGTACGTCCGGAGCA'
        results = grid.search(query, 0.0001)
        self.assertIn(query, results)
        self.assertGreaterEqual(len(results), 1)

    def test_all_search_grid_cover_broad(self):
        grid = GridCoverSearcher.from_filepath(GRID_COVER)
        n_hits, n_kmers = 0, 0
        for _, kmer in grid.db.get_kmers():
            n_kmers += 1
            results = grid.search(kmer, 10)
            if kmer in results:
                n_hits += 1
        self.assertEqual(n_hits, n_kmers)

    def test_all_coarse_search_grid_cover_tight(self):
        grid = GridCoverSearcher.from_filepath(GRID_COVER)
        n_hits, n_kmers = 0, 0
        for centroid_id, kmer in grid.db.get_kmers():
            n_kmers += 1
            results = grid._coarse_search(kmer, 0.0001)
            if centroid_id in results:
                n_hits += 1
        self.assertEqual(n_hits, n_kmers)

    def test_all_search_grid_cover_tight(self):

        grid = GridCoverSearcher.from_filepath(GRID_COVER)
        n_hits, n_kmers = 0, 0
        for _, kmer in grid.db.get_kmers():
            n_kmers += 1
            results = grid.search(kmer, 0.0001)
            if kmer in results:
                n_hits += 1
        self.assertEqual(n_hits, n_kmers)
