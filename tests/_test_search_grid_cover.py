
from os.path import join, dirname
from unittest import TestCase

from ariesk.grid_searcher import GridCoverSearcher

KMER_TABLE = join(dirname(__file__), 'small_31mer_table.csv')
KMER_FASTA = join(dirname(__file__), 'small_fasta.fa')
KMER_ROTATION = join(dirname(__file__), '../data/rotation_minikraken.json')
GRID_COVER = join(dirname(__file__), 'small_grid_cover.sqlite')
PRE_DB = join(dirname(__file__), 'small_pre_grid.sqlite')

KMER_31 = 'AATACGTCCGGAGTATCGACGCACACATGGT'


class TestSearchGridCover(TestCase):

    def test_one_coarse_search_grid_cover_broad(self):
        grid = GridCoverSearcher.from_filepath(GRID_COVER)
        query = KMER_31
        results = grid.py_coarse_search(query, 10)
        self.assertGreater(len(results), 0)

    def test_one_search_grid_cover_broad(self):
        grid = GridCoverSearcher.from_filepath(GRID_COVER)
        query = KMER_31
        results = grid.py_search(query, 10)
        self.assertIn(query, results)

    def test_all_coarse_search_grid_cover_broad(self):
        grid = GridCoverSearcher.from_filepath(GRID_COVER)
        n_hits, n_kmers = 0, 0
        for centroid_id, kmer in grid.db.get_kmers():
            n_kmers += 1
            results = grid.py_coarse_search(kmer, 10)
            if centroid_id in results:
                n_hits += 1
        self.assertEqual(n_hits, n_kmers)

    def test_one_search_grid_cover_tight(self):
        grid = GridCoverSearcher.from_filepath(GRID_COVER)
        query = KMER_31
        results = grid.py_search(query, 0.0001, inner_metric='needle', inner_radius=0.01)
        self.assertIn(query, results)
        self.assertEqual(len(results), 1)

    def test_one_search_grid_cover_exact(self):
        grid = GridCoverSearcher.from_filepath(GRID_COVER)
        query = KMER_31
        results = grid.py_search(query, 0, inner_metric='needle', inner_radius=0)
        self.assertIn(query, results)
        self.assertEqual(len(results), 1)

    def test_double_search_grid_cover_tight(self):
        grid = GridCoverSearcher.from_filepath(GRID_COVER)
        query = KMER_31
        grid.py_search(query, 0.0001)
        results = grid.py_search(query, 0.0001, inner_metric='needle', inner_radius=0.01)
        self.assertIn(query, results)
        self.assertEqual(len(results), 1)

    def test_all_search_grid_cover_broad(self):
        grid = GridCoverSearcher.from_filepath(GRID_COVER)
        n_hits, n_kmers = 0, 0
        for _, kmer in grid.db.get_kmers():
            n_kmers += 1
            results = grid.py_search(kmer, 10)
            if kmer in results:
                n_hits += 1
        self.assertEqual(n_hits, n_kmers)

    def test_all_coarse_search_grid_cover_tight(self):
        grid = GridCoverSearcher.from_filepath(GRID_COVER)
        n_hits, n_kmers = 0, 0
        for centroid_id, kmer in grid.db.get_kmers():
            n_kmers += 1
            results = grid.py_coarse_search(kmer, 0.0001)
            if centroid_id in results:
                n_hits += 1
        self.assertEqual(n_hits, n_kmers)

    def test_all_search_grid_cover_tight(self):

        grid = GridCoverSearcher.from_filepath(GRID_COVER)
        n_hits, n_kmers = 0, 0
        for _, kmer in grid.db.get_kmers():
            n_kmers += 1
            results = grid.py_search(kmer, 0.0001, inner_metric='needle', inner_radius=0.01)
            if kmer in results:
                n_hits += 1
        self.assertEqual(n_hits, n_kmers)
