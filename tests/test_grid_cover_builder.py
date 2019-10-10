
import sqlite3

from os.path import join, dirname
from unittest import TestCase

from ariesk.ram import RotatingRamifier
from ariesk.grid_builder import GridCoverBuilder
from ariesk.dbs.kmer_db import GridCoverDB
from ariesk.pre_db import PreDB
from ariesk.utils.parallel_build import coordinate_parallel_build

KMER_TABLE = join(dirname(__file__), 'small_31mer_table.csv')
KMER_FASTA = join(dirname(__file__), 'small_fasta.fa')
KMER_ROTATION = join(dirname(__file__), '../data/rotation_minikraken.json')
GRID_COVER = join(dirname(__file__), 'small_grid_cover.sqlite')
PRE_DB = join(dirname(__file__), 'small_pre_grid.sqlite')

KMER_31 = 'AATACGTCCGGAGTATCGACGCACACATGGT'


class TestBuildGridCover(TestCase):

    def test_build_grid_cover(self):
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        db = GridCoverDB(sqlite3.connect(':memory:'), ramifier=ramifier, box_side_len=0.5)
        grid = GridCoverBuilder(db)
        grid.fast_add_kmers_from_file(KMER_TABLE)
        grid.commit()
        n_centers = grid.db.centroids().shape[0]
        n_points = len(grid.db.get_kmers())
        self.assertGreater(n_centers, 0)
        self.assertLess(n_centers, 100)
        self.assertEqual(n_points, 100)

    def test_fast_build_grid_cover(self):
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        db = GridCoverDB(sqlite3.connect(':memory:'), ramifier=ramifier, box_side_len=0.5)
        grid = GridCoverBuilder(db)
        grid.fast_add_kmers_from_file(KMER_TABLE)
        grid.commit()
        n_centers = grid.db.centroids().shape[0]
        n_points = len(grid.db.get_kmers())
        self.assertGreater(n_centers, 0)
        self.assertLess(n_centers, 100)
        self.assertEqual(n_points, 100)

    def test_build_grid_cover_from_fasta(self):
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        db = GridCoverDB(sqlite3.connect(':memory:'), ramifier=ramifier, box_side_len=0.5)
        grid = GridCoverBuilder(db)
        grid.fast_add_kmers_from_fasta(KMER_FASTA)
        grid.commit()
        n_centers = grid.db.centroids().shape[0]
        n_points = len(grid.db.get_kmers())
        self.assertGreater(n_centers, 0)
        self.assertLess(n_centers, 98)
        self.assertEqual(n_points, 98)

    def test_build_grid_cover_from_pre(self):
        predb = PreDB.load_from_filepath(PRE_DB)
        grid = GridCoverBuilder.build_from_predb(':memory:', predb, 0.5)
        grid.db._build_indices()
        grid.commit()
        n_centers = grid.db.centroids().shape[0]
        n_points = len(grid.db.get_kmers())
        self.assertGreater(n_centers, 0)
        self.assertLess(n_centers, 98)
        self.assertEqual(n_points, 98)

    ''' Test is slow, not really that useful
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
    '''