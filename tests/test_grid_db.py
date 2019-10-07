
import random
import sqlite3
import numpy as np

from json import loads
from os import remove
from os.path import join, dirname
from unittest import TestCase

from ariesk.db import GridCoverDB
from ariesk.grid_searcher import GridCoverSearcher
from ariesk.pre_db import PreDB
from ariesk.ram import RotatingRamifier

KMER_TABLE = join(dirname(__file__), 'small_31mer_table.csv')
KMER_ROTATION = join(dirname(__file__), '../data/rotation_minikraken.json')
KMER_31 = 'ATCGATCGATCGATCGATCGATCGATCGATG'
KMER_30 = 'TTCGATCGATCGATCGATCGATCGATCGAC'

def reverse_convert_kmer(binary_kmer):
    base_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    out = ''
    for base in binary_kmer:
        out += base_map[base]
    return out


class TestGridCoverDB(TestCase):

    def test_add_kmer(self):
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        db = GridCoverDB(sqlite3.connect(':memory:'), ramifier=ramifier, box_side_len=0.5)
        db.py_add_point_to_cluster(np.array([0., 0., 0., 0.]), KMER_31)
        db.commit()
        members = db.py_get_cluster_members(0)
        self.assertEqual(len(members), 1)
        self.assertIn(KMER_31, [reverse_convert_kmer(member) for member in members])

    def test_add_kmer_to_pre(self):
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        db = PreDB(sqlite3.connect(':memory:'), ramifier=ramifier)
        db.py_add_kmer(KMER_31)
        db.commit()
        members = list(db.conn.execute('SELECT * FROM kmers'))
        self.assertEqual(len(members), 1)
        self.assertIn(KMER_31, [reverse_convert_kmer(member[1]) for member in members])

    def test_merge_dbs(self):
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        db1 = GridCoverDB(sqlite3.connect(':memory:'), ramifier=ramifier, box_side_len=0.5)
        db1.py_add_point_to_cluster(np.array([0., 0., 0., 0.]), KMER_30 + 'A')
        db1.py_add_point_to_cluster(np.array([1., 0., 0., 0.]), KMER_30 + 'T')
        db1.commit()
        db2 = GridCoverDB(sqlite3.connect(':memory:'), ramifier=ramifier, box_side_len=0.5)
        db2.py_add_point_to_cluster(np.array([0., 0., 0., 0.]), KMER_30 + 'C')
        db2.py_add_point_to_cluster(np.array([1., 1., 0., 0.]), KMER_30 + 'G')
        db2.commit()
        db1.load_other(db2)
        centroids = db1.centroids()
        self.assertEqual(centroids.shape, (3, 4))
        kmers = [el[1] for el in db1.get_kmers()]
        self.assertEqual(len(kmers), 4)
        for char in 'ATCG':
            self.assertIn(KMER_30 + char, kmers)

    def test_get_centroids(self):
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        db = GridCoverDB(sqlite3.connect(':memory:'), ramifier=ramifier, box_side_len=0.5)
        db.py_add_point_to_cluster(np.array([0., 0., 0., 0.]), KMER_30 + 'A')
        db.py_add_point_to_cluster(np.array([0., 0., 0., 0.]), KMER_30 + 'T')
        db.py_add_point_to_cluster(np.array([1., 0., 0., 0.]), KMER_30 + 'C')
        db.commit()
        centroids = db.centroids()
        self.assertEqual(centroids.shape, (2, 4))

    def test_pre_build_blooms(self):
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        db = GridCoverDB(sqlite3.connect(':memory:'), ramifier=ramifier, box_side_len=0.5)
        db.py_add_point_to_cluster(np.array([0., 0., 0., 0.]), KMER_30 + 'A')
        db.py_add_point_to_cluster(np.array([0., 0., 0., 0.]), KMER_30 + 'T')
        db.py_add_point_to_cluster(np.array([1., 0., 0., 0.]), KMER_30 + 'C')
        db.commit()
        searcher = GridCoverSearcher(db)
        for centroid_id in [0, 1]:
            db.build_and_store_bloom_grid(
                centroid_id, searcher.array_size, searcher.hash_functions, searcher.sub_k
            )
        bg_0 = db.retrieve_bloom_grid(0)
        bg_1 = db.retrieve_bloom_grid(1)
        self.assertEqual(max(bg_0.py_count_grid_contains(KMER_30 + 'A')), 32 - bg_0.col_k)
        self.assertEqual(max(bg_1.py_count_grid_contains(KMER_30 + 'C')), 32 - bg_1.col_k)

    def test_save(self):
        DB_SAVE_TEMP_FILE = join(dirname(__file__), 'temp.db_save_temp.sqlite')
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        db = GridCoverDB(sqlite3.connect(DB_SAVE_TEMP_FILE), ramifier=ramifier, box_side_len=0.5)
        db.py_add_point_to_cluster(np.array([0., 0., 0., 0.]), KMER_31)
        db.close()
        remove(DB_SAVE_TEMP_FILE)

    def test_save_and_reload(self):
        DB_SAVE_TEMP_FILE = join(dirname(__file__), 'temp.db_save_temp.sqlite')
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        db = GridCoverDB(sqlite3.connect(DB_SAVE_TEMP_FILE), ramifier=ramifier, box_side_len=0.5)
        db.py_add_point_to_cluster(np.array([0., 0., 0., 0.]), KMER_31)
        db.close()
        del db
        db = GridCoverDB.load_from_filepath(DB_SAVE_TEMP_FILE)
        members = db.py_get_cluster_members(0)
        self.assertEqual(len(members), 1)
        self.assertIn(KMER_31, [reverse_convert_kmer(member) for member in members])
        remove(DB_SAVE_TEMP_FILE)
