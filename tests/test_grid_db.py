
import random
import sqlite3
import numpy as np

from json import loads
from os import remove
from os.path import join, dirname
from unittest import TestCase

from ariesk.db import GridCoverDB
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
        members = db.py_get_cluster_members(0)
        self.assertEqual(len(members), 1)
        self.assertIn(KMER_31, [reverse_convert_kmer(member) for member in members])

    def test_merge_dbs(self):
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        db1 = GridCoverDB(sqlite3.connect(':memory:'), ramifier=ramifier, box_side_len=0.5)
        db1.py_add_point_to_cluster(np.array([0., 0., 0., 0.]), KMER_30 + 'A')
        db1.py_add_point_to_cluster(np.array([1., 0., 0., 0.]), KMER_30 + 'T')
        db2 = GridCoverDB(sqlite3.connect(':memory:'), ramifier=ramifier, box_side_len=0.5)
        db2.py_add_point_to_cluster(np.array([0., 0., 0., 0.]), KMER_30 + 'C')
        db2.py_add_point_to_cluster(np.array([1., 1., 0., 0.]), KMER_30 + 'G')
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
        centroids = db.centroids()
        self.assertEqual(centroids.shape, (2, 4))

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

        