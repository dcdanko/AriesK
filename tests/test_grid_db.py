
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
KMER_31 = 'ATCGATCGATCGATCGATCGATCGATCGATCG'
KMER_30 = 'TTCGATCGATCGATCGATCGATCGATCGATC'


class TestGridCoverDB(TestCase):

    def test_add_kmer(self):
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        db = GridCoverDB(sqlite3.connect(':memory:'), ramifier=ramifier, box_side_len=0.5)
        db.add_point_to_cluster(np.array([0, 0, 0, 0]), KMER_31)
        members = db.get_cluster_members(0)
        self.assertEqual(len(members), 1)
        self.assertIn(KMER_31, members)

    def test_get_centroids(self):
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        db = GridCoverDB(sqlite3.connect(':memory:'), ramifier=ramifier, box_side_len=0.5)
        db.add_point_to_cluster(np.array([0, 0, 0, 0]), KMER_30 + 'A')
        db.add_point_to_cluster(np.array([0, 0, 0, 0]), KMER_30 + 'T')
        db.add_point_to_cluster(np.array([1, 0, 0, 0]), KMER_30 + 'C')
        centroids = db.centroids()
        self.assertEqual(centroids.shape, (2, 4))

    def test_save(self):
        DB_SAVE_TEMP_FILE = join(dirname(__file__), 'temp.db_save_temp.sqlite')
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        db = GridCoverDB(sqlite3.connect(DB_SAVE_TEMP_FILE), ramifier=ramifier, box_side_len=0.5)
        db.add_point_to_cluster(np.array([0, 0, 0, 0]), KMER_31)
        db.close()
        remove(DB_SAVE_TEMP_FILE)

    def test_save_and_reload(self):
        DB_SAVE_TEMP_FILE = join(dirname(__file__), 'temp.db_save_temp.sqlite')
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        db = GridCoverDB(sqlite3.connect(DB_SAVE_TEMP_FILE), ramifier=ramifier, box_side_len=0.5)
        db.add_point_to_cluster(np.array([0, 0, 0, 0]), KMER_31)
        db.close()
        del db
        db = GridCoverDB.load_from_filepath(DB_SAVE_TEMP_FILE)
        members = db.get_cluster_members(0)
        self.assertEqual(len(members), 1)
        self.assertIn(KMER_31, members)
        remove(DB_SAVE_TEMP_FILE)

        