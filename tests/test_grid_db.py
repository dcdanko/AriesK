
import random
import sqlite3
import numpy as np

from json import loads

from os.path import join, dirname
from unittest import TestCase
from ariesk.db import GridCoverDB
from ariesk.utils import py_convert_kmer, py_reverse_convert_kmer

KMER_TABLE = join(dirname(__file__), 'small_31mer_table.csv')
KMER_ROTATION = join(dirname(__file__), '../data/rotation_minikraken.json')


class TestGridCoverDB(TestCase):

    def test_add_kmer(self):
        db = GridCoverDB(sqlite3.connect(':memory:'))
        db.add_point_to_cluster(np.array([0, 0, 0, 0]), 'ATCGATCG')
        members = db.get_cluster_members(0)
        self.assertEqual(len(members), 1)
        self.assertIn('ATCGATCG', members)