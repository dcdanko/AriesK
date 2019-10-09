
import random
import sqlite3
import numpy as np

from json import loads
from os import remove
from os.path import join, dirname
from unittest import TestCase

from ariesk.dbs.contig_db import ContigDB
from ariesk.ram import RotatingRamifier

KMER_TABLE = join(dirname(__file__), 'small_31mer_table.csv')
KMER_ROTATION = join(dirname(__file__), '../data/rotation_minikraken.json')
KMER_31 = 'ATCGATCGATCGATCGATCGATCGATCGATCG'


def random_kmer(k):
    return ''.join([random.choice('ATCG') for _ in range(k)])


class TestContigDB(TestCase):

    def test_modulo_deterministic(self):
        conn = sqlite3.connect(':memory:')
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        contig_db = ContigDB(conn, ramifier=ramifier, box_side_len=0.5)
        contig = random_kmer(2 * 10 * 1000)
        contig_db.py_add_contig('test_genome', 'test_contig', contig, gap=100)
        stored = contig_db.get_all_contigs()
        self.assertEqual(len(stored), 2)
