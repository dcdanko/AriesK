
import random
import sqlite3
import numpy as np

from json import loads
from os import remove
from os.path import join, dirname
from unittest import TestCase

from ariesk.dbs.contig_db import ContigDB
from ariesk.dbs.pre_contig_db import PreContigDB
from ariesk.contig_searcher import ContigSearcher
from ariesk.ram import RotatingRamifier

KMER_TABLE = join(dirname(__file__), 'small_31mer_table.csv')
KMER_ROTATION = join(dirname(__file__), '../data/rotation_minikraken.json')
KMER_31 = 'ATCGATCGATCGATCGATCGATCGATCGATCG'
KMER_FASTA = join(dirname(__file__), 'small_fasta.fa')


def random_kmer(k):
    return ''.join([random.choice('ATCG') for _ in range(k)])


class TestContigDB(TestCase):

    def test_build_contig_db(self):
        conn = sqlite3.connect(':memory:')
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        contig_db = ContigDB(conn, ramifier=ramifier, box_side_len=0.5)
        contig = random_kmer(2 * 10 * 1000)
        contig_db.py_add_contig('test_genome', 'test_contig', contig, gap=100)
        contig_db.commit()
        stored = contig_db.get_all_contigs()
        self.assertGreaterEqual(len(stored), 2)

    def test_build_merge_contig_db(self):
        conn_1 = sqlite3.connect(':memory:')
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        contig_db_1 = ContigDB(conn_1, ramifier=ramifier, box_side_len=0.5)
        contig = random_kmer(2 * 10 * 1000)
        contig_db_1.py_add_contig('test_genome_1', 'test_contig_1', contig, gap=100)
        contig_db_1.commit()
        n_stored = len(contig_db_1.get_all_contigs())

        conn_2 = sqlite3.connect(':memory:')
        contig_db_2 = ContigDB(conn_2, ramifier=ramifier, box_side_len=0.5)
        contig = random_kmer(2 * 10 * 1000)
        contig_db_2.py_add_contig('test_genome_2', 'test_contig_2', contig, gap=100)
        contig_db_2.commit()
        n_stored += len(contig_db_2.get_all_contigs())

        contig_db_1.load_other(contig_db_2)

        self.assertEqual(len(contig_db_1.get_all_contigs()), n_stored)

    def test_fileio_contig_db(self):
        fname = 'temp.test_contig_db.sqlite'
        try:
            remove(fname)
        except FileNotFoundError:
            pass
        conn = sqlite3.connect(fname)
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        contig_db = ContigDB(conn, ramifier=ramifier, box_side_len=1)
        contig = random_kmer(2 * 10 * 1000)
        contig_db.py_add_contig('test_genome', 'test_contig', contig, gap=100)
        contig_db.commit()
        from_store = ContigDB.load_from_filepath(fname)
        self.assertEqual(contig_db.current_seq_coord, from_store.current_seq_coord)
        self.assertEqual(len(contig_db.centroid_cache), len(from_store.centroid_cache))
        for key, val in contig_db.centroid_cache.items():
            self.assertIn(key, from_store.centroid_cache)
            self.assertEqual(val, from_store.centroid_cache[key])
        remove(fname)

    def test_build_contig_db_from_fasta(self):
        conn = sqlite3.connect(':memory:')
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        contig_db = ContigDB(conn, ramifier=ramifier, box_side_len=0.5)
        contig_db.fast_add_kmers_from_fasta(KMER_FASTA)
        contig_db.commit()
        stored = contig_db.get_all_contigs()
        self.assertGreaterEqual(len(stored), 3)


    def test_search_contig_db(self):
        conn = sqlite3.connect(':memory:')
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        contig_db = ContigDB(conn, ramifier=ramifier, box_side_len=0.5)
        contig = random_kmer(2 * 10 * 1000)
        contig_db.py_add_contig('test_genome', 'test_contig', contig, gap=10)
        contig_db.commit()
        stored = contig_db.get_all_contigs()
        searcher = ContigSearcher(contig_db)
        hits = searcher.py_search(contig[500:600], 0.1, 0.5)
        self.assertGreaterEqual(len(hits), 1)

    def test_search_contig_db_exact(self):
        conn = sqlite3.connect(':memory:')
        ramifier = RotatingRamifier.from_file(4, KMER_ROTATION)
        contig_db = ContigDB(conn, ramifier=ramifier, box_side_len=0.5)
        contig = random_kmer(2 * 10 * 1000)
        contig_db.py_add_contig('test_genome', 'test_contig', contig, gap=10)
        contig_db.commit()
        stored = contig_db.get_all_contigs()
        searcher = ContigSearcher(contig_db)
        hits = searcher.py_search(contig[500:1500], 0.000001, 1)
        self.assertGreaterEqual(len(hits), 1)

    def test_search_bigger_contig_db_exact(self):
        contig_db = ContigDB(
            sqlite3.connect(':memory:'),
            ramifier=RotatingRamifier.from_file(4, KMER_ROTATION),
            box_side_len=0.0001
        )
        n_contigs, contig_len = 3, 2 * 10 * 1000
        contigs = [random_kmer(contig_len) for _ in range(n_contigs)]
        for i, contig in enumerate(contigs):
            contig_db.py_add_contig(f'test_genome_{i}', f'test_contig_{i}', contig, gap=1)
        contig_db.commit()
        self.assertEqual(contig_db.centroids().shape[0], n_contigs * (contig_len - 31 + 1))

        searcher = ContigSearcher(contig_db)
        hits = searcher.py_search(contigs[0][500:600], 0, 1)
        self.assertEqual(len(hits), 1)
