
from kthread import KThread
from os.path import join, dirname
from os import remove
from unittest import TestCase

from ariesk.search_server import (
    SearchServer,
    SearchClient,
)

GRID_COVER = join(dirname(__file__), 'small_grid_cover.sqlite')
KMER_TABLE = join(dirname(__file__), 'small_31mer_table.csv')

PORT = 5432
KMER_31 = 'AATACGTCCGGAGTATCGACGCACACATGGT'


def run_server():
    SearchServer.from_filepath(PORT, GRID_COVER, auto_start=True)


class TestSearchServer(TestCase):

    def test_search_server(self):
        client = SearchClient(PORT)
        server_thread = KThread(target=run_server)
        try:
            server_thread.start()
            results = list(client.search(KMER_31, 0.001, 0.1))
            client.send_shutdown()
            self.assertIn(KMER_31, results)
        finally:
            if server_thread.is_alive():
                server_thread.terminate()

    def test_search_server_file(self):
        client = SearchClient(PORT)
        server_thread = KThread(target=run_server)
        outfile = 'temp.test_outfile.csv'
        try:
            server_thread.start()
            client.search(
                KMER_TABLE, 0.001, 0.1,
                result_file=outfile, query_type='file'
            )
            client.send_shutdown()
            results = []
            with open(outfile) as f:
                for line in f:
                    results.append(line.strip().split()[1])
            remove(outfile)
            self.assertIn(KMER_31, results)
        finally:
            if server_thread.is_alive():
                server_thread.terminate()

    def test_coarse_search_server(self):
        client = SearchClient(PORT)
        server_thread = KThread(target=run_server)
        try:
            server_thread.start()
            results = list(client.search(KMER_31, 0.001, 0.1, search_mode='coarse'))
            client.send_shutdown()
            self.assertGreaterEqual(len(results), 1)
        finally:
            if server_thread.is_alive():
                server_thread.terminate()

    def test_search_server_no_inner(self):
        client = SearchClient(PORT)
        server_thread = KThread(target=run_server)
        try:
            server_thread.start()
            results = list(client.search(KMER_31, 0.001, 0.1, inner_metric='none'))
            client.send_shutdown()
            self.assertIn(KMER_31, results)
        finally:
            if server_thread.is_alive():
                server_thread.terminate()
