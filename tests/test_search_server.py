
from kthread import KThread
from os.path import join, dirname
from unittest import TestCase

from ariesk.search_server import (
    SearchServer,
    SearchClient,
)

GRID_COVER = join(dirname(__file__), 'small_grid_cover.sqlite')

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
            server_thread.terminate()
