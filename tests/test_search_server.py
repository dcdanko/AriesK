
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


class TestSearchServer(TestCase):

    def test_search_server(self):
        client = SearchClient(PORT)
        run_server = lambda: SearchServer.from_filepath(PORT, GRID_COVER, auto_start=True)
        server_thread = KThread(target=run_server)
        print('ready')
        try:
            server_thread.start()
            results = list(client.search(KMER_31, 0.001, 0.1))
            client.send_shutdown()
            self.assertIn(KMER_31, results)
        finally:
            server_thread.terminate()
