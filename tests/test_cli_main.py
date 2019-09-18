
from kthread import KThread
from os.path import join, dirname, abspath
from os import remove
from unittest import TestCase
from click.testing import CliRunner

from ariesk.cli import main

from ariesk.search_server import (
    SearchServer,
    SearchClient,
)

GRID_COVER = abspath(join(dirname(__file__), 'small_grid_cover.sqlite'))
KMER_TABLE = abspath(join(dirname(__file__), 'small_31mer_table.csv'))
KMER_ROTATION = abspath(join(dirname(__file__), '../data/rotation_minikraken.json'))

PORT = 5431
KMER_31 = 'AATACGTCCGGAGTATCGACGCACACATGGT'


def run_server():
    SearchServer.from_filepath(PORT, GRID_COVER, auto_start=True)


class TestMainCli(TestCase):

    def test_search_server_cli(self):
        runner = CliRunner()
        server_thread = KThread(target=run_server)
        try:
            server_thread.start()
            results = runner.invoke(
                main,
                ['search-seq', f'-p {PORT}', '--search-mode=full', '-r 0', '-i 0.1', KMER_31]
            )
            self.assertIn(KMER_31, results.output)
            runner.invoke(main, ['shutdown-search-server', f'-p {PORT}'])
        finally:
            if server_thread.is_alive():
                server_thread.terminate()

    def test_search_file_server_cli(self):
        runner = CliRunner()
        server_thread = KThread(target=run_server)
        with runner.isolated_filesystem():
            outfile = 'temp.test_file_search.csv'
            try:
                server_thread.start()
                result = runner.invoke(
                    main,
                    [
                        'search-file', f'-p {PORT}', '--search-mode=coarse', '-r 0', '-i 0.1',
                        outfile, KMER_TABLE
                ])
                self.assertEqual(result.exit_code, 0)
                runner.invoke(main, ['shutdown-search-server', f'-p {PORT}'])
            finally:
                if server_thread.is_alive():
                    server_thread.terminate()

    def test_build_db_cli(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            db_path = 'temp.db_cli_test.sqlite'
            result = runner.invoke(
                main, [
                    'build-grid', '-r 0.5', '-d 4', '-n 50',
                    '-s 50', f'-o={db_path}',
                    KMER_ROTATION, KMER_TABLE
            ])
            self.assertEqual(result.exit_code, 0)

    def test_db_stats(self):
        runner = CliRunner()
        result = runner.invoke(main, ['stats', 'cover-stats', GRID_COVER])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('kmers\t100', result.output)
