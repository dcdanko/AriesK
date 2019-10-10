import click
from time import time

from ariesk.dbs.kmer_db import GridCoverDB
from ariesk.utils.parallel_build import coordinate_parallel_build
from ariesk.search_server import SearchClient, SearchServer
from ariesk.grid_searcher import GridCoverSearcher
from ariesk.contig_searcher import ContigSearcher
from ariesk.ram import (
    Ramifier,
    StatisticalRam,
    RotatingRamifier,
)

class TimingLogger:

    def __init__(self, logger):
        self.last_message_time = time()
        self.logger = logger

    def log(self, msg):
        time_elapsed = time() - self.last_message_time
        msg = f'[{time_elapsed:.5}s] {msg}'
        self.logger(msg)
        self.last_message_time = time()


@click.group('search')
def search_cli():
    pass


@search_cli.command('contig')
@click.option('-v/-q', '--verbose/--quiet', default=False)
@click.option('-r', '--radius', default=0.01)
@click.option('-f', '--kmer-fraction', default=0.5)
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('contig_db', type=click.Path())
@click.argument('contigs', nargs=-1)
def search_contig(verbose, radius, kmer_fraction, outfile, contig_db, contigs):
    logger = None
    if verbose:
        logger = TimingLogger(lambda el: click.echo(el, err=True)).log
    searcher = ContigSearcher.from_filepath(contig_db, logger=logger)
    for contig in contigs:
        min_time = 1000 * 1000
        for _ in range(3):  # for testing
            start = time()
            hits = searcher.py_search(contig, radius, kmer_fraction)
            elapsed = time() - start
            if elapsed < min_time:
                min_time = elapsed
        click.echo(f'Search complete in {min_time:.5}s')
        for hit in hits:
            print(f'{contig} {hit}', file=outfile)


@search_cli.command('seq')
@click.option('-p', '--port', default=5432)
@click.option('-r', '--radius', default=1.0)
@click.option('-i', '--inner-radius', default=1.0)
@click.option('-m', '--inner-metric', default='needle', type=click.Choice(['hamming', 'needle', 'none']))
@click.option('-s', '--search-mode', default='full', type=click.Choice(['full', 'coarse']))
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('kmers', nargs=-1)
def search_seq(port, radius, inner_radius, inner_metric, search_mode, outfile, kmers):
    searcher = SearchClient(port)
    for kmer in kmers:
        start = time()
        results = searcher.search(
            kmer, radius, inner_radius,
            search_mode=search_mode, inner_metric=inner_metric
        )
        elapsed = time() - start
        click.echo(f'Search complete in {elapsed:.5}s', err=True)
        for result in results:
            print(f'{kmer} {result}', file=outfile)


@search_cli.command('seq-manual')
@click.option('-p', '--port', default=5432)
@click.option('-r', '--radius', default=1.0)
@click.option('-i', '--inner-radius', default=1.0)
@click.option('-m', '--inner-metric', default='needle', type=click.Choice(['hamming', 'needle', 'none']))
@click.option('-s', '--search-mode', default='full', type=click.Choice(['full', 'coarse']))
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('kmers', nargs=-1)
def search_seq_manual(port, radius, inner_radius, inner_metric, search_mode, outfile, kmers):
    searcher = SearchClient(port)
    for kmer in kmers:
        start = time()
        results = searcher.search(
            kmer, radius, inner_radius,
            search_mode=search_mode, inner_metric=inner_metric
        )
        elapsed = time() - start
        click.echo(f'Search complete in {elapsed:.5}s', err=True)
        for result in results:
            print(f'{kmer} {result}', file=outfile)

@search_cli.command('file')
@click.option('-p', '--port', default=5432)
@click.option('-r', '--radius', default=1.0)
@click.option('-i', '--inner-radius', default=1.0)
@click.option('-m', '--inner-metric', default='needle')
@click.option('-s', '--search-mode', default='full')
@click.argument('outfile', type=click.Path())
@click.argument('seqfile', type=click.Path())
def search_file(port, radius, inner_radius, inner_metric, search_mode, outfile, seqfile):
    searcher = SearchClient(port)
    start = time()
    searcher.search(
        seqfile, radius, inner_radius,
        search_mode=search_mode, inner_metric=inner_metric,
        result_file=outfile, query_type='file'
    )
    elapsed = time() - start
    click.echo(f'Search complete in {elapsed:.5}s', err=True)


@search_cli.command('run-server')
@click.option('-v/-q', '--verbose/--quiet', default=False)
@click.option('-p', '--port', default=5432)
@click.argument('grid_cover', type=click.Path())
def run_search_server(verbose, port, grid_cover):
    logger = None
    if verbose:
        logger = lambda el: click.echo(el, err=True)
    server = SearchServer.from_filepath(port, grid_cover, logger=logger)
    click.echo(f'Starting server on port {port}', err=True)
    server.main_loop()


@search_cli.command('shutdown-server')
@click.option('-p', '--port', default=5432)
def search_file(port):
    searcher = SearchClient(port)
    searcher.send_shutdown()
