import click
import pandas as pd

from random import shuffle
from time import time
from json import dumps, loads
from shutil import copyfile
from os.path import isfile
from os import environ

from gimmebio.sample_seqs import EcoliGenome
from gimmebio.kmers import make_kmers

from ariesk.dists import DistanceFactory
from ariesk.searcher import GridCoverSearcher

from ariesk.ram import (
    Ramifier,
    StatisticalRam,
    RotatingRamifier,
)
from ariesk.grid_cover import GridCoverBuilder
from ariesk.db import GridCoverDB
from ariesk.parallel_build import coordinate_parallel_build
from ariesk.search_server import SearchClient, SearchServer

from .cli_dev import dev_cli
from .cli_stats import stats_cli


@click.group()
def main():
    pass


main.add_command(dev_cli)
main.add_command(stats_cli)


@main.command('rotate')
@click.option('-k', '--kmer-len', default=31)
@click.option('-n', '--num-kmers', default=1000, help='Number of kmers to compare.')
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('kmer_table', type=click.Path('r'))
def calculate_pca_rotation(kmer_len, num_kmers, outfile, kmer_table):
    """Calculate a PCA rotation from a set of k-mers."""
    stat_ram = StatisticalRam(kmer_len, num_kmers)
    stat_ram.add_kmers_from_file(kmer_table)
    out = {
        'k': kmer_len,
        'center': stat_ram.get_centers().tolist(),
        'scale': stat_ram.get_scales().tolist(),
        'rotation': stat_ram.get_rotation().tolist(),
    }
    outfile.write(dumps(out))


@main.command('build-grid')
@click.option('-r', '--radius', default=0.02, type=float)
@click.option('-d', '--dimension', default=8)
@click.option('-t', '--threads', default=1)
@click.option('-n', '--num-kmers', default=0, help='Number of kmers to cluster.')
@click.option('-s', '--start-offset', default=0)
@click.option('-o', '--outfile', default='ariesk_grid_cover_db.sqlite', type=click.Path())
@click.option('--preload/--no-preload', default=False, help='Load k-mers into RAM before processing')
@click.argument('rotation', type=click.Path())
@click.argument('kmer_table', type=click.Path())
def build_grid_cover(radius, dimension, threads, num_kmers, start_offset, outfile, preload, rotation, kmer_table):
    environ['OPENBLAS_NUM_THREADS'] = f'{threads}'  # numpy uses one of these two libraries
    environ['MKL_NUM_THREADS'] = f'{threads}'
    ramifier = RotatingRamifier.from_file(dimension, rotation)
    grid = GridCoverBuilder.from_filepath(outfile, ramifier, radius)
    start = time()
    n_added = grid.add_kmers_from_file(kmer_table, start=start_offset, num_to_add=num_kmers, preload=preload)
    grid.commit()
    n_centers = grid.db.centroids().shape[0]
    grid.close()
    add_time = time() - start
    click.echo(f'Added {n_added:,} kmers to {outfile} in {add_time:.5}s. {n_centers:,} clusters.', err=True)



@main.command('parallel-build-grid')
@click.option('-r', '--radius', default=0.02, type=float)
@click.option('-d', '--dimension', default=8)
@click.option('-t', '--threads', default=1)
@click.option('-c', '--chunk-size', default=100 * 1000)
@click.option('-n', '--num-kmers', default=0, help='Number of kmers to cluster.')
@click.option('-s', '--start-offset', default=0)
@click.option('-o', '--outfile', default='ariesk_grid_cover_db.sqlite', type=click.Path())
@click.argument('rotation', type=click.Path())
@click.argument('kmer_table', type=click.Path())
def build_grid_cover(radius, dimension, threads, chunk_size, num_kmers,
                     start_offset, outfile, rotation, kmer_table):
    def logger(num, total):
        click.echo(f'Finished {num + 1} chunks of {total}', err=True)
        if num + 1 == total:
            click.echo('Merging...', err=True)

    environ['OPENBLAS_NUM_THREADS'] = '2'  # numpy uses one of these two libraries
    environ['MKL_NUM_THREADS'] = '2'
    start = time()
    coordinate_parallel_build(
        outfile, kmer_table, rotation,
        (3 * threads) // 4, start_offset, num_kmers, radius, dimension,
        chunk_size=chunk_size, logger=logger
    )
    elapsed = time() - start
    click.echo(f'Built grid cover in {elapsed:.5}s.', err=True)


@main.command('merge-grid')
@click.argument('final_db', type=click.Path())
@click.argument('other_dbs', type=click.Path(), nargs=-1)
def merge_grid_cover(final_db, other_dbs):
    if not isfile(final_db):
        copyfile(other_dbs[0], final_db)
        other_dbs = other_dbs[1:]
    final_db = GridCoverDB.load_from_filepath(final_db)
    for other_db_filename in other_dbs:
        other_db = GridCoverDB.load_from_filepath(other_db_filename)
        final_db.load_other(other_db)


@main.command('search')
@click.option('--coarse/--full', default=False)
@click.option('--fast/--slow', default=False)
@click.option('-p', '--port', default=5432)
@click.option('-r', '--radius', default=1.0)
@click.option('-i', '--inner-radius', default=1.0)
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('kmers', nargs=-1)
def search(coarse, fast, port, radius, inner_radius, outfile, kmers):
    searcher = SearchClient(port)
    for kmer in kmers:
        if coarse:
            for result in searcher.coarse_search(kmer, radius):
                print(f'{kmer} {result}', file=outfile)
        else:
            for result in searcher.search(kmer, radius, inner_radius, fast=fast):
                print(f'{kmer} {result}', file=outfile)


@main.command('run-search-server')
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
