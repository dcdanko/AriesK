import click
import pandas as pd

import socket

from random import shuffle
from time import time
from json import dumps, loads

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
@click.option('-n', '--num-kmers', default=0, help='Number of kmers to cluster.')
@click.option('-s', '--start-offset', default=0)
@click.option('-o', '--outfile', default='ariesk_grid_cover_db.sqlite', type=click.Path())
@click.argument('rotation', type=click.Path())
@click.argument('kmer_table', type=click.Path())
def build_grid_cover(radius, dimension, num_kmers, start_offset, outfile, rotation, kmer_table):
    ramifier = RotatingRamifier.from_file(dimension, rotation)
    grid = GridCoverBuilder.from_filepath(outfile, ramifier, radius)
    start = time()
    grid.add_kmers_from_file(kmer_table, start=start_offset, num_to_add=num_kmers)
    add_time = time() - start
    click.echo(f'Added {num_kmers:,} kmers to cover in {add_time:.5}s.', err=True)

    start = time()
    grid.commit()
    cluster_time = time() - start
    n_centers = grid.db.centroids().shape[0]
    click.echo(f'Built grid cover in {cluster_time:.5}s. {n_centers:,} clusters.', err=True)

    grid.close()


@main.command('merge-grid')
@click.argument('final_db', type=click.Path())
@click.argument('other_dbs', type=click.Path(), nargs=-1)
def merge_grid_cover(final_db, other_dbs):
    final_db = GridCoverDB.load_from_filepath(final_db)
    for other_db_filename in other_dbs:
        other_db = GridCoverDB.load_from_filepath(other_db_filename)
        final_db.load_other(other_db)


@main.command('search')
@click.option('-p', '--port', default=50007)
@click.option('-r', '--radius', default=1.0)
@click.argument('kmer')
def search(port, radius, kmer):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', port))
        s.sendall(f'{kmer} {radius}'.encode('utf-8'))


@main.command('search-server')
@click.option('-p', '--port', default=50007)
@click.argument('grid_cover', type=click.Path())
def run_search_server(port, grid_cover):
    grid = GridCoverSearcher.from_filepath(grid_cover)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', port))
        s.listen(1)
        print(f'Listening on port {port}')
        while True:
            conn, addr = s.accept()
            with conn:
                while True:
                    data = conn.recv(1024).decode('utf-8')
                    if not data: break

                    kmer, radius = data.split()
                    print(f'Searching {kmer} with radius {radius}')
                    results = grid.search(kmer, float(radius))
                    for result in results:
                        print(f'{kmer} {result}')
                
