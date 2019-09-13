import click
import pandas as pd

import socket

from random import shuffle
from time import time
from json import dumps, loads

from gimmebio.sample_seqs import EcoliGenome
from gimmebio.kmers import make_kmers

from ariesk.rft_kdtree import RftKdTree
from ariesk.dists import DistanceFactory
from ariesk.searcher import GridCoverSearcher

from ariesk.ram import (
    Ramifier,
    StatisticalRam,
    RotatingRamifier,
)
from ariesk.plaid_cover import PlaidCoverBuilder
from ariesk.grid_cover import GridCoverBuilder

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


@main.command('build-plaid')
@click.option('-r', '--radius', default=1.0, type=float)
@click.option('-d', '--dimension', default=8)
@click.option('-n', '--num-kmers', default=1000, help='Number of kmers to cluster.')
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('rotation', type=click.Path())
@click.argument('kmer_table', type=click.Path())
def build_plaid_cover(radius, dimension, num_kmers, outfile, rotation, kmer_table):
    ramifier = RotatingRamifier.from_file(dimension, rotation)
    plaid = PlaidCoverBuilder(radius, num_kmers, ramifier)
    plaid.add_kmers_from_file(kmer_table)
    click.echo(f'Added {num_kmers} kmers to cover.', err=True)

    start = time()
    plaid.cluster()
    cluster_time = time() - start
    n_centers = len(plaid.clusters.keys())
    click.echo(f'Built plaid cover in {cluster_time:.5}s. {n_centers} clusters.', err=True)

    outfile.write(dumps(plaid.to_dict()))


@main.command('build-grid')
@click.option('-r', '--radius', default=0.02, type=float)
@click.option('-d', '--dimension', default=8)
@click.option('-n', '--num-kmers', default=1000, help='Number of kmers to cluster.')
@click.option('-s', '--start-offset', default=1000)
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('rotation', type=click.Path())
@click.argument('kmer_table', type=click.Path())
def build_grid_cover(radius, dimension, num_kmers, start_offset, outfile, rotation, kmer_table):
    ramifier = RotatingRamifier.from_file(dimension, rotation)
    grid = GridCoverBuilder(radius, num_kmers, ramifier)
    start = time()
    grid.add_kmers_from_file(kmer_table, start=start_offset)
    add_time = time() - start
    click.echo(f'Added {num_kmers:,} kmers to cover in {add_time:.5}s.', err=True)

    start = time()
    grid.cluster()
    cluster_time = time() - start
    n_centers = len(grid.clusters.keys())
    click.echo(f'Built grid cover in {cluster_time:.5}s. {n_centers:,} clusters.', err=True)

    outfile.write(dumps(grid.to_dict()))


@main.command('merge-grid')
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('grid_covers', nargs=-1, type=click.File('r'))
def merge_grid_cover(outfile, grid_covers):
    first = loads(grid_covers[0].read())
    out = {
        'type': 'grid_cover',
        'radius': first['radius'],
        'ramifier': first['ramifier'],
        'kmers': first['kmers'],
        'clusters': []
    }
    clusters = {tuple(cluster['centroid']): cluster['members'] for cluster in first['clusters']}
    for grid_cover in grid_covers[1:]:
        grid_cover = loads(grid_cover.read())
        for cluster in grid_cover['clusters']:
            centroid = tuple(cluster['centroid'])
            members = [el + len(out['kmers']) for el in cluster['members']]
            clusters[centroid] = clusters.get(centroid, []) + members

    n_centers = len(clusters.keys())
    for centroid, members in clusters.items():
        out['clusters'].append({
            'centroid': centroid,
            'members': members,

        })
    click.echo(f'Merged grid covers. {n_centers:,} clusters.', err=True)
    outfile.write(dumps(out))


@main.command('build-tree')
@click.option('-r', '--radius', default=0.1)
@click.option('-k', '--kmer-len', default=31)
@click.option('-n', '--num-kmers', default=1000, help='Number of kmers to cluster.')
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('kmer_table', type=click.File('r'))
def build_kdrft_cluster(radius, kmer_len, num_kmers, outfile, kmer_table):
    tree = RftKdTree(radius, kmer_len, num_kmers)

    start = time()
    for i, line in enumerate(kmer_table):
        if i >= num_kmers:
            break
        kmer = line.strip().split(',')[0]
        tree.add_kmer(kmer)
    add_time = time() - start
    click.echo(f'Added {num_kmers} kmers to cover in {add_time:.5}s.', err=True)

    start = time()
    tree.cluster_greedy(logger=lambda el: click.echo(el, err=True))
    cluster_time = time() - start
    click.echo(f'Clustered tree in {cluster_time:.5}s.', err=True)

    click.echo(tree.stats(), err=True)
    cover_as_dict = tree.to_dict()
    print(dumps(cover_as_dict), file=outfile)


@main.command('search')
@click.option('-p', '--port', default=50007)
@click.option('-r', '--radius', default=1.0)
@click.argument('kmer')
def search(port, radius, kmer):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', port))
        s.sendall(f'{kmer} {radius}'.encode('utf-8'))
        #data = s.recv(1024)
        #print('Received', repr(data))


@main.command('search-server')
@click.option('-p', '--port', default=50007)
@click.argument('grid_cover', type=click.File('r'))
def run_search_server(port, grid_cover):
    grid = GridCoverSearcher.from_dict(loads(grid_cover.read()))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', port))
        s.listen(1)
        print(f'Listening on port {port}')
        while True:
            conn, addr = s.accept()
            with conn:
                # print('Connected by', addr)
                while True:
                    data = conn.recv(1024).decode('utf-8')
                    if not data: break

                    kmer, radius = data.split()
                    print(f'Searching {kmer} with radius {radius}')
                    results = grid.search(kmer, float(radius))
                    for result in results:
                        print(f'{kmer} {result}')
                
