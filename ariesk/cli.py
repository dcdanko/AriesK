import click
import pandas as pd

from random import shuffle
from time import clock
from json import dumps

from gimmebio.sample_seqs import EcoliGenome
from gimmebio.kmers import make_kmers

from ariesk.rft_kdtree import RftKdTree
from ariesk.dists import DistanceFactory


@click.group()
def main():
    pass


@main.command('dists')
@click.option('-k', '--kmer-len', default=31)
@click.option('-n', '--num-kmers', default=1000, help='Number of kmers to compare.')
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('kmer_table', type=click.File('r'))
def calculate_kmer_dists_cluster(kmer_len, num_kmers, outfile, kmer_table):
    dist_factory = DistanceFactory(kmer_len)
    kmers = [line.strip().split(',')[0] for i, line in enumerate(kmer_table) if i < num_kmers]
    tbl = []
    start = clock()
    for k1 in kmers:
        for k2 in kmers:
            tbl.append(dist_factory.all_dists(k1, k2))
    run_time = clock() - start
    print(f'time: {run_time:.5}s')
    tbl = pd.DataFrame(tbl)
    tbl.to_csv(outfile)


@main.command('eval')
@click.option('-k', '--kmer-len', default=32)
@click.option('-r', '--radius', default=0.1)
@click.option('-n', '--num-kmers', default=1000, help='Number of kmers to cluster.')
@click.option('-o', '--outfile', default='-', type=click.File('w'))
def eval_kdrft_cluster(kmer_len, radius, num_kmers, outfile):
    ecoli_kmers = [str(el) for el in make_kmers(
        EcoliGenome().longest_contig()[:kmer_len + num_kmers - 1],
        kmer_len, canon=True
    )]
    print(f'Made {len(ecoli_kmers)} E. coli k-mers for testing')

    tree = RftKdTree(radius, kmer_len, num_kmers)
    start = clock()
    tree.bulk_add_kmers(ecoli_kmers)
    tree.cluster_greedy()
    build_time = clock() - start

    time_units = 's'
    if build_time < 1:
        build_time *= 1000
        time_units = 'ms'
    elif build_time > 1000:
        build_time /= 60
        time_units = 'm'

    print(f'Build time: {build_time:.5}{time_units}')
    print(tree.stats())


@main.command('build')
@click.option('-r', '--radius', default=0.1)
@click.option('-k', '--kmer-len', default=31)
@click.option('-n', '--num-kmers', default=1000, help='Number of kmers to cluster.')
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('kmer_table', type=click.File('r'))
def build_kdrft_cluster(radius, kmer_len, num_kmers, outfile, kmer_table):
    tree = RftKdTree(radius, kmer_len, num_kmers)

    start = clock()
    for i, line in enumerate(kmer_table):
        if i >= num_kmers:
            break
        kmer = line.strip().split(',')[0]
        tree.add_kmer(kmer)
    add_time = clock() - start
    click.echo(f'Added {num_kmers} kmers to cover in {add_time:.5}s.', err=True)

    start = clock()
    tree.cluster_greedy(logger=lambda el: click.echo(el, err=True))
    cluster_time = clock() - start
    click.echo(f'Clustered tree in {cluster_time:.5}s.', err=True)

    click.echo(tree.stats(), err=True)
    cover_as_dict = tree.to_dict()
    print(dumps(cover_as_dict), file=outfile)
