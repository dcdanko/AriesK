import click
from random import shuffle
from time import clock

from gimmebio.sample_seqs import EcoliGenome
from gimmebio.kmers import make_kmers

from ariesk.rft_kdtree import RftKdTree


@click.group()
def main():
    pass


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
