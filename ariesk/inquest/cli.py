
import click
from json import loads
from Bio import SeqIO
from random import sample

from .distance_table import DistanceBuilder
from .utils import (
    mutate_seq,
    sample_fasta,
)


@click.group()
def main():
    pass


@main.command('dists')
@click.option('--num-queries', default=25)
@click.option('--num-targets', default=10000)
@click.argument('fasta', type=click.File('r'))
def cli_dist_table(num_queries, num_targets, fasta):
    seqs = [rec.seq for rec in SeqIO.parse(fasta, 'fasta')]
    for k in [32]: #, 64, 128, 256, 512, 1024]:
        click.echo(f'Building distance table for k={k}...', err=True)
        kmers = sample_fasta(seqs, k, num_targets)
        click.echo(f'\tSampled k-mers...', err=True)
        raw_queries, queries = sample(kmers, num_queries), []
        for query in raw_queries:
            queries.append(query)
            for sub_rate, indel_rate in [(0.01, 0.005), (0.05, 0.01), (0.1, 0.05)]:
                for _ in range(3):
                    queries.append(mutate_seq(query, sub_rate, indel_rate))
        click.echo(f'\tBuilt queries...', err=True)
        builder = DistanceBuilder(
            queries, kmers,
            k=k, sub_ks=[5, 6, 7, 8, 10, 12, 16],
            rft_dims=[1, 2, 4, 8, 16],
        )
        builder.build(f'distance_comparison_k{k}.csv')
