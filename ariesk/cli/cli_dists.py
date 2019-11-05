import click
import pandas as pd
from time import time
from gimmebio.seqs import reverseComplement

from ariesk.search_server import SearchClient, SearchServer
from ariesk.contig_searcher import ContigSearcher
from ariesk.ram import (
    Ramifier,
    StatisticalRam,
    RotatingRamifier,
)
from ariesk.dist_matrix_builder import DistMatrixBuilder
from Bio import SeqIO
from ariesk.utils.kmers import py_needle


@click.group('dists')
def dist_cli():
    pass


@dist_cli.command('lev')
@click.option('-g', '--gap', default=100)
@click.option('-k', '--kmer-len', default=512)
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('fasta', type=click.File('r'))
def cli_lev_dist_matrix(gap, kmer_len, outfile, fasta):
    seqs = [str(el.seq) for el in SeqIO.parse(fasta, 'fasta')]
    kmers = set()
    for seq in seqs:
        for i in range(0, len(seq) - kmer_len, gap):
            kmer = seq[i:i + kmer_len]
            rc = reverseComplement(kmer)
            if hash(rc) < hash(kmer):
                kmer = rc
            kmers.add(kmer)
    click.echo(f'{len(kmers)} unique kmers.', err=True)
    start = time()
    dist_tbl = pd.DataFrame(py_needle(list(kmers)), columns=['k1', 'k2', 'lev'])
    elapsed = time() - start
    click.echo(f'{elapsed:.5}s to build distance matrix.', err=True)
    dist_tbl.to_csv(outfile)


@dist_cli.command('ram')
@click.option('-r', '--radius', default=10.)
@click.option('-d', '--dim', default=16)
@click.option('-g', '--gap', default=100)
@click.option('-k', '--kmer-len', default=512)
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('fasta', type=click.File('r'))
def cli_lev_dist_matrix(radius, dim, gap, kmer_len, outfile, fasta):
    seqs = [str(el.seq) for el in SeqIO.parse(fasta, 'fasta')]
    kmers = set()
    for seq in seqs:
        for i in range(0, len(seq) - kmer_len, gap):
            kmer = seq[i:i + kmer_len]
            rc = reverseComplement(kmer)
            if hash(rc) < hash(kmer):
                kmer = rc
            kmers.add(kmer)
    click.echo(f'{len(kmers)} unique kmers.', err=True)
    matrixer = DistMatrixBuilder(list(kmers), dim, kmer_len)

    start = time()
    dist_tbl = pd.DataFrame(matrixer.build(radius), columns=['k1', 'k2', 'lev'])
    elapsed = time() - start
    click.echo(f'{elapsed:.5}s to build distance matrix.', err=True)
    dist_tbl.to_csv(outfile)
