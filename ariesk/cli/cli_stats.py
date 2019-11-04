import click
import pandas as pd
import numpy as np

from json import dumps, loads
from time import clock
import random
from gimmebio.seqs import reverseComplement

from ariesk.grid_searcher import GridCoverSearcher
from ariesk.dbs.kmer_db import GridCoverDB
from ariesk.dbs.contig_db import ContigDB
from ariesk.utils.kmers import py_needle, py_needle_2
from ariesk.ram import RotatingRamifier, Ramifier
from Bio import SeqIO


@click.group('stats')
def stats_cli():
    pass


def mutate_seq(seq, max_rate=0.5):
    k = len(seq)
    snp_rate = max_rate * random.random()
    indel_rate = snp_rate / 10
    out = ''
    for base in seq:
        r = random.random()
        if r < indel_rate:
            if r < (indel_rate / 2):
                out += random.choice('ACGT') + base
        elif r < (snp_rate + indel_rate):
            out += random.choice('ACGT')
        else:
            out += base
    while len(out) < k:
        out += 'N'  # random.choice('ACGT')
    out = out[:k]
    return out


@stats_cli.command('calibrate')
@click.option('-d', '--dropout', default=1.0)
@click.option('-g', '--gap', default=10)
@click.option('-b', '--burst', default=2)
@click.option('-k', '--kmer-len', default=256)
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.option('-r', '--rotation', type=click.Path(), default=None)
@click.argument('fasta', type=click.File('r'))
def calibrate_db(dropout, gap, burst, kmer_len, outfile, rotation, fasta):
    seqs = [str(el.seq) for el in SeqIO.parse(fasta, 'fasta')]
    kmers = set()
    for seq in seqs:
        for i in range(0, len(seq) - kmer_len, gap):
            for j in range(burst):
                j = 0
                if random.random() < dropout:
                    kmer = seq[i + j:i + j + kmer_len]
                    # kmer = 'A' + kmer + 'C'
                    kmers.add(kmer)
                    # frac = 30
                    # mut_kmer = kmer[:(kmer_len // frac)]
                    # mut_kmer += mutate_seq(kmer[(kmer_len // frac):((frac - 1) * kmer_len // frac)])
                    # mut_kmer += kmer[((frac - 1) * kmer_len // frac):]
                    # kmers.add(mut_kmer)

    click.echo(f'{len(kmers)} kmers', err=True)
    dist_tbl = pd.DataFrame(py_needle(list(kmers)), columns=['k1', 'k2', 'f_lev'])

    if rotation is None:
        ramifier = Ramifier(kmer_len)
    else:
        ramifier = RotatingRamifier.from_file(rotation)

    def rc_lev(row):
        s1, s2 = row['k1'], reverseComplement(row['k2'])
        return py_needle([s1, s2])[0][2]
    dist_tbl['rc_lev'] = dist_tbl.apply(rc_lev, axis=1)
    dist_tbl['lev'] = dist_tbl.apply(lambda row: min(row['f_lev'], row['rc_lev']), axis=1)

    def ram_dist(row):
        r1, r2 = ramifier.ramify(row['k1']), ramifier.ramify(row['k2'])
        return np.abs(r1 - r2).sum()
    dist_tbl['ram'] = dist_tbl.apply(ram_dist, axis=1)
    dist_tbl.to_csv(outfile)


@stats_cli.command('dump-contigs')
@click.option('-s/-n', '--seq/--no-seq', default=False)
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('contig_db', type=click.Path())
def cli_dump_contigs(seq, outfile, contig_db):
    grid = ContigDB.load_from_filepath(contig_db)
    for cid, kmer, genome_name, contig_name, contig_coord in grid.get_all_contigs():
        if not seq:
            kmer = ''
        print(f'{cid} {genome_name} {contig_name} {contig_coord} {kmer}', file=outfile)


@stats_cli.command('cover-stats')
@click.argument('grid_cover', type=click.Path())
def cli_dump_kmers(grid_cover):
    click.echo(grid_cover)
    grid = GridCoverDB.load_from_filepath(grid_cover)
    n_centers = grid.centroids().shape[0]
    click.echo(f'centers\t{n_centers}')
    n_kmers = len(grid.get_kmers())
    click.echo(f'kmers\t{n_kmers}')
    box_side = grid.box_side_len
    click.echo(f'box_side\t{box_side}')
    dims = grid.ramifier.d
    click.echo(f'dims\t{dims}')


@stats_cli.command('cluster-sizes')
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('grid_cover', type=click.Path())
def cli_dump_kmers(outfile, grid_cover):
    grid = GridCoverDB.load_from_filepath(grid_cover)
    counts = {}
    for centroid_index, _ in grid.get_kmers():
        counts[centroid_index] = 1 + counts.get(centroid_index, 0)
    for centroid_index, count in counts.items():
        print(f'{centroid_index},{count}', file=outfile)


@stats_cli.command('dump-kmers')
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.option('-c/-k', '--cluster-ids/--kmers', default=False)
@click.argument('grid_cover', type=click.Path())
def cli_dump_kmers(outfile, cluster_ids, grid_cover):
    grid = GridCoverDB.load_from_filepath(grid_cover)
    for centroid_index, kmer in grid.get_kmers():
        if cluster_ids:
            print(f'{centroid_index},{kmer}', file=outfile)
        else:
            print(kmer, file=outfile)


@stats_cli.command('dump-centroids')
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('grid_cover', type=click.Path())
def cli_dump_kmers(outfile, grid_cover):
    grid = GridCoverDB.load_from_filepath(grid_cover)
    pd.DataFrame(grid.centroids()).to_csv(outfile, header=None, index=None)
