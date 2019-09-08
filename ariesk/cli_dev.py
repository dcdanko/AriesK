import click
import pandas as pd
import numpy as np

from time import clock

from ariesk.dists import DistanceFactory
from ariesk.ram import RotatingRamifier


@click.group('dev')
def dev_cli():
    pass


@dev_cli.command('dists')
@click.option('-k', '--kmer-len', default=31)
@click.option('-n', '--num-kmers', default=1000, help='Number of kmers to compare.')
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('kmer_table', type=click.File('r'))
def calculate_kmer_dists_cluster(kmer_len, num_kmers, outfile, kmer_table):
    dist_factory = DistanceFactory(kmer_len)
    kmers = []
    for i, line in enumerate(kmer_table):
        if i >= num_kmers:
            break
        kmers.append(line.strip().split(',')[0])
    tbl = []
    start = clock()
    for i, k1 in enumerate(kmers):
        for j, k2 in enumerate(kmers):
            if i < j:
                break
            tbl.append(dist_factory.all_dists(k1, k2))
    run_time = clock() - start
    click.echo(f'time: {run_time:.5}s', err=True)
    tbl = pd.DataFrame(tbl)
    tbl.to_csv(outfile)


@dev_cli.command('add-rotation-dists')
@click.option('-d', '--dimensions', default=8)
@click.option('-k', '--kmer-cols', nargs=2, default(1, 2))
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('rotation', type=click.Path())
@click.argument('dist_table', type=click.File('r'))
def add_rotation_dists(dimensions, kmer_cols, outfile, rotation, dist_table):
    """Add rotation distances to an existing distance table."""
    header = dist_table.readline().strip() + f',rotation_dist_{dimensions}\n'
    outfile.write(header)
    ramifier = RotatingRamifier.from_file(dimensions, rotation)
    for line in dist_table:
        line = line.strip()
        tkns = line.split(',')
        k1, k2 = tkns[kmer_cols[0]], tkns[kmer_cols[1]]
        rft1, rft2 = ramifier.ramify(k1), ramifier.ramify(k2)
        d = np.linalg.norm(rft1 - rft2)
        outfile.write(line + f',{d}\n')
