import click
import pandas as pd
import numpy as np

from json import dumps, loads
from time import clock

from ariesk.grid_searcher import GridCoverSearcher
from ariesk.dbs.kmer_db import GridCoverDB
from ariesk.dbs.contig_db import ContigDB

@click.group('stats')
def stats_cli():
    pass


@stats_cli.command('dump-contigs')
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('contig_db', type=click.Path())
def cli_dump_contigs(outfile, contig_db):
    grid = ContigDB.load_from_filepath(contig_db)
    for cid, kmer, genome_name, contig_name, contig_coord in grid.get_all_contigs():
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
