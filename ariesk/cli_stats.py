import click
import pandas as pd
import numpy as np

from json import dumps, loads
from time import clock

from ariesk.searcher import GridCoverSearcher
from ariesk.db import GridCoverDB
from .utils import py_reverse_convert_kmer


@click.group('stats')
def stats_cli():
    pass


@stats_cli.command('cover-stats')
@click.argument('grid_cover', type=click.Path())
def cli_dump_kmers(grid_cover):
    grid = GridCoverDB.load_from_filepath(grid_cover)
    n_centers = grid.centroids().shape[0]
    click.echo(f'centers\t{n_centers}')
    n_kmers = len(grid.get_kmers())
    click.echo(f'kmers\t{n_kmers}')
    box_side = grid.box_side_len
    click.echo(f'box_side\t{box_side}')
    dims = grid.ramifier.d
    click.echo(f'dims\t{dims}')


@stats_cli.command('dump-kmers')
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('grid_cover', type=click.Path())
def cli_dump_kmers(outfile, grid_cover):
    grid = GridCoverDB.load_from_filepath(grid_cover)
    for centroid_index, kmer in grid.get_kmers():
        print(kmer, file=outfile)


@stats_cli.command('dump-centroids')
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('grid_cover', type=click.Path())
def cli_dump_kmers(outfile, grid_cover):
    grid = GridCoverDB.load_from_filepath(grid_cover)
    pd.DataFrame(grid.centroids()).to_csv(outfile, header=None, index=None)
