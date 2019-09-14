import click
import pandas as pd
import numpy as np

from json import dumps, loads
from time import clock

from ariesk.searcher import GridCoverSearcher
from .utils import py_reverse_convert_kmer


@click.group('stats')
def stats_cli():
    pass


@stats_cli.command('cover-stats')
@click.argument('grid_cover', type=click.File('r'))
def cli_dump_kmers(grid_cover):
    grid = GridCoverSearcher.from_dict(loads(grid_cover.read()))
    n_centers = len(grid.clusters.keys())
    click.echo(f'centers\t{n_centers}')
    n_kmers = sum([len(val) for val in grid.clusters.values()])
    click.echo(f'kmers\t{n_kmers}')
    box_side = grid.box_side_len
    click.echo(f'box_side\t{box_side}')
    radius = grid.radius
    click.echo(f'radius\t{radius}')
    dims = grid.ramifier.d
    click.echo(f'dims\t{dims}')


@stats_cli.command('dump-kmers')
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('grid_cover', type=click.File('r'))
def cli_dump_kmers(outfile, grid_cover):
    grid = GridCoverSearcher.from_dict(loads(grid_cover.read()))
    for kmer in grid.kmers:
        print(py_reverse_convert_kmer(kmer), file=outfile)


@stats_cli.command('dump-centroids')
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('grid_cover', type=click.File('r'))
def cli_dump_kmers(outfile, grid_cover):
    grid = GridCoverSearcher.from_dict(loads(grid_cover.read()))
    pd.DataFrame(grid.centroids()).to_csv(outfile, header=None, index=None)
