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


@stats_cli.command('dump-kmers')
@click.argument('grid_cover', type=click.File('r'))
def cli_dump_kmers(grid_cover):
    grid = GridCoverSearcher.from_dict(loads(grid_cover.read()))
    for kmer in grid.kmers:
        click.echo(py_reverse_convert_kmer(kmer))
