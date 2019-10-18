
import click
from time import time
from json import dumps, loads
from shutil import copyfile
from os.path import isfile
from os import environ
from sys import stderr
import sqlite3

from ariesk.ram import (
    StatisticalRam,
    RotatingRamifier,
)
from ariesk.dbs.contig_db import ContigDB
from ariesk.dbs.pre_contig_db import PreContigDB

from ariesk.pre_db import PreDB
from ariesk.utils.parallel_build import coordinate_parallel_build


@click.group('contig')
def build_contig_cli():
    pass


@build_contig_cli.command('from-fasta')
@click.option('-r', '--radius', default=0.01, type=float)
@click.option('-d', '--dimension', default=8)
@click.option('-t', '--threads', default=1)
@click.option('-o', '--outfile', default='ariesk_grid_cover_db.sqlite', type=click.Path())
@click.argument('rotation', type=click.Path())
@click.argument('fasta_list', type=click.File('r'))
def build_contig_cover_fasta(radius, dimension, threads, outfile, rotation, fasta_list):
    environ['OPENBLAS_NUM_THREADS'] = f'{threads}'  # numpy uses one of these two libraries
    environ['MKL_NUM_THREADS'] = f'{threads}'
    fasta_list = [line.strip() for line in fasta_list]
    ramifier = RotatingRamifier.from_file(dimension, rotation)
    grid = ContigDB(
        sqlite3.connect(outfile), ramifier=ramifier, box_side_len=radius
    )
    click.echo(f'Adding {len(fasta_list)} fastas.', err=True)
    start = time()
    with click.progressbar(fasta_list) as fastas:
        for fasta_filename in fastas:
            n_added = grid.fast_add_kmers_from_fasta(fasta_filename)
    grid.close()
    add_time = time() - start
    click.echo(
        f'Added {n_added:,} kmers to {outfile} in {add_time:.5}s. ',
        err=True
    )


@build_contig_cli.command('from-pre')
@click.option('-r', '--radius', default=0.01, type=float)
@click.option('-t', '--threads', default=1)
@click.option('-o', '--outfile', default='ariesk_grid_cover_db.sqlite', type=click.Path())
@click.argument('pre_list', type=click.File('r'))
def build_contig_from_pre(radius, threads, outfile, pre_list):
    environ['OPENBLAS_NUM_THREADS'] = f'{threads}'  # numpy uses one of these two libraries
    environ['MKL_NUM_THREADS'] = f'{threads}'
    pre_list = [line.strip() for line in pre_list]
    click.echo(f'Adding {len(pre_list)} predbs.', err=True)
    start = time()
    predb = PreContigDB.load_from_filepath(pre_list[0])
    grid = ContigDB.from_predb(outfile, predb, radius)
    grid._drop_indices()
    with click.progressbar(pre_list) as pres:
        for i, predb_filename in enumerate(pres):
            if i > 0:
                grid.add_from_predb(PreContigDB.load_from_filepath(predb_filename))
    grid.commit()
    grid._build_indices()
    grid.close()
    add_time = time() - start
    click.echo(
        f'Added predbs to {outfile} in {add_time:.5}s. ',
        err=True
    )


@build_contig_cli.command('pre')
@click.option('-d', '--dimension', default=8)
@click.option('-t', '--threads', default=1)
@click.option('-o', '--outfile', default='ariesk_precover_db.sqlite', type=click.Path())
@click.argument('rotation', type=click.Path())
@click.argument('fasta_list', type=click.File('r'))
def build_precontig_cover_fasta(dimension, threads, outfile, rotation, fasta_list):
    environ['OPENBLAS_NUM_THREADS'] = f'{threads}'  # numpy uses one of these two libraries
    environ['MKL_NUM_THREADS'] = f'{threads}'
    fasta_list = [line.strip() for line in fasta_list]
    ramifier = RotatingRamifier.from_file(dimension, rotation)
    grid = PreContigDB(
        sqlite3.connect(outfile), ramifier=ramifier
    )
    grid._drop_indices()
    click.echo(f'Adding {len(fasta_list)} fastas.', err=True)
    start = time()
    with click.progressbar(fasta_list) as fastas:
        for fasta_filename in fastas:
            n_added = grid.fast_add_kmers_from_fasta(fasta_filename)
    grid.close()
    grid._build_indices()
    add_time = time() - start
    click.echo(
        f'Added {n_added:,} kmers to {outfile} in {add_time:.5}s. ',
        err=True
    )


@build_contig_cli.command('merge')
@click.argument('contig_dbs', nargs=-1)
def merge_contig_dbs(contig_dbs):
    main_db = ContigDB.load_from_filepath(contig_dbs[0])
    start = time()
    with click.progressbar(contig_dbs[1:]) as dbs:
        for filename in dbs:
            main_db.load_other(
                ContigDB.load_from_filepath(filename),
                rebuild_indices=False
            )
    main_db._build_indices()
    main_db.close()
    add_time = time() - start
    click.echo(
        f'Merged {len(contig_dbs)} dbs to {contig_dbs[0]} in {add_time:.5}s. ',
        err=True
    )
