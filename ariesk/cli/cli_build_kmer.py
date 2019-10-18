
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
from ariesk.grid_builder import GridCoverBuilder
from ariesk.grid_searcher import GridCoverSearcher
from ariesk.dbs.kmer_db import GridCoverDB

from ariesk.pre_db import PreDB
from ariesk.utils.parallel_build import coordinate_parallel_build


@click.group('kmer')
def build_kmer_cli():
    pass


@build_kmer_cli.command('grid')
@click.option('-r', '--radius', default=0.02, type=float)
@click.option('-d', '--dimension', default=8)
@click.option('-t', '--threads', default=1)
@click.option('-n', '--num-kmers', default=0, help='Number of kmers to cluster.')
@click.option('-s', '--start-offset', default=0)
@click.option('-o', '--outfile', default='ariesk_grid_cover_db.sqlite', type=click.Path())
@click.option('--preload/--no-preload', default=False, help='Load k-mers into RAM before processing')
@click.argument('rotation', type=click.Path())
@click.argument('kmer_table', type=click.Path())
def build_grid_cover(radius, dimension, threads, num_kmers, start_offset, outfile, preload, rotation, kmer_table):
    environ['OPENBLAS_NUM_THREADS'] = f'{threads}'  # numpy uses one of these two libraries
    environ['MKL_NUM_THREADS'] = f'{threads}'
    ramifier = RotatingRamifier.from_file(dimension, rotation)
    grid = GridCoverBuilder.from_filepath(outfile, ramifier, radius)
    start = time()
    n_added = grid.fast_add_kmers_from_file(kmer_table, num_to_add=num_kmers)
    grid.commit()
    n_centers = grid.db.centroids().shape[0]
    grid.close()
    add_time = time() - start
    click.echo(f'Added {n_added:,} kmers to {outfile} in {add_time:.5}s. {n_centers:,} clusters.', err=True)


@build_kmer_cli.command('pre-grid-fasta')
@click.option('-d', '--dimension', default=8)
@click.option('-t', '--threads', default=1)
@click.option('-o', '--outfile', default='ariesk_pre_grid_db.sqlite', type=click.Path())
@click.argument('rotation', type=click.Path())
@click.argument('fasta_list', type=click.File('r'))
def build_grid_cover_fasta(dimension, threads, outfile, rotation, fasta_list):
    environ['OPENBLAS_NUM_THREADS'] = f'{threads}'  # numpy uses one of these two libraries
    environ['MKL_NUM_THREADS'] = f'{threads}'
    fasta_list = [line.strip() for line in fasta_list]
    ramifier = RotatingRamifier.from_file(dimension, rotation)
    predb = PreDB.load_from_filepath(outfile, ramifier=ramifier)
    start = time()
    with click.progressbar(fasta_list) as fastas:
        for fasta_filename in fastas:
            n_added = predb.fast_add_kmers_from_fasta(fasta_filename)
    predb.close()
    add_time = time() - start
    click.echo(
        f'Added {n_added:,} kmers to {outfile} in {add_time:.5}s.',
        err=True
    )


@build_kmer_cli.command('grid-fasta')
@click.option('-r', '--radius', default=0.02, type=float)
@click.option('-d', '--dimension', default=8)
@click.option('-t', '--threads', default=1)
@click.option('-o', '--outfile', default='ariesk_grid_cover_db.sqlite', type=click.Path())
@click.argument('rotation', type=click.Path())
@click.argument('fasta_list', type=click.File('r'))
def build_grid_cover_fasta(radius, dimension, threads, outfile, rotation, fasta_list):
    environ['OPENBLAS_NUM_THREADS'] = f'{threads}'  # numpy uses one of these two libraries
    environ['MKL_NUM_THREADS'] = f'{threads}'
    fasta_list = [line.strip() for line in fasta_list]
    ramifier = RotatingRamifier.from_file(dimension, rotation)
    grid = GridCoverBuilder.from_filepath(outfile, ramifier, radius)
    start = time()
    with click.progressbar(fasta_list) as fastas:
        for fasta_filename in fastas:
            n_added = grid.fast_add_kmers_from_fasta(fasta_filename)
    n_centers = grid.db.centroids().shape[0]
    grid.close()
    add_time = time() - start
    click.echo(
        (
            f'Added {n_added:,} kmers to {outfile} in {add_time:.5}s. '
            f'{n_centers:,} clusters.'
        ),
        err=True
    )


@build_kmer_cli.command('prebuild-blooms')
@click.argument('grid_db', type=click.Path())
def build_grid_cover(grid_db):
    db = GridCoverDB.load_from_filepath(grid_db)
    start = time()
    n_centers = db.centroids().shape[0]
    with click.progressbar(list(range(n_centers))) as centroid_ids:
        for centroid_id in centroid_ids:
            db.build_and_store_bloom_grid(centroid_id)
    db.close()
    add_time = time() - start
    click.echo(f'Built {n_centers} bloom filters in {add_time:.5}s.', err=True)


@build_kmer_cli.command('grid-parallel')
@click.option('-r', '--radius', default=0.02, type=float)
@click.option('-d', '--dimension', default=8)
@click.option('-t', '--threads', default=1)
@click.option('-c', '--chunk-size', default=100 * 1000)
@click.option('-n', '--num-kmers', default=0, help='Number of kmers to cluster.')
@click.option('-s', '--start-offset', default=0)
@click.option('-o', '--outfile', default='ariesk_grid_cover_db.sqlite', type=click.Path())
@click.argument('rotation', type=click.Path())
@click.argument('kmer_table', type=click.Path())
def build_grid_cover(radius, dimension, threads, chunk_size, num_kmers,
                     start_offset, outfile, rotation, kmer_table):
    def logger(num, total):
        click.echo(f'Finished {num + 1} chunks of {total}', err=True)
        if num + 1 == total:
            click.echo('Merging...', err=True)

    environ['OPENBLAS_NUM_THREADS'] = '2'  # numpy uses one of these two libraries
    environ['MKL_NUM_THREADS'] = '2'
    start = time()
    coordinate_parallel_build(
        outfile, kmer_table, rotation,
        (3 * threads) // 4, start_offset, num_kmers, radius, dimension,
        chunk_size=chunk_size, logger=logger
    )
    elapsed = time() - start
    click.echo(f'Built grid cover in {elapsed:.5}s.', err=True)


@build_kmer_cli.command('grid-merge')
@click.argument('final_db', type=click.Path())
@click.argument('other_dbs', type=click.Path(), nargs=-1)
def merge_grid_cover(final_db, other_dbs):
    if not isfile(final_db):
        copyfile(other_dbs[0], final_db)
        other_dbs = other_dbs[1:]
    final_db = GridCoverDB.load_from_filepath(final_db)
    for other_db_filename in other_dbs:
        other_db = GridCoverDB.load_from_filepath(other_db_filename)
        final_db.load_other(other_db)


@build_kmer_cli.command('grid-from-pre')
@click.option('-r', '--radius', default=0.02, type=float)
@click.option('-d', '--dimension', default=8)
@click.option('-t', '--threads', default=1)
@click.option('-o', '--outfile', default='ariesk_grid_cover_db.sqlite', type=click.Path())
@click.argument('predb_list', type=click.File('r'))
def build_grid_cover_fasta(radius, dimension, threads, outfile, predb_list):
    environ['OPENBLAS_NUM_THREADS'] = f'{threads}'  # numpy uses one of these two libraries
    environ['MKL_NUM_THREADS'] = f'{threads}'
    predb_list = [line.strip() for line in predb_list]
    logger = lambda n: stderr.write(f'\rAdded {n:,} k-mers to db')
    start = time()
    predb = PreDB.load_from_filepath(predb_list[0])
    grid = GridCoverBuilder.build_from_predb(outfile, predb, radius, logger=logger)
    with click.progressbar(predb_list) as predbs:
        for i, predb_filename in enumerate(predbs):
            if i == 0:
                continue
            predb = PreDB.load_from_filepath(predb_filename)
            n_added = grid.add_kmers_from_predb(predb, logger=logger)
    grid.db._build_indices()  # indices are disabled by `GCB.build_from_predb`
    n_centers = grid.db.centroids().shape[0]
    grid.close()
    add_time = time() - start
    click.echo(
        (
            f'Added {n_added:,} kmers to {outfile} in {add_time:.5}s. '
            f'{n_centers:,} clusters.'
        ),
        err=True
    )
