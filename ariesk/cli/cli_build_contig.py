
import click
import pandas as pd
import numpy as np
import random
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
from ariesk.utils.kmers import py_needle, py_needle_2
from Bio import SeqIO


@click.group('contig')
def build_contig_cli():
    pass


@build_contig_cli.command('from-fasta')
@click.option('-r', '--radius', default=0.01, type=float)
@click.option('-d', '--dimension', default=8)
@click.option('-t', '--threads', default=1)
@click.option('-o', '--outfile', default='ariesk_contig_cover_db.sqlite', type=click.Path())
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


def select_one_kmer(seq, k):
    start = random.randint(0, len(seq) - k)
    return seq[start:start + k]


def mutate_seq(seq, k, max_rate=0.4):
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


@build_contig_cli.command('calibrate')
@click.option('-n', '--num-seqs', default=100)
@click.option('-m', '--num-mutants', default=1)
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('database', type=click.Path())
def calibrate_db(num_seqs, num_mutants, outfile, database):
    db = ContigDB.load_from_filepath(database)
    click.echo(f'K: {db.ramifier.k}', err=True)
    prek = int(db.ramifier.k * 1.1)
    contigs = random.sample(db.get_all_contigs(), num_seqs)
    contigs = [
        db.py_get_seq(contig_name, start_coord, start_coord + prek + 100)
        for contig_name, _, start_coord, end_coord in contigs
    ]
    contigs = [select_one_kmer(seq, prek) for seq in contigs if len(seq) > prek]
    click.echo(f'Total contigs: {len(contigs)}', err=True)
    mutated = [mutate_seq(seq, db.ramifier.k) for seq in contigs for _ in range(num_mutants)]
    contigs = [select_one_kmer(kmer, db.ramifier.k) for kmer in contigs] + mutated
    click.echo(f'Comparisons: {(len(contigs) ** 2) / 2 - len(contigs)}', err=True)
    dist_tbl = pd.DataFrame(py_needle(contigs), columns=['k1', 'k2', 'levenshtein'])

    def ram_dist(row):
        r1, r2 = db.ramifier.ramify(row['k1']), db.ramifier.ramify(row['k2'])
        return np.abs(r1 - r2).sum()
    dist_tbl['ram'] = dist_tbl.apply(ram_dist, axis=1)
    dist_tbl.to_csv(outfile)


@build_contig_cli.command('probe-calibrate')
@click.option('-n', '--num-seqs', default=100)
@click.option('-c', '--contig-multiplier', default=1)
@click.option('-p', '--probe-multiplier', default=1)
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('probes', type=click.File('r'))
@click.argument('database', type=click.Path())
def probe_calibrate_db(num_seqs, contig_multiplier, probe_multiplier, outfile, probes, database):
    db = ContigDB.load_from_filepath(database)
    click.echo(f'K: {db.ramifier.k}', err=True)
    probes = [str(el.seq) for el in SeqIO.parse(probes, 'fasta')]
    probes = [
        select_one_kmer(seq, db.ramifier.k)
        for seq in probes for _ in range(probe_multiplier)
    ]
    contigs = random.sample(db.get_all_contigs(), num_seqs)
    contigs = [
        db.py_get_seq(contig_name, start_coord, end_coord)
        for contig_name, _, start_coord, end_coord in contigs
    ]
    contigs = [
        select_one_kmer(seq, db.ramifier.k)
        for seq in contigs for _ in range(contig_multiplier)
    ]
    click.echo(f'Comparisons: {len(contigs) * len(probes):,}', err=True)
    dist_tbl = py_needle_2(contigs, probes)
    dist_tbl = pd.DataFrame(dist_tbl, columns=['contig', 'probe', 'levenshtein'])

    def ram_dist(row):
        r1, r2 = db.ramifier.ramify(row['contig']), db.ramifier.ramify(row['probe'])
        return np.abs(r1 - r2).sum()
    dist_tbl['ram'] = dist_tbl.apply(ram_dist, axis=1)
    dist_tbl.to_csv(outfile)


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
