"""Make a table of pairwise distances."""
import pandas as pd
import numpy as np
import click

from ariesk.ram import RotatingRamifier, StatisticalRam
from gimmebio.kmers import make_kmers

from ariesk.utils import py_needle, py_encode_kmer
from .utils import Buildable, make_kmers


class DistanceBuilder(Buildable):

    def __init__(self, queries, targets, **kwargs):
        self.k = kwargs['k']
        self.sub_ks = kwargs['sub_ks']
        self.rft_dims = kwargs['rft_dims']
        stat_ram = StatisticalRam(self.k, len(targets))
        stat_ram.bulk_add_kmers(targets)
        self.rotation = {
            'center': stat_ram.get_centers(),
            'scale': stat_ram.get_scales(),
            'rotation': stat_ram.get_rotation(),
        }
        self.queries = queries
        self.targets = targets

    def _build(self, outfile):
        self.tbl = self.needle_dists()
        click.echo('\tBuilt Needle Dists...')
        self.ram_dists()
        click.echo('\tBuilt Ram Dists...')
        self.subk_dists()
        click.echo('\tBuilt SubK Dists...')
        self.tbl.to_csv(outfile)

    def subk_dists(self):
        for sub_k in self.sub_ks:

            def sub_k_dist(row):
                q_ks = make_kmers(row['query'], sub_k)
                t_ks = make_kmers(row['target'], sub_k)
                return len(q_ks & t_ks) / len(q_ks)

            self.tbl[f'subk_{sub_k}'] = self.tbl.apply(sub_k_dist, axis=1)

    def ram_dists(self):
        for rft_dim in self.rft_dims:
            ramifier = RotatingRamifier(
                self.k, rft_dim,
                self.rotation['rotation'],
                self.rotation['center'],
                self.rotation['scale']
            )
            euc, manhattan = [], []
            for i, row in self.tbl.iterrows():
                q_rft = ramifier.ramify(row['query'])
                t_rft = ramifier.ramify(row['target'])
                euc.append(np.linalg.norm(q_rft - t_rft, ord=2))
                manhattan.append(np.linalg.norm(q_rft - t_rft, ord=1))
            self.tbl[f'ram_{rft_dim}_euc'] = euc
            self.tbl[f'ram_{rft_dim}_manhattan'] = manhattan

    def needle_dists(self):
        tbl = []
        for qseq in self.queries:
            qseq = qseq[:self.k]
            for tseq in self.targets:
                tseq = tseq[:self.k]
                tbl += py_needle([qseq, tseq])
        return pd.DataFrame(tbl, columns=['query', 'target', 'needle'])

