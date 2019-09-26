
from os.path import isfile
from random import random, choice

import warnings
warnings.filterwarnings('ignore')


class Buildable:

    def _build(self, outfile):
        raise NotImplementedError()

    def build(self, outfile):
        if not isinstance(outfile, str):
            outfile = outfile.name
        if not isfile(outfile):
            self._build(outfile)


def make_kmers(seq, k):
    out = set()
    for i in range(len(seq) - k + 1):
        out.add(seq[i:i + k])
    return out


def mutate_seq(kmer, sub_rate, indel_rate):
    out = ''
    for base in kmer:
        if random() < indel_rate:
            if random() < 0.5:  # deletion
                continue
            else:
                out += choice('ATCG')
        elif random() < (indel_rate + sub_rate):
            out += choice('ATCG')
        else:
            out += base
    while len(out) < len(kmer):
        out += choice('ATCG')
    return out[:len(kmer)]


def sample_fasta(seqs, k, n):
    kmers = set()
    while len(kmers) < n:
        for seq in seqs:
            for kmer in make_kmers(seq, k):
                if random() < 0.0001:
                    kmers.add(str(kmer))
                if len(kmers) >= n:
                    break
            if len(kmers) >= n:
                break
    return list(kmers)
