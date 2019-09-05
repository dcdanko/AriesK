
import numpy as np
cimport numpy as npc

from .ramft import build_rs_matrix


cdef long convert_kmer(str kmer):
    cdef long out = 0
    for i, base in enumerate(kmer):
        val = 0
        if base == 'C':
            val = 1
        elif base == 'G':
            val = 2
        elif base == 'T':
            val = 3
        out += val + (4 ** i) 
    return out


cdef class RftKdTree:
    cdef public float radius
    cdef public long k, num_kmers_added, max_size, rft_dims
    cdef public long [:] kmers
    cdef public double [:, :] rfts
    cdef public double [:, :] rs_matrix

    def __cinit__(self, radius, k, max_size):
        self.radius = radius
        self.k = k
        self.rft_dims = 12
        self.max_size = max_size
        self.num_kmers_added = 0

        self.rs_matrix = build_rs_matrix(self.k)

        self.kmers = npc.ndarray((self.max_size,), dtype=long)
        self.rfts = npc.ndarray((self.max_size, self.rft_dims))

    def  _ramify(self, int index, str kmer):
        cdef long [:, :] binary_kmer = np.array([
            [1 if base == seqb else 0 for seqb in kmer]
            for base in 'ACGT'
        ]).T
        cdef double [:, :] rft = abs(np.dot(self.rs_matrix, binary_kmer))
        cdef double [:] power_series = np.sum(rft, axis=1)
        self.rfts[self.num_kmers_added] = power_series[1:(1 + self.rft_dims)]

    def add_kmer(self, str kmer):
        assert self.num_kmers_added < self.max_size
        cdef long kmer_code = convert_kmer(kmer)
        self.kmers[self.num_kmers_added] = kmer_code
        self._ramify(self.num_kmers_added, kmer)
        self.num_kmers_added += 1

