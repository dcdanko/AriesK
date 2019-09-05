
cimport numpy as np


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
    cdef public long k, num_kmers_added, max_size
    cdef public long [:] kmers

    def __cinit__(self, radius, k, max_size):
        self.radius = radius
        self.k = k
        self.max_size = max_size
        self.num_kmers_added = 0
        self.kmers = np.ndarray((self.max_size,), dtype=long)

    def add_kmer(self, str kmer):
        assert self.num_kmers_added < self.max_size
        cdef long kmer_code = convert_kmer(kmer)
        self.kmers[0] = kmer_code
        self.num_kmers_added += 1
