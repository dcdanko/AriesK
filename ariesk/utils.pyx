
import numpy as np
cimport numpy as npc


def py_convert_kmer(kmer):
    return convert_kmer(kmer, len(kmer))

def py_reverse_convert_kmer(kmer):
    return reverse_convert_kmer(kmer)


cdef long [:] convert_kmer(str kmer, int k):
    cdef long [:] encoded = np.ndarray((k,), dtype=long)
    for i, base in enumerate(kmer):
        val = 0
        if base == 'C':
            val = 1
        elif base == 'G':
            val = 2
        elif base == 'T':
            val = 3
        elif base == 'A':
            val = 4
        encoded[i] = val
    return encoded


cdef str reverse_convert_kmer(long [:] encoded):
    cdef str out = ''
    for code in encoded:
        if code == 0:
            out += 'N'
        elif code == 1:
            out += 'C'
        elif code == 2:
            out += 'G'
        elif code == 3:
            out += 'T'
        else:
            out += 'A'
    return out


cdef class KmerAddable:

    cpdef add_kmer(self, str kmer):
        raise NotImplementedError()

    def bulk_add_kmers(self, kmers):
        for kmer in kmers:
            self.add_kmer(kmer)

    def add_kmers_from_file(self, str filename, sep=',', start=0):
        with open(filename) as f:
            for i, line in enumerate(f):
                if i < start:
                    continue
                if (i - start) >= self.max_size:
                    break
                kmer = line.split(sep)[0]
                self.add_kmer(kmer)
