
import numpy as np


def py_convert_kmer(kmer):
    return convert_kmer(kmer)

def py_reverse_convert_kmer(kmer):
    return reverse_convert_kmer(kmer)

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
        out += val * (4 ** i) 
    return out


cdef str reverse_convert_kmer(long kmer):
    cdef str base4 = np.base_repr(kmer, base=4)
    cdef str out = ''
    for code in base4[::-1]:
        if code == '0':
            out += 'A'
        elif code == '1':
            out += 'C'
        elif code == '2':
            out += 'G'
        else:
            out += 'T'
    return out


cdef class KmerAddable:

    cpdef add_kmer(self, str kmer):
        raise NotImplementedError()

    def bulk_add_kmers(self, kmers):
        for kmer in kmers:
            self.add_kmer(kmer)

    def add_kmers_from_file(self, str filename, sep=',', start=0):
        with open(filename) as f:
            if start:
                f.readlines(start)
            for i, line in enumerate(f):
                if i >= self.max_size:
                    break
                kmer = line.split(sep)[0]
                self.add_kmer(kmer)
