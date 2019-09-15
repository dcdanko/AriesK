
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


cdef double needle_dist(k1, k2):
    cdef double [:, :] score = np.zeros((len(k1) + 1, len(k2) + 1))
    cdef double match_score = 0
    cdef double mismatch_penalty = 1.5
    cdef double gap_penalty = 1.6
    for i in range(len(k1) + 1):
        score[i][0] = gap_penalty * i
    for j in range(len(k2) + 1):
        score[0][j] = gap_penalty * j

    def _match_score(b1, b2):
        return match_score if b1 == b2 else mismatch_penalty

    for i in range(1, len(k1) + 1):
        for j in range(1, len(k2) + 1):
            match = score[i - 1][j - 1] + _match_score(k1[i - 1], k2[j - 1])
            delete = score[i - 1][j] + gap_penalty
            insert = score[i][j - 1] + gap_penalty
            score[i][j] = min(match, delete, insert)
    final_score = score[len(k1)][len(k2)]

    return final_score / len(k1)


cdef class KmerAddable:

    cpdef add_kmer(self, str kmer):
        raise NotImplementedError()

    def bulk_add_kmers(self, kmers):
        for kmer in kmers:
            self.add_kmer(kmer)

    def add_kmers_from_file(self, str filename, sep=',', start=0, num_to_add=0):
        with open(filename) as f:
            n_added = 0
            for i, line in enumerate(f):
                if i < start:
                    continue
                n_added += 1
                if num_to_add > 0 and n_added > num_to_add:
                    break
                kmer = line.split(sep)[0]
                self.add_kmer(kmer)
