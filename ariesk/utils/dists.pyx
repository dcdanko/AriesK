# cython: profile=True
# cython: linetrace=True
# cython: language_level=3

import numpy as np
cimport numpy as npc

from .ramft import build_rs_matrix


cdef class DistanceFactory:
    cdef public long k
    cdef public double [:, :] rs_matrix

    def __cinit__(self, k):
        self.k = k
        self.rs_matrix = build_rs_matrix(self.k)

    cdef double hamming_dist(self, str k1, str k2):
        cdef double dist = 0.
        for b1, b2 in zip(k1, k2):
            if b1 != b2:
                dist += 1
        return dist

    cdef double subkmer_dist(self, str k1, str k2):
        cdef int subk = 4
        cdef double dist = 0.
        sub_k1 = {k1[i:i + subk] for i in range(self.k - subk + 1)}
        sub_k2 = {k2[i:i + subk] for i in range(self.k - subk + 1)}
        cdef double numer = len(sub_k1 & sub_k2)
        dist = 1 - (numer / len(sub_k1 | sub_k2))
        return dist

    cdef double rft_sum_euclid_dist(self, str k1, str k2):
        cdef long [:, :] binary_k1 = np.array([
            [1 if base == seqb else 0 for seqb in k1]
            for base in 'ACGT'
        ]).T
        cdef long [:, :] binary_k2 = np.array([
            [1 if base == seqb else 0 for seqb in k2]
            for base in 'ACGT'
        ]).T
        cdef double [:, :] rft1 = abs(np.dot(self.rs_matrix, binary_k1))
        cdef double [:, :] rft2 = abs(np.dot(self.rs_matrix, binary_k2))
        cdef double [:] power_series1 = np.sum(rft1, axis=1)
        cdef double [:] power_series2 = np.sum(rft2, axis=1)
        cdef double dist = 0.
        for v1, v2 in zip(power_series1, power_series2):
            dist += (v1 - v2) ** 2
        return dist ** (0.5)

    cdef double rft_concat_manhattan_dist(self, str k1, str k2):
        cdef long [:, :] binary_k1 = np.array([
            [1 if base == seqb else 0 for seqb in k1]
            for base in 'ACGT'
        ]).T
        cdef long [:, :] binary_k2 = np.array([
            [1 if base == seqb else 0 for seqb in k2]
            for base in 'ACGT'
        ]).T
        cdef npc.ndarray rft1 = abs(np.dot(self.rs_matrix, binary_k1))
        cdef npc.ndarray rft2 = abs(np.dot(self.rs_matrix, binary_k2))
        cdef npc.ndarray power_series1 = rft1.flatten()
        cdef npc.ndarray power_series2 = rft2.flatten()
        cdef double dist = 0.
        for v1, v2 in zip(power_series1, power_series2):
            dist += abs(v1 - v2)
        return dist

    cdef double needle_dist(self, k1, k2):
        cdef double [:, :] score = np.zeros((len(k1) + 1, len(k2) + 1))
        cdef double match_score = -1
        cdef double mismatch_penalty = 0.5
        cdef double gap_penalty = 0.6
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

        return final_score


    cpdef all_dists(self, str k1, str k2):
        return {
            'kmer_1': k1,
            'kmer_2': k2,
            'hamming': self.hamming_dist(k1, k2),
            'needle': self.needle_dist(k1, k2),
            'subkmer': self.subkmer_dist(k1, k2),
            'rft_sum_euclid': self.rft_sum_euclid_dist(k1, k2),
            'rft_concat_manhattan': self.rft_concat_manhattan_dist(k1, k2),
        }
