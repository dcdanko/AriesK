# cython: profile=False
# cython: linetrace=False
# cython: language_level=3

import sqlite3
import numpy as np
cimport numpy as npc
from libc.stdio cimport *
from posix.stdio cimport *  # FILE, fopen, fclose
from libc.stdlib cimport malloc, free
from math import ceil
from libc.math cimport floor
from ariesk.ckdtree cimport cKDTree

from ariesk.ram cimport StatisticalRam, RotatingRamifier

from ariesk.utils.kmers cimport encode_kmer, needle_dist

cdef class DistMatrixBuilder:
    cdef public npc.uint8_t[:, :] kmers
    cdef public double[:, :] rfts
    cdef public cKDTree tree

    def __cinit__(self, kmers, d, k):
        self.kmers = np.ndarray((len(kmers), k), dtype=np.uint8)
        self.rfts = np.ndarray((len(kmers), d))
        cdef int i, j
        cdef double[:] rft
        cdef npc.uint8_t[:] kmer

        print('building matrixer')
        for i in range(self.kmers.shape[0]):
            kmer = encode_kmer(kmers[i])
            for j in range(k):
                self.kmers[i, j] = kmer[j]
        print('encoded kmers')
        cdef StatisticalRam stat_ram = StatisticalRam(k, self.kmers.shape[0])
        for i in range(self.kmers.shape[0]):
            stat_ram.c_add_kmer(self.kmers[i,:])
        print('reduced dims')
        cdef RotatingRamifier ramifier = RotatingRamifier(
            k, d, stat_ram.get_rotation(), stat_ram.get_centers(), stat_ram.get_scales()
        )
        print('built ramifier')
        for i in range(self.kmers.shape[0]):
            rft = ramifier.c_ramify(self.kmers[i, :])
            for j in range(d):
                self.rfts[i, j] = rft[j]
        self.tree = cKDTree(self.rfts)
        print('built tree')

    def build(self, double radius):
        cdef list results = []
        cdef double dist
        cdef int i = 0
        for hit_list in self.tree.query_ball_tree(self.tree, radius):
            for hit in hit_list:
                if i < hit:
                    dist = needle_dist(self.kmers[i,:], self.kmers[hit, :], False)
                    results.append((i, hit, dist))
            i += 1
        return results
