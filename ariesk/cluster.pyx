
import numpy as np
cimport numpy as npc

from ariesk.bloom_filter cimport BloomFilter


cdef class Cluster:

    def __cinit__(self, centroid_id, seqs, sub_k=5):
        self.centroid_id = centroid_id
        self.seqs = seqs
        self.sub_k = sub_k

    cdef build_bloom_filter(self):
        cdef int n_subk = self.seqs.shape[0] * (self.seqs.shape[1] - self.sub_k + 1)
        self.bloom_filter = BloomFilter(5, n_subk, 0.01)
        cdef int i, j
        for i in range(self.seqs.shape[0]):
            for j in range(self.seqs.shape[1] - self.sub_k + 1):
                self.bloom_filter.add(self.seqs[i, j:j + 5])

    cdef bint test_membership(self, npc.uint8_t[:] query_seq, int allowed_misses):
        cdef int i, n_hits
        cdef int min_hits = self.seqs.shape[1] - self.sub_k + 1 - (self.sub_k * allowed_misses)
        for i in range(self.seqs.shape[1] - self.sub_k + 1):
            if self.bloom_filter.contains(query_seq[i:i + self.sub_k]):
                n_hits += 1
        return n_hits >= min_hits
